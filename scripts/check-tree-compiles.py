#!/usr/bin/env python3
"""Compile the translation units this change can break, before the push.

Row `ENG-PREFLIGHT-COMPILES`, spec `.agents/specs/preflight-compiles.md`,
issue #2401.

WHY THIS EXISTS. On 2026-08-31 `main` was pushed twice in a state that does not
compile, and `scripts/agent-preflight.sh` was green both times. `5263ac31f` was
a shell line continuation at the end of a `//` comment, which `-Wcomment` under
this tree's `-Werror` rejects, in a `tools/bench/` target that existed so the
probe could not rot and had never been built. `08fa2f5aa` bound
`MlaSharedSelection*` to `ForwardMlaAttentionBlock`'s `vt::Tensor*` parameter,
because two rows landing at the same time each appended a trailing optional
pointer (#2395). Preflight's 30 record checkers and 60 suites validate records,
prose, anchors and trailers against a tree, and every one of them passes on a
tree that does not build. CI compiles four ways and would have caught both, up
to two hours after the push, and both landed by a direct push to `main`.

WHAT IT DOES, in order, and it says each step in its own output:

  1. Resolves the changed paths: the committed range, the staged diff, the
     unstaged diff and the untracked files, each counted separately.
  2. If nothing in that set is a C++ source, a header or a build file, the
     affected set is PROVABLY empty. It says so in words with the range it
     read, and exits 0 without configuring anything. That is the arm that keeps
     this off the 30 of the last 60 commits on `main` that touch no C++.
  3. Configures a build directory for `compile_commands.json`. Measured 1.67 s
     and 14 MB on this tree. Nothing is compiled by the configure.
  4. If a header changed, runs `c++ -MM -MG` over every translation unit with
     that unit's own flags and inverts the result. This is the real
     preprocessor with the real include path, so a conditional include, a
     macro-formed path and a generated header resolve the way the compiler
     resolves them. Measured 15.9 s for 1218 units at -j8. Skipped when only
     `.cpp` files changed, because the affected set is then exactly those files.
  5. Compiles the affected set with the RECORDED command minus `-c` and `-o`,
     plus `-fsyntax-only`. The front end runs and `-Werror` applies; no object
     file is written, so this costs no disk and cannot collide with another
     agent's build directory.

WHAT THE EXIT CODES MEAN, and the third is never a pass:

  0  every translation unit in scope compiled, or the set was provably empty
  1  at least one did not compile, and the compiler's own message is printed
  2  CANNOT-VERIFY: this run could not take the measurement at all

UN-RUN IS NOT PASSING. A C++ source in scope that no target compiles in this
configuration is reported BY NAME and is not added to the compiled total, and a
run whose attempt count does not equal the affected count exits 2. A summary
that folds "was not compiled" into "compiled" reads a build that never ran as a
build that passed, which is exactly how `-k 0` handed to cmake instead of ninja
produced "681 of 685 tests failed" over a tree nothing had touched.

WHAT IT DOES NOT COVER. Read this beside the summary line, because a checker's
message defines what it enforces and no gate holds this text and that behaviour
together:

  * It does not LINK. An undefined symbol, a duplicate definition, a missing
    vtable, an ODR violation and an unregistered CTest target all pass here.
  * ONE configuration: the default host compiler and `-DCMAKE_BUILD_TYPE=
    Release`, matching the CI build lanes, with CUDA, HIP, Metal and MSVC off.
    A `.cu`, `.hip` or `.mm` unit is not in `compile_commands.json` and is
    reported as uncompiled rather than checked.
  * ONE compiler. A clang-only or gcc-15-only diagnostic stays CI's.
  * It RUNS nothing. A tree that compiles can fail every test in it.
  * Scope follows `#include` edges only. A unit a change reaches through a
    CMake option, a generated header written by a script the diff changed, or
    an embedded data file is not pulled in.
  * A build-file edit is covered for configure errors and for units the diff
    names. An edit that changes flags for a unit it does not name is not
    re-verified.
  * It reads the WORKING TREE, not each commit in the range. A range red at
    commit 3 and green at commit 5 reads green: the right answer for what is
    about to be pushed, the wrong one for a bisect.

Usage:
  scripts/check-tree-compiles.py [--base REF] [--head REF] [--jobs N]
                                 [--source-dir DIR] [--build-dir DIR]
                                 [--build-type TYPE] [--list]
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# A source is a translation unit CMake can compile. A header is a file that
# reaches one through `#include`, and is why step 4 exists at all. A build file
# changes what CMake configures. Anything else is out of scope by construction,
# and step 2 says so rather than implying it with silence.
SOURCE_SUFFIXES = {".cpp", ".cc", ".cxx", ".c", ".cu", ".mm", ".m"}
HEADER_SUFFIXES = {".h", ".hpp", ".hh", ".hxx", ".cuh", ".inc", ".ipp", ".tpp"}
BUILD_NAMES = {"CMakeLists.txt"}
BUILD_SUFFIXES = {".cmake"}

CANNOT_VERIFY = 2


class CannotVerify(Exception):
    """The measurement could not be taken. Never reported as a pass."""


def git(source: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(source), *args], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise CannotVerify(
            f"git {' '.join(args)} exited {result.returncode} in {source}: "
            f"{result.stderr.strip()}"
        )
    return result.stdout


def classify(paths: set[str]) -> tuple[list[str], list[str], list[str]]:
    sources, headers, builds = [], [], []
    for path in sorted(paths):
        name = os.path.basename(path)
        suffix = os.path.splitext(name)[1]
        if suffix in SOURCE_SUFFIXES:
            sources.append(path)
        elif suffix in HEADER_SUFFIXES:
            headers.append(path)
        elif name in BUILD_NAMES or suffix in BUILD_SUFFIXES:
            builds.append(path)
    return sources, headers, builds


def resolve_scope(source: Path, base: str, head: str) -> tuple[set[str], list[str]]:
    """The changed paths, and one narration line per component.

    The scope is a UNION of four components rather than the committed range
    alone. A base that moves forward past a branch's own commits narrows the
    committed component; it cannot empty a set that also carries the work in
    hand. Each count prints, so a zero is visible as a zero instead of being
    inferred from a verdict.
    """

    try:
        base_sha = git(source, "rev-parse", "--verify", "-q", f"{base}^{{commit}}").strip()
    except CannotVerify:
        base_sha = ""
    if not base_sha:
        raise CannotVerify(
            f"{base} does not resolve to a commit in {source}, so this run cannot "
            f"tell which files the change touched. Fetch the remote, or name the "
            f"base explicitly with --base. Unknown is not an empty range."
        )
    head_sha = git(source, "rev-parse", "--verify", f"{head}^{{commit}}").strip()

    committed = git(source, "diff", "--name-only", base_sha, head_sha).split()
    staged = git(source, "diff", "--cached", "--name-only").split()
    unstaged = git(source, "diff", "--name-only").split()
    untracked = git(source, "ls-files", "--others", "--exclude-standard").split()

    narration = [
        f"check-tree-compiles: base {base_sha[:12]} ({base})  head {head_sha[:12]} ({head})",
        f"  committed {base_sha[:12]}..{head_sha[:12]}: {len(committed)} path(s)",
        f"  staged: {len(staged)} path(s)",
        f"  unstaged: {len(unstaged)} path(s)",
        f"  untracked: {len(untracked)} path(s)",
    ]
    return set(committed) | set(staged) | set(unstaged) | set(untracked), narration


def configure(source: Path, build: Path, build_type: str) -> list[dict]:
    if shutil.which("cmake") is None:
        raise CannotVerify("cmake is not on PATH, so no build configuration can be read")
    build.mkdir(parents=True, exist_ok=True)
    argv = [
        "cmake",
        "-S",
        str(source),
        "-B",
        str(build),
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        f"-DCMAKE_BUILD_TYPE={build_type}",
    ]
    if shutil.which("ninja") is not None:
        argv[1:1] = ["-G", "Ninja"]
    result = subprocess.run(argv, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        tail = "\n".join((result.stdout + result.stderr).splitlines()[-25:])
        raise CannotVerify(
            "the CMake configure failed, so there are no compile commands to check "
            f"against. This is not a verdict on the code:\n{tail}"
        )
    database = build / "compile_commands.json"
    if not database.is_file():
        raise CannotVerify(
            f"{database} was not written by the configure, so this run has no flags "
            "to compile with. CMAKE_EXPORT_COMPILE_COMMANDS needs a Ninja or "
            "Makefile generator."
        )
    return json.loads(database.read_text(encoding="utf-8"))


def command_of(entry: dict) -> list[str]:
    if "arguments" in entry:
        return list(entry["arguments"])
    return shlex.split(entry["command"])


def strip_output(argv: list[str]) -> list[str]:
    """The recorded command with `-c` and `-o <file>` removed.

    The flags come from CMake and are never reconstructed here, so `-Werror`,
    `-ffp-contract=off`, every `-I` and every `-D` are the ones the build would
    use. A checker that assembled its own flags could drift from the build
    silently, and the first defect this row exists for is a `-Werror` warning.
    """

    out: list[str] = []
    index = 0
    while index < len(argv):
        if argv[index] == "-o":
            index += 2
            continue
        if argv[index] == "-c":
            index += 1
            continue
        out.append(argv[index])
        index += 1
    return out


def run_one(entry: dict, extra: list[str]) -> tuple[str, int, str]:
    argv = strip_output(command_of(entry))
    argv[1:1] = extra
    result = subprocess.run(
        argv, cwd=entry.get("directory", "."), capture_output=True, text=True, check=False
    )
    return entry["file"], result.returncode, (result.stdout + result.stderr)


def dependency_map(
    database: list[dict], jobs: int
) -> tuple[dict[str, set[str]], list[str]]:
    """header realpath -> the units that reach it, plus the units that would not scan.

    `-MM -MG` is the compiler's own preprocessor over the compiler's own
    include path, so this map is what the build sees and not what a textual
    `#include` scan guesses.

    A unit whose scan FAILS -- an unterminated `#if`, a bad include path -- has
    no edges, and treating "no edges" as "depends on nothing" would silently
    drop it from the closure of every header change. So the failures are
    returned, and the caller compiles them unconditionally: whatever the header
    did, that unit is checked, and the compiler gets to say what is wrong with
    it instead of this scan swallowing it.
    """

    reverse: dict[str, set[str]] = {}
    unscannable: list[str] = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        results = list(pool.map(lambda e: run_one(e, ["-MM", "-MG", "-E"]), database))
    for path, code, text in results:
        if code != 0:
            unscannable.append(path)
            continue
        body = text.replace("\\\n", " ")
        _, _, rhs = body.partition(":")
        for token in rhs.split():
            reverse.setdefault(os.path.realpath(token), set()).add(path)
    return reverse, sorted(unscannable)


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve()
    parser = argparse.ArgumentParser(
        description="Compile the translation units this change can break.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base", default="origin/main")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--source-dir", default=str(here.parents[1]))
    # A SCRATCH directory, and the default makes one. Pointed at an existing
    # build tree it reuses that tree's CMake cache, so a CUDA-on cache would put
    # `nvcc` entries in the database and this gate would hand them
    # `-fsyntax-only`. Preflight never passes it; only the suite does, with a
    # fresh directory per case.
    parser.add_argument("--build-dir", default="")
    parser.add_argument("--build-type", default="Release")
    parser.add_argument("--jobs", type=int, default=0)
    parser.add_argument(
        "--list",
        action="store_true",
        help="print the affected translation units and stop, compiling nothing",
    )
    args = parser.parse_args(argv)

    # Line-buffered deliberately. The steps below can take minutes on a wide
    # header change, and a gate that prints nothing until it finishes looks
    # hung to a reader and looks like a gate that never ran to a log.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):  # pragma: no cover - exotic stdout
        pass

    jobs = args.jobs or max(1, min(8, (os.cpu_count() or 2) // 2))
    source = Path(args.source_dir).resolve()

    try:
        changed, narration = resolve_scope(source, args.base, args.head)
    except CannotVerify as failure:
        print(f"CANNOT-VERIFY: {failure}")
        return CANNOT_VERIFY
    for line in narration:
        print(line)

    sources, headers, builds = classify(changed)
    print(
        f"  in scope: {len(sources)} C++ source(s), {len(headers)} header(s), "
        f"{len(builds)} build file(s), out of {len(changed)} changed path(s)"
    )
    if not (sources or headers or builds):
        print(
            "no translation unit is in scope: nothing this change touches is a C++ "
            "source, a header or a build file, so there is nothing to check. "
            "This is a derived empty set, not a skipped gate."
        )
        return 0

    scratch = None
    if args.build_dir:
        build = Path(args.build_dir).resolve()
    else:
        scratch = tempfile.mkdtemp(prefix="tree-compiles-")
        build = Path(scratch)
    try:
        started = time.time()
        try:
            database = configure(source, build, args.build_type)
        except CannotVerify as failure:
            print(f"CANNOT-VERIFY: {failure}")
            return CANNOT_VERIFY
        print(
            f"  configured {build} in {time.time() - started:.1f}s: "
            f"{len(database)} translation unit(s) in this configuration"
        )

        by_file = {os.path.realpath(entry["file"]): entry for entry in database}
        affected: set[str] = set()
        unbuilt: list[str] = []
        for relative in sources:
            absolute = os.path.realpath(source / relative)
            if absolute in by_file:
                affected.add(absolute)
            else:
                unbuilt.append(relative)

        if headers:
            started = time.time()
            reverse, unscannable = dependency_map(database, jobs)
            print(
                f"  dependency scan: {len(database)} unit(s) in "
                f"{time.time() - started:.1f}s (a header is in scope)"
            )
            for relative in headers:
                absolute = os.path.realpath(source / relative)
                for path in reverse.get(absolute, ()):
                    affected.add(os.path.realpath(path))
            if unscannable:
                # Forced in, not counted out. A unit the preprocessor could not
                # read has no edges, and a closure that reads that as "reaches
                # no header" is short by exactly the units nobody can vouch for.
                print(
                    f"  {len(unscannable)} unit(s) would not preprocess, so their "
                    "include edges are unknown and they are checked "
                    "unconditionally: "
                    + ", ".join(
                        os.path.relpath(path, source) for path in unscannable[:10]
                    )
                )
                for path in unscannable:
                    affected.add(os.path.realpath(path))

        if unbuilt:
            print(
                f"  {len(unbuilt)} C++ source(s) in scope that no target in this "
                f"configuration compiles, so nothing checked them: "
                + ", ".join(unbuilt)
            )

        entries = [by_file[path] for path in sorted(affected)]
        if args.list:
            for entry in entries:
                print(entry["file"])
            return 0
        if not entries:
            print(
                "check-tree-compiles: 0 of 0 translation unit(s) in scope needed "
                "checking; the configure succeeded and nothing this change touches "
                "reaches a unit this configuration builds"
            )
            return 0

        print(
            f"  checking {len(entries)} translation unit(s) with -fsyntax-only "
            f"at -j{jobs}"
        )
        started = time.time()
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(lambda e: run_one(e, ["-fsyntax-only"]), entries))
        elapsed = time.time() - started

        # The attempt count is asserted rather than assumed. A pool that dropped
        # work would otherwise report a clean run over units nobody compiled,
        # which is the failure this checker's own summary line exists to make
        # impossible to state.
        if len(results) != len(entries):
            print(
                f"CANNOT-VERIFY: {len(results)} of {len(entries)} translation "
                "unit(s) were actually attempted, so this run measured less than "
                "it was asked to. Un-run is not passing."
            )
            return CANNOT_VERIFY

        failures = [(path, text) for path, code, text in results if code != 0]
        for path, text in failures:
            print(f"FAILED to compile {os.path.relpath(path, source)}")
            for line in text.rstrip().splitlines():
                print(f"    {line}")
        if failures:
            print(
                f"check-tree-compiles: {len(failures)} of {len(entries)} translation "
                f"unit(s) in scope did not compile ({elapsed:.1f}s). This checks "
                "syntax and semantics only; it does not link, run, or leave the "
                "default host configuration."
            )
            return 1
        print(
            f"check-tree-compiles: {len(entries)} of {len(entries)} translation "
            f"unit(s) in scope compiled ({elapsed:.1f}s). This checks syntax and "
            "semantics only; it does not link, run, or leave the default host "
            "configuration."
        )
        return 0
    finally:
        if scratch:
            shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
