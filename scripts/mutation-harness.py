#!/usr/bin/env python3
"""Run a mutation pass and print the FOUR facts that stop a false green.

Row LTX25-RES2S-LOOP, issue #921. Written for that row's section 8 and kept
general, because every one of the four failure modes below was paid for here in
a different file.

A mutation pass answers one question: does the gate detect this defect? The
answer is the test binary's exit code, and there are four distinct ways to get
exit 0 from a mutation that proved nothing:

1. THE MUTATION NEVER APPLIED. The anchor text moved, or was in the header
   rather than the .cpp. `git diff --stat` is empty, the build is clean, the
   exit code is 0, and it reads exactly like a passing test. This harness
   REFUSES the run when the anchor is absent, and prints the diffstat when it is
   not.
2. THE MUTATION DID NOT BUILD. A failed compile leaves the previous binary on
   disk and running it reports a pass over unmutated code. This harness prints
   whether the build succeeded and how many `: error:` lines it emitted, and
   marks a non-building mutation `BUILD_FAILED` rather than scoring it.
3. THE FILTER MATCHED NOTHING. doctest's `--test-case` splits its argument on
   COMMAS, so a case name containing one is truncated and matches zero cases —
   and doctest then prints `SUCCESS!` and exits 0. This harness runs the WHOLE
   BINARY by default and asserts a NON-ZERO case count and a non-zero assertion
   count before it will call anything a survivor.
4. THE BINARY WAS NOT REBUILT. `git checkout --` restores a file with an old
   mtime, so ninja can decide the object is current and carry the previous
   mutation's binary forward. Every restore here re-stamps the file with
   `os.utime(None)`.

Usage:

    python3 scripts/mutation-harness.py --build build \\
        --test test_ltx2_pipeline \\
        --mutation "res2s-second-eval:src/vllm/model_executor/models/ltx2_samplers.cpp:\\
hooks.denoise(mid_v, mid_a, sub_sigma:hooks.denoise(video.latent, audio.latent, sigma"

Each `--mutation` is `NAME:PATH:FIND:REPLACE` (the first two colons split;
FIND and REPLACE are separated by the last colon-free split, so pass them with
`--find`/`--replace` when either contains a colon). A mutation file may also be
supplied with `--plan FILE`, one JSON object per line:

    {"name": "...", "path": "...", "find": "...", "replace": "..."}

The tree is restored byte-for-byte after every mutation, verified by sha256, and
the harness refuses to start on a dirty working tree so a restore failure cannot
be mistaken for the developer's own edit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys

CASE_RE = re.compile(r"test cases:\s*(\d+)\s*\|\s*(\d+) passed\s*\|\s*(\d+) failed")
ASSERT_RE = re.compile(r"assertions:\s*(\d+)\s*\|\s*(\d+) passed\s*\|\s*(\d+) failed")


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_clean(root: pathlib.Path) -> None:
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if dirty:
        raise SystemExit(
            "the working tree is dirty. A mutation harness that starts from "
            "uncommitted edits cannot tell its own restore failure from your work:\n"
            + dirty
        )


def diffstat(root: pathlib.Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "diff", "--stat"],
        check=True, capture_output=True, text=True,
    ).stdout.strip().replace("\n", " ; ")


def build(build_dir: pathlib.Path, target: str) -> tuple[bool, int, str]:
    """Returns (built, compile_error_count, tail)."""
    proc = subprocess.run(
        ["cmake", "--build", str(build_dir), "--target", target, "-j", "6"],
        capture_output=True, text=True,
    )
    text = proc.stdout + proc.stderr
    return proc.returncode == 0, text.count(": error:"), text[-1200:]


def run_binary(build_dir: pathlib.Path, target: str) -> dict:
    """Run the WHOLE binary. Exit code captured directly, never after a pipe."""
    binary = build_dir / "tests" / target
    if not binary.is_file():
        return {"exit": None, "cases": 0, "failed_cases": 0, "asserts": 0,
                "failed_asserts": 0, "note": f"{binary} does not exist"}
    proc = subprocess.run([str(binary)], capture_output=True, text=True)
    code = proc.returncode
    text = proc.stdout + proc.stderr
    cases = CASE_RE.search(text)
    asserts = ASSERT_RE.search(text)
    return {
        "exit": code,
        "cases": int(cases.group(1)) if cases else 0,
        "failed_cases": int(cases.group(3)) if cases else 0,
        "asserts": int(asserts.group(1)) if asserts else 0,
        "failed_asserts": int(asserts.group(3)) if asserts else 0,
        # A thrown doctest case prints `0 failed` beside `Status: FAILURE!`, so
        # the summary line is not the authority and the exit code is.
        "status_failure": "Status: FAILURE!" in text,
        "note": "",
    }


def restore(root: pathlib.Path, path: str, want_sha: str) -> None:
    subprocess.run(["git", "-C", str(root), "checkout", "--", path], check=True)
    full = root / path
    # RE-STAMP. A restored file older than its object makes ninja skip the
    # rebuild and carry the previous mutation's binary into the next run.
    os.utime(full, None)
    got = sha256(full)
    if got != want_sha:
        raise SystemExit(f"restore of {path} did not reproduce the original: {got} != {want_sha}")


def apply_mutation(root: pathlib.Path, path: str, find: str, replace: str) -> bool:
    full = root / path
    if not full.is_file():
        print(f"  ANCHOR NOT FOUND: {path} does not exist")
        return False
    text = full.read_text()
    hits = text.count(find)
    if hits != 1:
        print(f"  ANCHOR NOT FOUND: {hits} occurrences of the find text in {path} "
              f"(exactly one is required, so a moved or duplicated anchor is a "
              f"refusal rather than a silent no-op)")
        return False
    full.write_text(text.replace(find, replace, 1))
    os.utime(full, None)
    return True


def parse_mutations(args) -> list[dict]:
    out: list[dict] = []
    if args.plan:
        for line in pathlib.Path(args.plan).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(json.loads(line))
    if args.name:
        out.append({"name": args.name, "path": args.path,
                    "find": args.find, "replace": args.replace})
    if not out:
        raise SystemExit("no mutations: pass --plan or --name/--path/--find/--replace")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=pathlib.Path, default=pathlib.Path.cwd())
    ap.add_argument("--build", type=pathlib.Path, required=True)
    ap.add_argument("--test", required=True, help="the ctest/doctest binary target name")
    ap.add_argument("--plan", help="a file of one JSON mutation per line")
    ap.add_argument("--name")
    ap.add_argument("--path")
    ap.add_argument("--find")
    ap.add_argument("--replace", default="")
    args = ap.parse_args()

    root = args.root.resolve()
    require_clean(root)
    mutations = parse_mutations(args)

    # THE BASELINE IS PART OF THE EVIDENCE. A suite that is already red, or that
    # runs zero cases, makes every mutation below unreadable.
    ok, errors, tail = build(args.build, args.test)
    if not ok:
        raise SystemExit(f"the UNMUTATED tree does not build ({errors} errors):\n{tail}")
    base = run_binary(args.build, args.test)
    print(f"BASELINE {args.test}: exit={base['exit']} cases={base['cases']} "
          f"({base['failed_cases']} failed) assertions={base['asserts']} "
          f"({base['failed_asserts']} failed)")
    if base["exit"] != 0 or base["cases"] == 0 or base["asserts"] == 0:
        raise SystemExit(
            "the baseline is not a clean, non-empty green. Zero cases or zero "
            "assertions is a SKIP wearing a pass, and every mutation below would "
            "read as a survivor."
        )

    rows = []
    for mutation in mutations:
        name, path = mutation["name"], mutation["path"]
        print(f"\n--- {name} ({path})")
        want_sha = sha256(root / path)
        if not apply_mutation(root, path, mutation["find"], mutation["replace"]):
            rows.append((name, "-", "-", "-", "-", "-", "ANCHOR NOT FOUND"))
            continue
        stat = diffstat(root)
        try:
            built, errors, tail = build(args.build, args.test)
            if not built:
                print(f"  diff: {stat}\n  BUILT: NO  compile_err: {errors}\n{tail}")
                rows.append((name, stat, "NO", str(errors), "-", "-", "BUILD_FAILED"))
                continue
            result = run_binary(args.build, args.test)
            code = result["exit"]
            print(f"  diff: {stat}")
            print(f"  BUILT: YES  compile_err: {errors}")
            print(f"  EXIT: {code}  cases: {result['cases']}/{result['failed_cases']}F  "
                  f"assertions: {result['asserts']}/{result['failed_asserts']}F  "
                  f"status_failure: {result['status_failure']}")
            if result["cases"] == 0 or result["asserts"] == 0:
                verdict = "NO CASES RAN"
            elif code != 0:
                verdict = "DETECTED"
            else:
                verdict = "SURVIVED"
            print(f"  VERDICT: {verdict}")
            rows.append((name, stat, "YES", str(errors), str(code),
                         f"{result['cases']}/{result['failed_cases']}F "
                         f"{result['asserts']}/{result['failed_asserts']}F", verdict))
        finally:
            restore(root, path, want_sha)

    print("\n| # | Mutation | diff --stat | BUILT | cc-err | EXIT | cases/asserts | Verdict |")
    print("|---|---|---|---|---|---|---|---|")
    for i, row in enumerate(rows, 1):
        print(f"| M{i} | " + " | ".join(row) + " |")

    # Rebuild once at the end so the tree the developer is left with matches the
    # sources, rather than the last mutation's objects.
    build(args.build, args.test)
    sys.exit(0 if all(r[-1] == "DETECTED" for r in rows) else 1)


if __name__ == "__main__":
    main()
