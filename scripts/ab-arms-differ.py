#!/usr/bin/env python3
"""Decide whether two A/B arms are actually two arms, and say what proved it.

WHY THIS IS NOT ONE `sha256sum` COMPARISON. `.agents/specs/minimax-music3.md`
§16.6a voided a Thor pair because both arms were one binary, and drew the right
rule: whenever two artifacts are required to differ, assert that they differ
before believing anything downstream. The tell was not the times, which were
0.26 % apart and looked like noise. It was the IDENTICAL CALL COUNT.

The hash cannot carry that rule here (#1516). The timed program for that pair,
`minimax-music3-gen`, is a 72 744-byte client of `libvllm_shared.so`
(`examples/CMakeLists.txt:425-426`), no `CMAKE_SKIP_BUILD_RPATH` is set, and so
CMake writes the build-tree RPATH into it. Measured on a minimal project of the
same shape: two byte-identical source trees built into two build directories
give two different client hashes with equal sizes, and making the library change
for real leaves the client byte-for-byte the hash it already had. The client's
hash is a function of the BUILD DIRECTORY, not of the change.

Hashing the library instead only moves the problem: it is stable across two
build directories and differs across two SOURCE directories, because `VT_CHECK`
embeds `__FILE__` and nothing sets `-ffile-prefix-map`, and separate clones are
the shape §16.6a's own repair mandates.

So this tool renders a verdict from three legs and prints what each is worth:

1. EQUAL hashes stay `FATAL`. That defect is real and this does not soften it.
   Different hashes are reported and decide NOTHING.
2. Location dependence: each artifact is searched for the byte string of its own
   build or source root. A hit names the root and the offset. With no `--root`
   given the line reads `UNMEASURED`, never `no`, because an absent probe must
   not read like a passing one.
3. Controls: `--control NAME A B`, repeatable. At least one must have MOVED, or
   the verdict is `FATAL`. Offering none is `FATAL: NO_CONTROL` -- a hash-only
   verdict is refused by name.

Two kinds of control, and they catch different failures. A BEHAVIOURAL control is
a value the arms computed -- a bucket's call count, an emitted stage -- and it is
the only leg that catches a stale binary, which is what §16.6a suffered. A SOURCE
control is the hash of the file the change lives in, and it catches two arms that
are the same source. Neither subsumes the other.

Where a runtime switch can turn the change off inside ONE binary, prefer that: a
same-binary A/B has no second artifact and therefore no vacuous hash.

    scripts/ab-arms-differ.py --artifact-a A --artifact-b B \\
        --root-a /tmp/b-old --root-b /tmp/b-new \\
        --control ar.depth_forward 1414 808

Exit statuses: 0 the arms are separated by something that depends on the change,
5 they are not (matching the `FATAL` exit `scripts/music3-vocoder-conv-ab.sh`
already uses), 2 usage.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

EXIT_PASS = 0
EXIT_FATAL = 5

READ_CHUNK = 1 << 20


def digest(path: Path) -> tuple[str, int]:
    """``(sha256, size)`` for one artifact, read in chunks."""

    sha = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(READ_CHUNK)
            if not chunk:
                break
            size += len(chunk)
            sha.update(chunk)
    return sha.hexdigest(), size


def embedded_root(path: Path, roots: list[str]) -> tuple[str, int] | None:
    """The first named root this artifact carries as bytes, and its offset.

    This is the RPATH in the reported case, and it is deliberately not an ELF
    query: an absolute source path baked in by `__FILE__` contaminates a hash
    exactly as much as a RUNPATH does, and a byte search finds both without
    needing `readelf` on the measuring host.
    """

    if not roots:
        return None
    blob = path.read_bytes()
    for root in roots:
        offset = blob.find(root.encode("utf-8", "surrogateescape"))
        if offset >= 0:
            return root, offset
    return None


def verdict(
    artifact_a: Path,
    artifact_b: Path,
    roots_a: list[str],
    roots_b: list[str],
    controls: list[tuple[str, str, str]],
) -> tuple[int, list[str]]:
    """``(exit status, report lines)``. The report is printed whatever the
    verdict, because a passing run has to show what carried it."""

    report: list[str] = []
    sha_a, size_a = digest(artifact_a)
    sha_b, size_b = digest(artifact_b)
    report.append(f"ARM_A sha256={sha_a} size={size_a} path={artifact_a}")
    report.append(f"ARM_B sha256={sha_b} size={size_b} path={artifact_b}")
    report.append(f"HASHES_DIFFER={'no' if sha_a == sha_b else 'yes'}")

    if not roots_a and not roots_b:
        report.append(
            "HASH_LOCATION_DEPENDENT=UNMEASURED (no --root-a/--root-b given; "
            "this is not a `no`)"
        )
    else:
        found_a = embedded_root(artifact_a, roots_a)
        found_b = embedded_root(artifact_b, roots_b)
        hits = [
            (name, found)
            for name, found in (("ARM_A", found_a), ("ARM_B", found_b))
            if found is not None
        ]
        if hits:
            report.append(
                "HASH_LOCATION_DEPENDENT=yes -- the hash above reflects WHERE "
                "this was built as well as what it contains, so a difference "
                "between the arms is not evidence about the change"
            )
            for name, (root, offset) in hits:
                report.append(f"    {name} embeds {root!r} at offset {offset}")
        else:
            report.append("HASH_LOCATION_DEPENDENT=no (no named root is embedded)")

    moved = 0
    for name, value_a, value_b in controls:
        state = "IDENTICAL" if value_a == value_b else "MOVED"
        moved += state == "MOVED"
        report.append(f"CONTROL {name}: {value_a!r} vs {value_b!r} -> {state}")

    if sha_a == sha_b:
        report.append(
            "VERDICT=FATAL ARMS_IDENTICAL: both arms are the SAME artifact, so "
            "no measurement taken from them is about the change (spec "
            "minimax-music3.md 16.6a)"
        )
        return EXIT_FATAL, report
    if not controls:
        report.append(
            "VERDICT=FATAL NO_CONTROL: no --control was offered, and two "
            "different hashes are not on their own evidence that the arms "
            "differ in the change -- a build-tree RPATH alone produces two "
            "hashes from identical source (#1516)"
        )
        return EXIT_FATAL, report
    if moved == 0:
        report.append(
            "VERDICT=FATAL NO_CONTROL_MOVED: every control reads the same on "
            "both arms. Equal times are noise; equal counts are identity"
        )
        return EXIT_FATAL, report
    report.append(
        f"VERDICT=PASS {moved} of {len(controls)} control(s) moved. That, and "
        "not the hash, is what separates these arms"
    )
    return EXIT_PASS, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact-a", type=Path, required=True)
    parser.add_argument("--artifact-b", type=Path, required=True)
    parser.add_argument(
        "--root-a", action="append", default=[],
        help="a build or source root arm A may have baked in; repeatable",
    )
    parser.add_argument("--root-b", action="append", default=[])
    parser.add_argument(
        "--control", action="append", nargs=3, default=[],
        metavar=("NAME", "ARM_A", "ARM_B"),
        help="a value that MUST differ between the arms; repeatable",
    )
    args = parser.parse_args()

    for artifact in (args.artifact_a, args.artifact_b):
        if not artifact.is_file():
            parser.error(f"not a readable file: {artifact}")

    status, report = verdict(
        args.artifact_a,
        args.artifact_b,
        args.root_a,
        args.root_b,
        [tuple(control) for control in args.control],
    )
    for line in report:
        print(line)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
