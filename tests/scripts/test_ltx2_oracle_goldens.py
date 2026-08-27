#!/usr/bin/env python3
"""The #1864 reference-render evidence is checked, not merely stored.

`.agents/oracles/ltx-2.md` reads `gateable = yes` and names
`tests/parity/goldens/ltx2_oracle/ltx2_oracle_manifest.json` as its `evidence`.
`scripts/check-oracle-pins.py` only asserts that the path EXISTS: falsifying the
manifest's contents or truncating the mp4 beside it leaves that checker green,
which a fresh review demonstrated by doing both. `SHA256SUMS` was a
transcription and not a gate, and a digest nothing recomputes is a comment.

This suite recomputes it, and cross-checks the manifest against the record that
cites it.

**What that does and does not buy, stated plainly, because a review found the
first wording over-claimed.** It catches an artefact edited on its own — a
falsified manifest, a truncated mp4 — and it catches a manifest that stops
agreeing with the oracle record that cites it. It does NOT catch an edit that
updates `SHA256SUMS` to match: the digest half is self-referential, both sides
of it live in this tree, and no in-tree file can anchor itself. The real anchors
are the artefacts on the NAS at `/workspace/ltx2-oracle/out/` and the job log
that printed the executing script's own sha256, and neither is committed. Adding
machinery here would not change that, so the limit is written down instead.

It needs no build, no GPU, no network and no third-party module.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GOLDENS = ROOT / "tests/parity/goldens/ltx2_oracle"
SUMS = GOLDENS / "SHA256SUMS"
MANIFEST = GOLDENS / "ltx2_oracle_manifest.json"
MP4 = GOLDENS / "upstream-render.mp4"
ORACLE = ROOT / ".agents/oracles/ltx-2.md"

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"  ok   {msg}")
    else:
        print(f"  FAIL {msg}")
        failures.append(msg)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_sums(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, _, name = line.partition("  ")
        out[name.strip()] = digest.strip()
    return out


print("ltx2 oracle goldens:")

for p in (SUMS, MANIFEST, MP4):
    check(p.is_file(), f"{p.relative_to(ROOT)} exists")
if failures:
    print("FAILED (missing evidence)")
    sys.exit(1)

sums = parse_sums(SUMS.read_text())

# The two committed artefacts are the ones this suite can recompute. The 25 PPM
# frames and audio.wav are deliberately NOT committed, so their digests here stay
# a record for whoever fetches them from the NAS; a suite that silently skipped
# them would be worse than one that says which it covers.
covered = {"upstream-render.mp4": MP4, "ltx2_oracle_manifest.json": MANIFEST}
for name, path in covered.items():
    check(name in sums, f"SHA256SUMS names {name}")
    if name in sums:
        check(sha256(path) == sums[name],
              f"{name} matches its recorded sha256")

# Names, not a count. `len(...) == 26` passed a scratch mutation that renamed
# every entry to `junk_N.bin`, which is a cardinality bound wearing the message
# of an identity check.
expected_uncommitted = {"audio.wav"} | {f"frame_{i:06d}.ppm" for i in range(25)}
uncommitted = {n for n in sums if n not in covered}
check(uncommitted == expected_uncommitted,
      "SHA256SUMS records exactly audio.wav and frame_000000..000024.ppm as the "
      "uncommitted NAS artefacts"
      + ("" if uncommitted == expected_uncommitted else
         f" (missing {sorted(expected_uncommitted - uncommitted)}, "
         f"unexpected {sorted(uncommitted - expected_uncommitted)})"))

manifest = json.loads(MANIFEST.read_text())

check(manifest.get("issue") == 1864, "manifest names issue 1864")

pin_line = re.search(r"^pin\s*=\s*([0-9a-f]{40})\s*$", ORACLE.read_text(), re.M)
check(pin_line is not None, ".agents/oracles/ltx-2.md declares a 40-hex pin")
if pin_line:
    check(manifest["identity"]["revision"] == pin_line.group(1),
          "manifest revision equals the pin the oracle record declares")

ev = re.search(r"^evidence\s*=\s*(\S+)\s*$", ORACLE.read_text(), re.M)
check(ev is not None and (ROOT / ev.group(1)).resolve() == MANIFEST.resolve(),
      "the oracle record's evidence field points at this manifest")

check(manifest["result"]["frames_decoded"] > 0,
      "the manifest records a render that produced frames")
check(manifest["result"]["video_bytes"] == MP4.stat().st_size,
      "the committed mp4's size equals the manifest's video_bytes")

for name, entry in manifest["checkpoints"].items():
    check(entry["size_matches_expected"] is True,
          f"checkpoint {name} loaded at its expected size")
    check(re.fullmatch(r"[0-9a-f]{64}", entry["sha256"] or "") is not None,
          f"checkpoint {name} carries a 64-hex sha256")

if failures:
    print(f"FAILED ({len(failures)})")
    sys.exit(1)
print("PASSED")
