#!/usr/bin/env python3
"""DSR ratchet — Device-Specific References in the device-agnostic shared layer.

Work row `S1` of [.agents/specs/accelerator-seam-audit.md]. Runs standalone with
no CUDA toolkit, no GPU and no build:

    python3 scripts/check-device-leakage.py            # gate against the baseline
    python3 scripts/check-device-leakage.py --report   # per-file breakdown
    python3 scripts/check-device-leakage.py --write-baseline   # after a REDUCTION

WHY THIS EXISTS. `src/vllm/` + `include/vllm/` is the layer that is supposed to
be device-agnostic: the engine, the worker, the model definitions and the
platform seam. Upstream vLLM keeps essentially no device branching in
`model_executor/models/` (14 sites across 287 files) because
`model_executor/layers/` absorbs it once for everyone. We never ported that
library, so the same branching lands in our model TUs instead — 71% of it in a
single file. The audit re-measured the leakage and found it had DRIFTED UPWARD
since the previous study with **no bad commit**: DeepSeek-V2, Qwen3-Coder and the
attention-registry work each added a device test in passing. Leakage grows
silently under well-executed work, so it needs a ratchet, not a cleanup.

WHAT IT COUNTS. Four buckets, over non-comment / non-string-literal source:

  kcuda      textual `DeviceType::kCUDA` (or bare `kCUDA`) references
  is_cuda    `is_cuda()` call sites
  cuda_inc   `#include "vt/cuda/…"` / `<cuda_runtime…>` NOT inside a CUDA or
             VT_* preprocessor guard (i.e. a non-CUDA build cannot compile)
  vt_ifdef   `#ifdef VT_*` / `#if defined(VT_*)` build-time kernel-feature gates

Comments and string literals are stripped before matching in the first two
buckets. That is a DELIBERATE correction to the audit's composition, which
excluded comment lines from `is_cuda` but not from `kCUDA` — a prose mention of
a device name is not device leakage, and making the rule uniform is what keeps
the metric honest in both directions.

HOW THE RATCHET WORKS. `scripts/device-leakage-baseline.json` holds the accepted
per-bucket counts. Any bucket ABOVE its baseline fails. Any bucket BELOW its
baseline also fails, with the instruction to re-run `--write-baseline` and commit
the lowered baseline IN THE SAME COMMIT as the reduction. So the number can only
ever move down, and it moves down only deliberately. Per-bucket enforcement (not
just the total) is what stops a removed `#ifdef` from paying for a new `kCUDA`.

THE ALLOWLIST is per-file, per-bucket, with an exact expected count and a stated
reason. These sites ARE the CUDA leg — the `(kCUDA, name)` registrar keys, the
platform-priority walk, the `is_cuda()` definition — and counting them would
punish adding a backend, which is the opposite of the point. An allowlist entry
whose count no longer matches fails too, so the allowlist itself is a ratchet and
cannot silently absorb new leakage.

THE ESCAPE HATCH is `// DSR-ALLOW(<row-id>): <reason>` on the offending line or
the line directly above it. Such sites are excluded from the count but are
COUNTED AND PRINTED separately on every run, so the exception budget is visible
in CI output rather than invisible in the diff. Repair the code, do not grow this
list; and never raise a baseline to make a failing state pass.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "scripts/device-leakage-baseline.json"

# The layer that is supposed to be device-agnostic. `src/vt/cuda/`,
# `src/vt/metal/` and `src/vt/vulkan/` are device legs BY DEFINITION and are not
# scanned at all — they are where device-specific code is SUPPOSED to live.
SCAN_ROOTS = ("src/vllm", "include/vllm")
SOURCE_SUFFIXES = {".h", ".hpp", ".cpp", ".cc", ".cu", ".cuh"}

BUCKETS = ("kcuda", "is_cuda", "cuda_inc", "vt_ifdef")

RE_KCUDA = re.compile(r"\bkCUDA\b")
RE_IS_CUDA = re.compile(r"\bis_cuda\s*\(\s*\)")
RE_CUDA_INCLUDE = re.compile(r'^\s*#\s*include\s*[<"](?:vt/cuda/|cuda_runtime)')
RE_PP_IF = re.compile(r"^\s*#\s*(ifdef|ifndef|if)\b(.*)$")
RE_PP_ELIF = re.compile(r"^\s*#\s*elif\b(.*)$")
RE_PP_ELSE = re.compile(r"^\s*#\s*else\b")
RE_PP_ENDIF = re.compile(r"^\s*#\s*endif\b")
RE_VT_IFDEF = re.compile(r"^\s*#\s*(?:ifdef\s+VT_\w+|if\s+defined\s*\(\s*VT_\w+)")
# A preprocessor condition that makes an enclosed CUDA include legal in a
# non-CUDA build. `VT_*` gates are CUDA kernel-feature macros, only ever defined
# by the CUDA build (cmake/CudaArchFeatures.cmake).
RE_CUDA_GUARD = re.compile(r"\b(VLLM_CPP_CUDA|VT_\w+|__CUDACC__|CUDA_VERSION)\b")

RE_DSR_ALLOW = re.compile(r"//\s*DSR-ALLOW\(\s*([A-Za-z0-9_.\-]+)\s*\)\s*:\s*(\S.*?)\s*$")


# --- allowlist ---------------------------------------------------------------
#
# path -> {bucket: (expected_count | "*", reason)}
#
# "*" means the whole file IS a device leg and the bucket is not counted there.
# An integer means EXACTLY that many references are the platform definition; a
# 17th `kCUDA` in a registrar file is leakage and fails. Every entry states why
# the site is the CUDA leg rather than the shared layer branching on it.
ALLOWLIST: dict[str, dict[str, tuple[object, str]]] = {
    "src/vllm/platforms/cuda.cpp": {
        "kcuda": ("*", "IS the CUDA platform leg — `device_type()`, `backend()` "
                       "and the static `RegisterPlatform(kCUDA, …)`. Counting a "
                       "platform file's own device name would punish adding a "
                       "platform, which is the opposite of the metric's point."),
        "cuda_inc": ("*", "the CUDA platform leg is the one TU that may include "
                          "<cuda_runtime.h> unconditionally; it is compiled only "
                          "when VLLM_CPP_CUDA is on (see its CMake guard)."),
    },
    "src/vllm/platforms/platform.cpp": {
        "kcuda": (1, "the device PRIORITY WALK "
                     "`{kCUDA, kXPU, kVULKAN, kMETAL, kCPU}` — a data list that "
                     "names every platform equally, mirroring upstream's "
                     "`platforms/__init__.py` import probe."),
    },
    "include/vllm/platforms/interface.h": {
        "kcuda": (1, "the `is_cuda()` DEFINITION itself "
                     "(`device_type() == DeviceType::kCUDA`), mirroring "
                     "`vllm/platforms/interface.py:189-215`. The definition is "
                     "the seam; its 11 CALL SITES in the shared layer are not."),
        "is_cuda": (1, "same line — the definition, not a call site."),
    },
    "src/vllm/v1/attention/backend.cpp": {
        "kcuda": (2, "`(kCUDA, name)` REGISTRAR KEYS for TRITON_MLA and "
                     "FLASH_ATTN. Registration keys are how a backend declares "
                     "which device it serves — upstream's "
                     "`AttentionBackendEnum` (registry.py:34-120) spends a "
                     "closed enum entry on exactly the same thing."),
    },
    "src/vllm/v1/attention/backends/gdn_attn.cpp": {
        "kcuda": (1, "the `(kCUDA, \"GDN_ATTN\")` registrar key — same reason as "
                     "backend.cpp above."),
    },
}


@dataclass
class Hit:
    path: str
    line: int
    bucket: str
    text: str


@dataclass
class Result:
    counts: dict[str, int] = field(default_factory=lambda: dict.fromkeys(BUCKETS, 0))
    per_file: dict[str, dict[str, int]] = field(default_factory=dict)
    hits: list[Hit] = field(default_factory=list)
    allowed: dict[str, dict[str, int]] = field(default_factory=dict)
    exempt: list[Hit] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def strip_comments_and_strings(text: str) -> list[str]:
    """Blank out //, /* */ and "…" / '…' spans, preserving line structure.

    A device name inside a comment or a diagnostic message is prose, not a
    branch. Newlines are preserved so line numbers survive.
    """
    out: list[str] = []
    i, n = 0, len(text)
    state = "code"  # code | line_comment | block_comment | string | char
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if state == "code":
            if c == "/" and nxt == "/":
                state = "line_comment"
                out.append("  ")
                i += 2
                continue
            if c == "/" and nxt == "*":
                state = "block_comment"
                out.append("  ")
                i += 2
                continue
            if c == '"':
                state = "string"
                out.append(" ")
                i += 1
                continue
            if c == "'":
                state = "char"
                out.append(" ")
                i += 1
                continue
            out.append(c)
            i += 1
            continue
        if state == "line_comment":
            if c == "\n":
                state = "code"
                out.append("\n")
            else:
                out.append(" ")
            i += 1
            continue
        if state == "block_comment":
            if c == "*" and nxt == "/":
                state = "code"
                out.append("  ")
                i += 2
                continue
            out.append("\n" if c == "\n" else " ")
            i += 1
            continue
        # string / char
        if c == "\\" and nxt:
            out.append("  ")
            i += 2
            continue
        if (state == "string" and c == '"') or (state == "char" and c == "'"):
            state = "code"
            out.append(" ")
            i += 1
            continue
        out.append("\n" if c == "\n" else " ")
        i += 1
    return "".join(out).splitlines()


def cuda_guard_depth(raw_lines: list[str]) -> list[bool]:
    """Per line: is it inside a CUDA / VT_* preprocessor conditional?

    An unconditional `#include "vt/cuda/…"` breaks the non-CUDA build outright;
    the same include under `#ifdef VT_MARLIN_NVFP4` does not. Only the former is
    leakage.
    """
    guarded: list[bool] = []
    stack: list[bool] = []
    for line in raw_lines:
        m = RE_PP_IF.match(line)
        if m:
            stack.append(bool(RE_CUDA_GUARD.search(m.group(2))))
            guarded.append(any(stack))
            continue
        m = RE_PP_ELIF.match(line)
        if m and stack:
            stack[-1] = bool(RE_CUDA_GUARD.search(m.group(1)))
            guarded.append(any(stack))
            continue
        if RE_PP_ELSE.match(line) and stack:
            # The negative arm of a CUDA guard is the PORTABLE arm; a CUDA
            # include there would be a genuine break, so stop treating it as
            # guarded.
            stack[-1] = False
            guarded.append(any(stack))
            continue
        if RE_PP_ENDIF.match(line):
            if stack:
                stack.pop()
            guarded.append(any(stack))
            continue
        guarded.append(any(stack))
    return guarded


def source_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for rel in SCAN_ROOTS:
        base = root / rel
        if not base.is_dir():
            continue
        files.extend(
            p for p in sorted(base.rglob("*")) if p.suffix in SOURCE_SUFFIXES and p.is_file()
        )
    return files


def dsr_allow_lines(raw_lines: list[str]) -> dict[int, tuple[str, str]]:
    """1-based line -> (row_id, reason) for the line the escape hatch covers.

    The marker covers its OWN line and the line directly below it, so it can sit
    above a statement that has no room for a trailing comment.
    """
    covered: dict[int, tuple[str, str]] = {}
    for idx, line in enumerate(raw_lines, start=1):
        m = RE_DSR_ALLOW.search(line)
        if not m:
            continue
        covered[idx] = (m.group(1), m.group(2))
        covered.setdefault(idx + 1, (m.group(1), m.group(2)))
    return covered


def scan(root: Path) -> Result:
    res = Result()
    for path in source_files(root):
        rel = path.relative_to(root).as_posix()
        raw = path.read_text(encoding="utf-8", errors="replace")
        raw_lines = raw.splitlines()
        code_lines = strip_comments_and_strings(raw)
        guarded = cuda_guard_depth(raw_lines)
        allow_map = dsr_allow_lines(raw_lines)
        entry = ALLOWLIST.get(rel, {})

        found: dict[str, list[Hit]] = {b: [] for b in BUCKETS}
        for lineno, code in enumerate(code_lines, start=1):
            for m in RE_KCUDA.finditer(code):
                del m
                found["kcuda"].append(Hit(rel, lineno, "kcuda", raw_lines[lineno - 1].strip()))
            for m in RE_IS_CUDA.finditer(code):
                del m
                found["is_cuda"].append(Hit(rel, lineno, "is_cuda", raw_lines[lineno - 1].strip()))
        for lineno, line in enumerate(raw_lines, start=1):
            if RE_CUDA_INCLUDE.match(line) and not guarded[lineno - 1]:
                found["cuda_inc"].append(Hit(rel, lineno, "cuda_inc", line.strip()))
            if RE_VT_IFDEF.match(line):
                found["vt_ifdef"].append(Hit(rel, lineno, "vt_ifdef", line.strip()))

        for bucket, hits in found.items():
            allow = entry.get(bucket)
            if allow is not None:
                expected, _reason = allow
                if expected == "*":
                    res.allowed.setdefault(rel, {})[bucket] = len(hits)
                    continue
                if len(hits) != expected:
                    res.errors.append(
                        f"ALLOWLIST STALE: {rel} bucket '{bucket}' has {len(hits)} "
                        f"reference(s) but the allowlist expects exactly {expected}. "
                        "The allowlist is itself a ratchet: if the platform leg "
                        "genuinely changed shape, edit ALLOWLIST in "
                        "scripts/check-device-leakage.py and say why in the same "
                        "commit — never widen it to absorb a new device branch."
                    )
                res.allowed.setdefault(rel, {})[bucket] = min(len(hits), int(expected))
                hits = hits[int(expected):]

            kept: list[Hit] = []
            for hit in hits:
                cover = allow_map.get(hit.line)
                if cover:
                    res.exempt.append(hit)
                else:
                    kept.append(hit)
            if kept:
                res.per_file.setdefault(rel, dict.fromkeys(BUCKETS, 0))
                res.per_file[rel][bucket] += len(kept)
                res.counts[bucket] += len(kept)
                res.hits.extend(kept)
    return res


def load_baseline() -> dict[str, int]:
    if not BASELINE_PATH.exists():
        return {}
    data = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    return {b: int(data["buckets"][b]) for b in BUCKETS}


def write_baseline(res: Result) -> None:
    payload = {
        "_comment": [
            "DSR baseline for scripts/check-device-leakage.py (work row S1 of",
            ".agents/specs/accelerator-seam-audit.md). Device-Specific References",
            "in the device-agnostic shared layer (src/vllm/ + include/vllm/),",
            "net of the checker's per-file platform-leg allowlist.",
            "THIS NUMBER MAY ONLY EVER GO DOWN. Lower it in the SAME commit as the",
            "reduction that earned it, by running:",
            "  python3 scripts/check-device-leakage.py --write-baseline",
            "It is a leakage budget, never to be raised to make a failing check pass.",
        ],
        "total": res.total,
        "buckets": {b: res.counts[b] for b in BUCKETS},
    }
    BASELINE_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def report(res: Result, out=sys.stdout) -> None:
    print("Device-Specific References (DSR) in the shared layer "
          f"({' + '.join(SCAN_ROOTS)})", file=out)
    print("", file=out)
    width = max(len(f) for f in res.per_file) if res.per_file else 40
    print(f"{'file':<{width}}  " + "  ".join(f"{b:>9}" for b in BUCKETS) + "      DSR", file=out)
    for rel in sorted(res.per_file, key=lambda r: (-sum(res.per_file[r].values()), r)):
        row = res.per_file[rel]
        print(
            f"{rel:<{width}}  "
            + "  ".join(f"{row[b]:>9}" for b in BUCKETS)
            + f"  {sum(row.values()):>7}",
            file=out,
        )
    print("", file=out)
    print(f"{'TOTAL':<{width}}  "
          + "  ".join(f"{res.counts[b]:>9}" for b in BUCKETS)
          + f"  {res.total:>7}", file=out)
    print("", file=out)
    if res.allowed:
        print("Allowlisted platform-leg sites (NOT counted — these ARE the CUDA leg):",
              file=out)
        for rel in sorted(res.allowed):
            for bucket, n in sorted(res.allowed[rel].items()):
                reason = ALLOWLIST[rel][bucket][1]
                print(f"  {rel} [{bucket}] x{n}: {reason}", file=out)
        print("", file=out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--report", action="store_true", help="print the per-file breakdown")
    ap.add_argument("--list", action="store_true", help="print every counted site")
    ap.add_argument(
        "--write-baseline",
        action="store_true",
        help="rewrite the baseline to the measured counts (only ever DOWNWARD)",
    )
    ap.add_argument("--root", default=str(ROOT), help="tree to scan (testing)")
    args = ap.parse_args(argv)

    res = scan(Path(args.root))

    if args.report:
        report(res)
    if args.list:
        for hit in res.hits:
            print(f"{hit.path}:{hit.line}: [{hit.bucket}] {hit.text}")
        print("")

    # The escape hatch is always LOUD: printed on every run, pass or fail, so a
    # growing exception budget is visible in CI output.
    print(f"DSR-ALLOW exemptions in force: {len(res.exempt)}")
    for hit in res.exempt:
        print(f"  DSR-ALLOW {hit.path}:{hit.line} [{hit.bucket}] {hit.text}")

    print("DSR by bucket: "
          + ", ".join(f"{b}={res.counts[b]}" for b in BUCKETS)
          + f"  -> total {res.total}")

    if args.write_baseline:
        prev = load_baseline()
        if prev and res.total > sum(prev.values()):
            print("REFUSING to write a HIGHER baseline "
                  f"({sum(prev.values())} -> {res.total}). The DSR ratchet only "
                  "turns one way: reduce the leakage instead.", file=sys.stderr)
            return 1
        write_baseline(res)
        try:
            shown = BASELINE_PATH.relative_to(ROOT)
        except ValueError:  # a scratch baseline under --root (mutation suite)
            shown = BASELINE_PATH
        print(f"baseline written: {shown} -> {res.total}")
        return 0

    errors = list(res.errors)
    baseline = load_baseline()
    if not baseline:
        errors.append(
            f"no baseline at {BASELINE_PATH}; run --write-baseline to establish one"
        )
    else:
        for bucket in BUCKETS:
            got, want = res.counts[bucket], baseline[bucket]
            if got > want:
                errors.append(
                    f"DSR REGRESSION in bucket '{bucket}': {got} > baseline {want}. "
                    "A device-specific reference was added to the device-agnostic "
                    "shared layer. Ask the op/provider table the question instead "
                    "(vt::OpRegistered / Platform capability), or — if the site is "
                    "genuinely the platform leg — add it to ALLOWLIST with a "
                    "reason. NEVER raise the baseline to make this pass."
                )
            elif got < want:
                errors.append(
                    f"DSR baseline STALE in bucket '{bucket}': {got} < baseline "
                    f"{want}. A reduction must lower the baseline in the SAME "
                    "commit: run `python3 scripts/check-device-leakage.py "
                    "--write-baseline` and commit the result."
                )

    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        print(f"\ncheck-device-leakage: FAIL ({len(errors)} error(s))", file=sys.stderr)
        return 1

    print(f"check-device-leakage: OK (DSR {res.total} == baseline "
          f"{sum(baseline.values())}, ratchet holds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
