#!/usr/bin/env python3
"""Refuse a harness that launches the pinned vLLM oracle SERVER on a handicapped path.

This is the structural half of
[#414](https://github.com/mudler/vllm.cpp/issues/414), and the gate that #607
wave L4 owes (`.agents/specs/multimodal-track.md` §1.6).

The defect it removes
---------------------
At the pin, `vllm/config/multimodal.py:78` defaults `language_model_only` to
`False`. A Qwen3.6 checkpoint loads as `Qwen3_5*ForConditionalGeneration`, so its
`multimodal_config` is non-`None`, so
`vllm/model_executor/models/qwen3_next.py:324-331` computes `text_only == False`
and DISABLES `use_fused_qk_norm_rope_gate`. The oracle then issues four ops per
full-attention layer where its own production configuration issues one, while
our arm has issued the single fused launch by default throughout
(`src/vllm/model_executor/models/qwen3_5.cpp::FuseAttnPreambleOn`).

A ratio measured that way compares two different algorithms, which AGENTS.md
§Gates forbids, and the error runs in OUR favour, hardest on TTFT. The
2026-08-13 clock-controlled series repaired it by passing
`--language-model-only`. That repaired the RUN, not the HARNESS:
`scripts/dgx-online-serving.sh` still launched the oracle without the flag while
`tools/bench/run_serve_low.py` passed it, so the two disagreed about the
oracle's configuration and the next canonical campaign would have reproduced the
defect. A grep for the flag reads as coverage while the canonical driver lacks
it, which is why this is a checker and not a comment.

Scope, and what is deliberately outside it
------------------------------------------
This gate covers the ONE surface where the published parity ratios come from: a
command that launches the oracle SERVER through its CLI. A token `serve`
immediately following an oracle client token (`vllm`, `.../vllm`, or a
`${client}`-style variable) is a launch. `vllm bench serve` is the timed CLIENT
rather than a server, and is not matched, because `serve` there follows `bench`.

The same defect exists on the IN-PROCESS surface: `LLM(...)` in
`tools/bench/{profile_vllm_online_gate,vllm_closed_loop_metrics,dump_vllm_tokens}.py`
constructs the oracle with `language_model_only` left at its `False` default and
exposes no way to set it. Those harnesses take `--model` as a path, so no static
rule can know whether a given run points at a multimodal checkpoint; the repair
is to thread the knob through and record the resolved value beside the
measurement. That is tracked separately and is NOT enforced here, because a
checker that cannot decide its own question is worse than none.

Each matched launch must pass `--language-model-only` or carry an explicit
exemption naming a reason:

    # ORACLE-DENOMINATOR-EXEMPT: <reason>

An exemption is for an architecture with no `multimodal_config` at all, where
`text_only` is already `True` and the flag decides nothing. It is not a place to
park a multimodal checkpoint.

Why this is not a lock
----------------------
The checker holds its expectation in its own code and reads harness sources. No
pull request has to edit a shared record to pass it, and adding a harness costs
nothing unless that harness launches an oracle server.

    scripts/check-oracle-denominator-flags.py            # gate
    scripts/check-oracle-denominator-flags.py --json     # the discovered set
    scripts/check-oracle-denominator-flags.py --root DIR # scan another tree

`--json` exists so a test can pin the DISCOVERED set, not only the violation
count. A detector whose terms stopped matching would otherwise report a clean
tree, which is the failure this project has already paid for elsewhere: a null
grep proves the terms wrong, never an absence.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# Harness surfaces. A file outside these globs cannot launch a gate oracle
# without first becoming a harness, at which point it lands here in the same
# change.
SCAN_GLOBS = ("scripts/*.sh", "scripts/*.py", "tools/bench/*.py", "tools/bench/*.sh")

# This checker reads itself out of the scan: its own prose names every token it
# looks for, and matching that prose would be a tautology.
SELF = "check-oracle-denominator-flags.py"

REQUIRED_FLAG = "--language-model-only"
EXEMPT_MARKER = "ORACLE-DENOMINATOR-EXEMPT:"

# A token naming the pinned oracle CLI: either the binary spelled out, or ANY
# shell or Python variable holding it. Naming the two variables this tree
# happens to use today would have made the detector a list of spellings, and a
# harness calling its binary `${ORACLE}` or `${server_bin}` would have launched
# the oracle unscanned while the checker reported a clean tree. What identifies
# a launch is `serve` sitting directly after the binary, so the binary may be
# anonymous. `bench serve` is unaffected and stays a client: the token before
# `serve` there is `bench`, which is no expansion at all.
#
# Case-sensitive on purpose: prose writes "vLLM", commands write "vllm". The
# optional closing brace is not cosmetic -- `_TOKEN_STRIP` eats `}` off the end
# of a token, so `"${client}"` reaches this pattern as `${client`.
_ORACLE_CLIENT = re.compile(r"\A(?:(?:.*/)?vllm|\$\{?[A-Za-z0-9_]+\}?)\Z")
_TOKEN_STRIP = "\"'`,()[]{}\\"
# A line that opens a command block, so a flag placed before `serve` still counts.
_OPENER = re.compile(r"[(\[]\s*$")
# A line that closes one.
_CLOSER = re.compile(r"\A\s*[)\]]\s*,?\s*\Z")


def _tokens(line: str) -> list[str]:
    return [t.strip(_TOKEN_STRIP) for t in line.split()]


def _is_comment(line: str) -> bool:
    stripped = line.lstrip()
    return stripped.startswith("#") or stripped.startswith("*")


def _block(lines: list[str], index: int) -> tuple[int, int]:
    """The command block containing line ``index`` (0-based), as [start, end)."""
    start = index
    while start > 0:
        if _OPENER.search(lines[start - 1]):
            start -= 1
            break
        if not lines[start - 1].strip():
            break
        start -= 1
    end = index + 1
    while end < len(lines):
        if _CLOSER.match(lines[end]):
            end += 1
            break
        if not lines[end].strip():
            break
        end += 1
    return start, end


def _exempt(lines: list[str], start: int, end: int) -> str | None:
    """The exemption reason for a block, searching the block and its 8-line lead."""
    lead = max(0, start - 8)
    for line in lines[lead:end]:
        if EXEMPT_MARKER in line:
            reason = line.split(EXEMPT_MARKER, 1)[1].strip().strip(_TOKEN_STRIP).strip()
            return reason or None
    return None


def _serve_launches(lines: list[str]) -> list[int]:
    """0-based indices of lines launching the oracle SERVER through its CLI."""
    found = []
    for i, line in enumerate(lines):
        if _is_comment(line):
            continue
        tokens = _tokens(line)
        for j, token in enumerate(tokens):
            if token != "serve" or j == 0:
                continue
            if _ORACLE_CLIENT.match(tokens[j - 1]):
                found.append(i)
                break
    return found


def scan(root: Path) -> list[dict]:
    """Every oracle server launch in the tree, with its verdict."""
    results: list[dict] = []
    paths: list[Path] = []
    for glob in SCAN_GLOBS:
        paths.extend(sorted(root.glob(glob)))
    for path in paths:
        if path.name == SELF:
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for index in _serve_launches(lines):
            start, end = _block(lines, index)
            body = "\n".join(lines[start:end])
            results.append(
                {
                    "file": str(path.relative_to(root)),
                    "line": index + 1,
                    "has_flag": REQUIRED_FLAG in body,
                    "exempt_reason": _exempt(lines, start, end),
                }
            )
    results.sort(key=lambda r: (r["file"], r["line"]))
    return results


def violations(results: list[dict]) -> list[dict]:
    return [r for r in results if not r["has_flag"] and not r["exempt_reason"]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--json", action="store_true", help="print the discovered set")
    args = parser.parse_args(argv)

    results = scan(args.root)
    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))

    if not results:
        print(
            "check-oracle-denominator-flags: found NO oracle server launch at all. "
            "That is a broken detector, not a clean tree -- this repository has at "
            "least one harness that launches the pinned oracle.",
            file=sys.stderr,
        )
        return 2

    bad = violations(results)
    if bad:
        for r in bad:
            print(
                f"{r['file']}:{r['line']}: launches the pinned vLLM oracle server "
                f"without {REQUIRED_FLAG}. The oracle then runs its UNFUSED "
                f"QK-norm+RoPE+gate path (qwen3_next.py:324-331 at the pin) while "
                f"our arm runs the fused one, so every ratio from this harness is "
                f"FLATTERED (#414). Pass the flag, or mark the block "
                f"'{EXEMPT_MARKER} <reason>' when the architecture has no "
                f"multimodal_config and the flag decides nothing.",
                file=sys.stderr,
            )
        print(
            f"\n{len(bad)} handicapped oracle launch(es) of {len(results)} "
            f"discovered. Never weaken this checker to make a harness pass: the "
            f"denominator it protects is the whole comparison.",
            file=sys.stderr,
        )
        return 1

    exempt = [r for r in results if r["exempt_reason"]]
    print(
        f"check-oracle-denominator-flags: {len(results)} oracle server launch(es), "
        f"{len(results) - len(exempt)} carrying {REQUIRED_FLAG}, "
        f"{len(exempt)} exempt."
    )
    for r in exempt:
        print(f"  exempt {r['file']}:{r['line']} -- {r['exempt_reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
