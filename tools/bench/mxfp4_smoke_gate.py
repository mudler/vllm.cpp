#!/usr/bin/env python3
"""MXFP4 e2e smoke battery — the q3mxfp4 online-serving model gate.

This is the correctness precondition the online-serving grid runs for the
``q3mxfp4`` key (``Yi30/Qwen3-8B-MXFP4``, a classic dense ``Qwen3ForCausalLM`` on
the native Marlin W4A16 MXFP4 keep-quant path). The NVFP4 gate models ("27"/"35")
use a committed npy near-tie paged-engine ctest; the MXFP4 8B has no such golden,
so its gate reproduces the #44 e2e battery instead: greedy (temperature 0, seed 0)
completion of four fixed prompts via ``vllm-cli``, compared to the oracle golden
``docs/bench-evidence/mxfp4-qwen/golden_marlin_w4a16.json`` (captured with
``VLLM_DISABLED_KERNELS=FlashInferMxFp4LinearKernel`` → Marlin W4A16).

That golden was captured from the **then-pinned** 0.25.0 / ``702f481`` oracle. It
is provenance, not the current pin: the pin advanced to ``555967922`` (vLLM
0.26.0.dev0) on 2026-07-26 and ``.agents/upstream-sync.md`` is the record (this
docstring said "the pinned 0.25.0 oracle" in the present tense until #520). The
golden stands on its capture provenance; RE-capturing it runs against the current
pin, and ``~/venvs/vllm-oracle`` is host state that still resolves to the 0.25.0
rollback (#375), so check what it points at before capturing anything.

The battery encodes exactly the #44 verdict: the three DETERMINISTIC prompts
(capitals, arithmetic, fibonacci) must be TOKEN-EXACT vs the golden text, and the
one open-ended NEAR-TIE prompt (the story) must be COHERENT (not the async-race
degeneration " Paris ... I I I ... !!!!!" that the pre-#44 classic-dense path
produced). A deterministic-prompt mismatch is a real regression and fails the
gate; a degenerate story is the #31/#44 async-mirror regression and fails the gate.

The gate runs the DEFAULT engine config (async scheduling ON + the classic-dense
device-mirror fix from #44) so a PASS proves the benched binary reproduces #44's
state before any throughput number is trusted.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import subprocess
import sys
from typing import Any, Sequence

# Golden index -> regime. Indices not listed default to DETERMINISTIC.
DEFAULT_NEAR_TIE_INDICES = (2,)  # "Once upon a time, in a small village,"

# A completion counts as degenerate when its most frequent whitespace token
# dominates (the async-race failure repeats a single token) or it is trivially
# short. Coherent greedy prose from this 8B never approaches these bounds.
MIN_WORDS = 8
MAX_TOP_WORD_RATIO = 0.5


def _run_vllm_cli(
    vllm_cli: pathlib.Path,
    snapshot: pathlib.Path,
    prompt: str,
    max_tokens: int,
) -> str:
    """Return the greedy completion text (stdout) for one prompt."""

    command = [
        str(vllm_cli),
        "--model",
        str(snapshot),
        "--prompt",
        prompt,
        "--max-tokens",
        str(max_tokens),
        "--temperature",
        "0",
        "--seed",
        "0",
    ]
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"vllm-cli exited {completed.returncode} for prompt {prompt!r}\n"
            f"stderr:\n{completed.stderr}"
        )
    # vllm-cli prints only the completion text (+ newline) to stdout; the loading
    # banner and finish_reason/token counts go to stderr.
    return completed.stdout.rstrip("\n")


def _is_coherent(text: str) -> tuple[bool, str]:
    words = text.split()
    if len(words) < MIN_WORDS:
        return False, f"only {len(words)} words (< {MIN_WORDS})"
    counts = collections.Counter(words)
    top_word, top_count = counts.most_common(1)[0]
    ratio = top_count / len(words)
    if ratio >= MAX_TOP_WORD_RATIO:
        return False, (
            f"token {top_word!r} repeats {top_count}/{len(words)} "
            f"(ratio {ratio:.2f} >= {MAX_TOP_WORD_RATIO})"
        )
    return True, f"{len(words)} words, top-word ratio {ratio:.2f}"


def run_battery(
    vllm_cli: pathlib.Path,
    snapshot: pathlib.Path,
    golden: Sequence[dict[str, Any]],
    near_tie_indices: Sequence[int],
) -> int:
    near_tie = set(near_tie_indices)
    deterministic_ok = 0
    deterministic_total = 0
    near_tie_ok = 0
    near_tie_total = 0
    failures: list[str] = []

    for index, entry in enumerate(golden):
        prompt = entry["prompt"]
        expected = entry["text"]
        max_tokens = len(entry["token_ids"])
        got = _run_vllm_cli(vllm_cli, snapshot, prompt, max_tokens)
        regime = "near-tie" if index in near_tie else "deterministic"
        print(f"--- prompt[{index}] ({regime}): {prompt!r}")
        print(f"    max_tokens={max_tokens}")
        print(f"    expected: {expected!r}")
        print(f"    got     : {got!r}")

        if index in near_tie:
            near_tie_total += 1
            coherent, why = _is_coherent(got)
            if coherent:
                near_tie_ok += 1
                verdict = "COHERENT (near-tie OK)"
            else:
                verdict = f"DEGENERATE ({why})"
                failures.append(f"prompt[{index}] near-tie degenerate: {why}")
        else:
            deterministic_total += 1
            if got.strip() == expected.strip():
                deterministic_ok += 1
                verdict = "TOKEN-EXACT"
            else:
                verdict = "MISMATCH (deterministic regression)"
                failures.append(
                    f"prompt[{index}] deterministic mismatch vs golden"
                )
        print(f"    verdict : {verdict}")

    print(
        f"\nMXFP4 smoke battery: deterministic {deterministic_ok}/{deterministic_total} "
        f"token-exact; near-tie {near_tie_ok}/{near_tie_total} coherent."
    )
    passed = (
        deterministic_ok == deterministic_total
        and near_tie_ok == near_tie_total
        and deterministic_total > 0
    )
    if passed:
        print("MXFP4 smoke battery: PASS (reproduces #44 3/4 token-exact + near-tie).")
        return 0
    print("MXFP4 smoke battery: FAIL")
    for failure in failures:
        print(f"  - {failure}")
    return 1


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-cli", type=pathlib.Path, required=True)
    parser.add_argument("--snapshot", type=pathlib.Path, required=True)
    parser.add_argument("--golden", type=pathlib.Path, required=True)
    parser.add_argument(
        "--near-tie-index",
        type=int,
        action="append",
        default=None,
        help="golden index treated as an open-ended near-tie prompt (repeatable; "
        "default: 2)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.vllm_cli.is_file():
        print(f"vllm-cli binary is absent: {args.vllm_cli}", file=sys.stderr)
        return 2
    if not args.snapshot.is_dir():
        print(f"snapshot directory is absent: {args.snapshot}", file=sys.stderr)
        return 2
    golden = json.loads(args.golden.read_text(encoding="utf-8"))
    if not isinstance(golden, list) or not golden:
        print(f"golden is not a non-empty list: {args.golden}", file=sys.stderr)
        return 2
    near_tie_indices = (
        DEFAULT_NEAR_TIE_INDICES
        if args.near_tie_index is None
        else tuple(args.near_tie_index)
    )
    try:
        return run_battery(args.vllm_cli, args.snapshot, golden, near_tie_indices)
    except RuntimeError as error:
        print(f"MXFP4 smoke battery: ERROR\n{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
