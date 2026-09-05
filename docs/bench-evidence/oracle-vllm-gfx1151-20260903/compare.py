#!/usr/bin/env python3
"""#2740 -- score the pinned vLLM's gfx1151 generation against the two token
streams already recorded for this arm.

Inputs, all recorded BEFORE this run and none of them re-derived here:
  A  vLLM at 5559679229 on gfx1151        -- tokens-*.json from phase 3
  B  llama.cpp b10451 HIP on gfx1151      -- GEN_IDS lines of oracle_hip.txt
  C  vllm.cpp ROCm arm on gfx1151         -- ours_gen_ids_1.json

No timing is read out of anything here. AGENTS.md Gates admits no performance
result from an arm whose declared token gate has not passed.
"""
import json
import sys
from pathlib import Path

PROMPTS = [
    "The capital city of France is",
    "The three primary colors are",
    "Water boils at a temperature of",
    "The Pythagorean theorem states that",
    "In 1969, humans first walked on",
    "A prime number is a natural number",
]


def read_oracle(path: Path) -> list[list[int]]:
    gen: dict[int, list[int]] = {}
    for line in path.read_text().splitlines():
        if line.startswith("GEN_IDS "):
            f = line.split()
            gen[int(f[1])] = [int(x) for x in f[2:]]
    return [gen[i] for i in sorted(gen)]


def read_ours(path: Path) -> list[list[int]]:
    return json.loads(path.read_text())


def read_vllm(path: Path) -> list[list[int]]:
    d = json.loads(path.read_text())
    return [r["gen_ids"] for r in d["records"]]


def first_diff(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def score(name_a: str, A: list[list[int]], name_b: str, B: list[list[int]]) -> int:
    div = 0
    print(f"\n### {name_a}  vs  {name_b}")
    print("| prompt | verdict | first diff | A | B |")
    print("|---|---|---:|---:|---:|")
    for i, p in enumerate(PROMPTS):
        k = first_diff(A[i], B[i])
        if k is None:
            print(f"| {i} `{p}` | **TOKEN-EXACT {len(A[i])}/{len(A[i])}** | — | — | — |")
        else:
            div += 1
            print(f"| {i} `{p}` | DIVERGE | {k} | {A[i][k]} | {B[i][k]} |")
    print(f"\nDIVERGENCES({name_a} vs {name_b}) = {div}/6")
    return div


if __name__ == "__main__":
    vllm_json, oracle_txt, ours_json = (Path(x) for x in sys.argv[1:4])
    A = read_vllm(vllm_json)
    B = read_oracle(oracle_txt)
    C = read_ours(ours_json)
    for nm, X in (("A vLLM@5559679229/gfx1151", A),
                  ("B llama.cpp b10451/HIP", B),
                  ("C vllm.cpp ROCm arm", C)):
        print(f"{nm}: {len(X)} prompts, lengths {[len(x) for x in X]}")
    d_ab = score("A vLLM", A, "B llama.cpp", B)
    d_cb = score("C vllm.cpp", C, "B llama.cpp", B)
    d_ca = score("C vllm.cpp", C, "A vLLM", A)
    print()
    print(f"VLLM_vs_LLAMACPP_DIVERGENCES={d_ab}/6")
    print(f"VLLMCPP_vs_LLAMACPP_DIVERGENCES={d_cb}/6")
    print(f"VLLMCPP_vs_VLLM_DIVERGENCES={d_ca}/6")
