#!/usr/bin/env python3
"""#2884 -- the LIMB-3 STRICT verdict on the vehicle.

Limb 3 of the ratified near-tie methodology asks for a **clean STRICT
token-exact pass** on a second dense model through the SAME forward code the
Qwen3.8-27B Q4_K_M ROCm arm runs, scored against the PRIMARY oracle. STRICT
means free-running, every token, every prompt: 6 of 6 or it is not a pass. This
script computes exactly that and nothing else.

It prints no timing and computes no ratio. AGENTS.md Gates admits no
performance result from an arm whose declared token gate has not passed.

It narrates which file supplied which side, per .agents/verification.md.
"""
import hashlib
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
# PRE-REGISTERED. Committed in three evidence documents that predate this run;
# the file form is one prompt per line, newline-terminated.
PROMPTS_SHA256 = "c8a5080046c3206a1186b42a320a21e887691eff43d2116364192f58e5776c7e"


def read_ours(path):
    return json.loads(Path(path).read_text())


def read_vllm(path):
    d = json.loads(Path(path).read_text())
    return [r["gen_ids"] for r in d["records"]], d


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def score(name_a, A, name_b, B, ntok):
    div = 0
    print(f"\n### {name_a}  vs  {name_b}")
    print("| prompt | verdict | first diff | A | B |")
    print("|---|---|---:|---:|---:|")
    for i, p in enumerate(PROMPTS):
        if i >= len(A) or i >= len(B):
            div += 1
            print(f"| {i} `{p}` | MISSING | — | — | — |")
            continue
        k = first_diff(A[i], B[i])
        if k is None and len(A[i]) == ntok:
            print(f"| {i} `{p}` | **TOKEN-EXACT {ntok}/{ntok}** | — | — | — |")
        else:
            div += 1
            ka = A[i][k] if k is not None and k < len(A[i]) else "-"
            kb = B[i][k] if k is not None and k < len(B[i]) else "-"
            print(f"| {i} `{p}` | DIVERGE | {k} | {ka} | {kb} |")
    print(f"\nDIVERGENCES({name_a} vs {name_b}) = {div}/{len(PROMPTS)}")
    return div


if __name__ == "__main__":
    prompts_txt, ours_json, vllm_eager, vllm_compiled, ntok_s = sys.argv[1:6]
    ntok = int(ntok_s)
    # The REPEAT legs are optional arguments rather than a separate script, so
    # ONE artifact carries the whole verdict. A determinism claim that lives in
    # a different step from the verdict is a claim a reader has to assemble.
    rep_eager = sys.argv[6] if len(sys.argv) > 6 else None
    rep_compiled = sys.argv[7] if len(sys.argv) > 7 else None

    raw = Path(prompts_txt).read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    print("WHAT IS COMPARED AGAINST WHAT")
    print(f"  prompts file        : {prompts_txt}")
    print(f"  prompts_sha256      : {got}")
    print(f"  prompts_sha256_want : {PROMPTS_SHA256}")
    print(f"  PROMPTS_ARE_THE_PREREGISTERED_SET = {got == PROMPTS_SHA256}")
    print(f"  tokens per prompt   : {ntok}")
    print(f"  side A (vllm.cpp ROCm)  : {ours_json}   (vllm-bench --output-token-ids)")
    print(f"  side B (vLLM eager)     : {vllm_eager}")
    print(f"  side C (vLLM compiled)  : {vllm_compiled}")
    if got != PROMPTS_SHA256:
        print("STRICT_LIMB3=ERROR")
        print("REASON=the prompt file is not the pre-registered one; nothing is scored")
        sys.exit(2)

    missing = [x for x in (ours_json, vllm_eager, vllm_compiled)
               if not (Path(x).exists() and Path(x).stat().st_size)]
    if missing:
        # A missing side is an INSTRUMENT outcome, not a divergence. Scoring
        # around it would turn a leg that never ran into a token mismatch.
        print("STRICT_LIMB3=NOT_MEASURED")
        print("REASON=these inputs are absent or empty, so nothing is scored: "
              + ", ".join(missing))
        sys.exit(0)

    A = read_ours(ours_json)
    B, mb = read_vllm(vllm_eager)
    C, mc = read_vllm(vllm_compiled)
    for nm, X in (("A vllm.cpp ROCm k-quant", A),
                  ("B vLLM eager", B),
                  ("C vLLM compiled", C)):
        print(f"{nm}: {len(X)} prompts, lengths {[len(x) for x in X]}")

    # THE ORACLE'S OWN SELF-CONSISTENCY, FIRST. A vehicle whose oracle
    # disagrees with itself across a compilation flag is not a denominator, and
    # picking the configuration that agrees with us would be the error this
    # campaign has already made once.
    d_bc = score("B vLLM eager", B, "C vLLM compiled", C, ntok)
    print(f"\nORACLE_SELF_CONSISTENT_ACROSS_COMPILE = {d_bc == 0}")
    repro = {}
    for nm, first, rep in (("eager", B, rep_eager),
                           ("compiled", C, rep_compiled)):
        if rep is None or not (Path(rep).exists() and Path(rep).stat().st_size):
            print(f"ORACLE_REPRODUCES_ITSELF[{nm}] = NOT_MEASURED")
            repro[nm] = None
            continue
        again, _ = read_vllm(rep)
        repro[nm] = again == first
        print(f"ORACLE_REPRODUCES_ITSELF[{nm}] = {repro[nm]}  ({rep})")
    pf_b = mb.get("prefill")
    pf_c = mc.get("prefill")
    for nm, pf in (("eager", pf_b), ("compiled", pf_c)):
        if pf is None:
            print(f"PREFILL_ARGMAX[{nm}] = NOT_MEASURED")
        else:
            n = sum(len(p["mismatches"]) for p in pf)
            print(f"PREFILL_ARGMAX[{nm}]_MISMATCHES = {n}"
                  f"  AGREES_WITH_ITS_OWN_DECODE = {n == 0}")

    d_ab = score("A vllm.cpp", A, "B vLLM eager", B, ntok)
    d_ac = score("A vllm.cpp", A, "C vLLM compiled", C, ntok)

    print()
    print(f"STRICT_EXACT_vs_VLLM_EAGER    = {len(PROMPTS) - d_ab}/{len(PROMPTS)}")
    print(f"STRICT_EXACT_vs_VLLM_COMPILED = {len(PROMPTS) - d_ac}/{len(PROMPTS)}")

    # THE VERDICT. Limb 3 is satisfied only by a clean strict pass against a
    # deterministic oracle. Both halves are required and neither substitutes
    # for the other.
    if any(v is False for v in repro.values()):
        bad = [k for k, v in repro.items() if v is False]
        print("STRICT_LIMB3=NOT_MEASURED")
        print("REASON=the oracle does not reproduce ITSELF on this vehicle in "
              f"{bad}; a denominator that changes between two identical runs "
              "cannot be compared against, and nothing here is scored as a "
              "divergence of ours")
    elif d_bc != 0:
        print("STRICT_LIMB3=NO")
        print("REASON=the ORACLE is not deterministic on this vehicle: its own "
              "eager and compiled configurations disagree with each other, so "
              "there is no single denominator to be exact against")
    elif d_ab == 0 and d_ac == 0:
        print("STRICT_LIMB3=YES")
        print("REASON=free-running token-exact on every prompt against a "
              "self-consistent pinned oracle")
    else:
        print("STRICT_LIMB3=NO")
        print(f"REASON=the oracle is self-consistent and our arm diverges on "
              f"{d_ab} of {len(PROMPTS)} prompts")
