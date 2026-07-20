#!/usr/bin/env python3
# ADDITIVE-MODEL W4 — capture the vLLM 0.25.0 oracle greedy continuation for
# Qwen3-0.6B on the standard prompt battery, dumping the committed token-exact
# golden tests/parity/goldens/qwen3_greedy_0_6b/greedy_ids.npy ([N, T] i32).
#
# The prompt list MUST match tests/parity/test_qwen3_paged_engine.cpp::Prompts()
# exactly (goldens and gate never drift). Run on dgx with the oracle venv:
#   ~/venvs/vllm-oracle/bin/python scripts/qwen3-oracle-capture.py
import os
import sys
import numpy as np

PROMPTS = [
    "The capital of France is",
    "Once upon a time,",
    "In the beginning God created",
    "The quick brown fox jumps over",
    "def fibonacci(n):",
    "Water boils at a temperature of",
    "The theory of relativity was developed by",
    "To be or not to be, that is",
    "The largest planet in our solar system is",
    "Machine learning is a subfield of",
    "The mitochondria is the powerhouse of",
    "Roses are red, violets are",
    "The first president of the United States was",
    "E equals m c",
    "A journey of a thousand miles begins with",
    "The chemical symbol for gold is",
]
MAX_TOKENS = 16

def main():
    from vllm import LLM, SamplingParams

    model = os.environ.get("QWEN3_MODEL", "Qwen/Qwen3-0.6B")
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tests", "parity", "goldens", "qwen3_greedy_0_6b",
    )
    os.makedirs(out_dir, exist_ok=True)

    llm = LLM(model=model, dtype="bfloat16", enforce_eager=True)
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
    outs = llm.generate(PROMPTS, sp)

    # vLLM may reorder; map back by prompt.
    by_prompt = {o.prompt: o for o in outs}
    greedy = np.zeros((len(PROMPTS), MAX_TOKENS), dtype="<i4")
    for i, p in enumerate(PROMPTS):
        o = by_prompt[p]
        ids = list(o.outputs[0].token_ids)
        if len(ids) != MAX_TOKENS:
            print(f"WARNING prompt[{i}] produced {len(ids)} tokens (< {MAX_TOKENS})",
                  file=sys.stderr)
        for j in range(min(MAX_TOKENS, len(ids))):
            greedy[i, j] = ids[j]
        # Also dump the oracle prompt tokenization (diagnostic).
        np.array(o.prompt_token_ids, dtype="<i4").tofile(
            os.path.join(out_dir, f"p{i}_prompt.i32"))
        print(f"prompt[{i}] {p!r} -> {ids}  ({o.outputs[0].text!r})")

    np.save(os.path.join(out_dir, "greedy_ids.npy"), greedy)
    print(f"wrote {out_dir}/greedy_ids.npy shape {greedy.shape}")

if __name__ == "__main__":
    main()
