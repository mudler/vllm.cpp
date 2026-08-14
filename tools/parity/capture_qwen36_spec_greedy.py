#!/usr/bin/env python3
"""Capture the vLLM 0.25.0 35B (nvidia/Qwen3.6-35B-A3B-NVFP4) MTP spec-decode
greedy continuation + acceptance, for the M-mtp-2 three-way identity gate.

This is the oracle side of tests/parity/test_qwen36_spec_decode.cpp: it confirms,
DIRECTLY (not only transitively), that vLLM's own greedy continuation is the
SAME with spec-decode ON (method mtp, k=1) and OFF, and equal to the committed
`tests/parity/goldens/qwen36_logits_35b/greedy_ids.npy` anchor. It also reports
vLLM's own draft-acceptance so our 16/16 can be compared to vLLM's on the same
prompt. Mirrors the LLM setup of tools/parity/dump_qwen3_5_mtp.py
(enforce_eager, bfloat16, conservative gpu_memory_utilization).

Usage (dgx, in the vllm-oracle venv, under flock "${GPU_LOCK:-$HOME/gpu.lock}"):
  python tools/parity/capture_qwen36_spec_greedy.py \
      --model <35B-snapshot> --spec on  --max-tokens 32 --gpu-mem 0.45
  python tools/parity/capture_qwen36_spec_greedy.py \
      --model <35B-snapshot> --spec off --max-tokens 32 --gpu-mem 0.45
"""
import argparse
import json


PROMPT = "The capital of France is Paris, and the"
# qwen36_logits_35b/greedy_ids.npy (the committed anchor); duplicated here so the
# capture is self-checking without loading numpy against the golden dir.
GREEDY_IDS_16 = [6511, 314, 9564, 369, 19241, 13, 198, 760,
                 6511, 314, 9338, 369, 11751, 11, 321, 279]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--spec", choices=["on", "off"], required=True)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--gpu-mem", type=float, default=0.45)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=args.model,
        enforce_eager=True,
        max_model_len=256,
        gpu_memory_utilization=args.gpu_mem,
        dtype="bfloat16",
        disable_log_stats=False,
    )
    if args.spec == "on":
        kwargs["speculative_config"] = {
            "method": "mtp", "num_speculative_tokens": 1}

    llm = LLM(**kwargs)

    out = llm.generate(
        [PROMPT],
        SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
    )
    gen = list(out[0].outputs[0].token_ids)
    prompt_ids = list(out[0].prompt_token_ids)

    prefix16 = gen[:16]
    matches_anchor = prefix16 == GREEDY_IDS_16

    # Best-effort acceptance extraction from vLLM V1 metrics.
    accepted = draft = None
    try:
        for m in llm.get_metrics():
            name = getattr(m, "name", "")
            val = getattr(m, "value", None)
            if name.endswith("spec_decode_num_accepted_tokens") or \
               name.endswith("num_accepted_tokens_total"):
                accepted = val
            if name.endswith("spec_decode_num_draft_tokens") or \
               name.endswith("num_draft_tokens_total"):
                draft = val
    except Exception as exc:  # noqa: BLE001
        accepted = draft = None
        print(f"[capture] metrics unavailable: {exc}")

    accept_rate = None
    if accepted is not None and draft:
        accept_rate = accepted / draft

    result = {
        "spec": args.spec,
        "prompt": PROMPT,
        "prompt_ids": prompt_ids,
        "generated_ids": gen,
        "prefix16": prefix16,
        "greedy_ids_anchor": GREEDY_IDS_16,
        "prefix16_matches_anchor": matches_anchor,
        "vllm_num_accepted_tokens": accepted,
        "vllm_num_draft_tokens": draft,
        "vllm_acceptance_rate": accept_rate,
        "decoded_text": out[0].outputs[0].text,
    }
    print("CAPTURE_JSON " + json.dumps(result))


if __name__ == "__main__":
    main()
