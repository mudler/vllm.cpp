#!/usr/bin/env python3
"""Pinned Gemma 3 prompts and production-config vLLM token capture."""

import argparse
import json
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("generate", "capture", "bench", "dtype"))
    p.add_argument("model")
    p.add_argument("manifest", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--kv-cache-memory-bytes", type=int)
    args = p.parse_args()
    if args.kv_cache_memory_bytes is not None and args.kv_cache_memory_bytes <= 0:
        p.error("--kv-cache-memory-bytes must be positive")
    if args.mode == "generate":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        prompts = [
            "Explain how a computer uses memory to store information. " * 12,
            "Read these observations: the garden contains roses, lavender, and mint. "
            * 36
            + "\nSummarize the observations in one short sentence.",
            "The river flows through the valley and provides water for the village. "
            * 92
            + "\nWhat does the river provide?",
        ]
        cases = []
        for i, text in enumerate(prompts):
            ids = tokenizer.encode(text, add_special_tokens=True)
            if not 64 <= len(ids) < 4000:
                raise RuntimeError(f"prompt misses the attention gate: {len(ids)}")
            cases.append(dict(name=f"prompt{i}", text=text, prompt_token_ids=ids))
        args.manifest.write_text(
            json.dumps(dict(output_len=32, cases=cases), indent=2) + "\n"
        )
        print("PROMPT LENGTHS", [len(c["prompt_token_ids"]) for c in cases])
        return
    import vllm
    from vllm import LLM, SamplingParams

    if "e126687a9" not in vllm.__version__:
        raise RuntimeError(f"wrong primary pin: {vllm.__version__}")
    manifest = json.loads(args.manifest.read_text())
    # Keep production compilation/graphs and the default attention selection.
    cache_kwargs = (
        {}
        if args.kv_cache_memory_bytes is None
        else {"kv_cache_memory_bytes": args.kv_cache_memory_bytes}
    )
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=4096,
        max_num_seqs=1,
        gpu_memory_utilization=0.85,
        seed=0,
        block_size=manifest.get("block_size", 16),
        **cache_kwargs,
    )
    if args.mode == "dtype":

        def inspect_head(model):
            import torch

            weight = model.lm_head.weight
            hidden = torch.zeros(
                (1, weight.shape[1]), device=weight.device, dtype=weight.dtype
            )
            logits = model.compute_logits(hidden)
            return dict(
                weight_dtype=str(weight.dtype),
                hidden_dtype=str(hidden.dtype),
                projection_dtype=str(logits.dtype),
                configured_head_dtype=str(model.logits_processor.head_dtype),
            )

        args.output.write_text(
            json.dumps(llm.apply_model(inspect_head), indent=2) + "\n"
        )
        return
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=manifest["output_len"],
        ignore_eos=True,
        logprobs=None if args.mode == "bench" else 5,
    )
    report = {
        "vllm_version": vllm.__version__,
        "block_size": manifest.get("block_size", 16),
        "cases": [],
    }
    report["kv_cache_memory_bytes"] = args.kv_cache_memory_bytes
    if args.mode == "bench":
        import importlib.util

        path = Path(__file__).resolve().parents[1] / "bench/vllm_closed_loop_metrics.py"
        spec = importlib.util.spec_from_file_location("closed_loop_metrics", path)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        if "warmup" in manifest:
            warm = llm.generate(
                [dict(prompt_token_ids=manifest["warmup"]["prompt_token_ids"])],
                sampling,
            )[0]
            report["warmup_output_token_ids"] = list(warm.outputs[0].token_ids)
        ids = [entry["prompt_token_ids"] for entry in manifest["cases"]]
        duration, records = helper.run_closed_loop(
            llm, [dict(prompt_token_ids=values) for values in ids], sampling, 1, 0
        )
        metrics, outputs = helper.derive_metrics(
            records,
            ids,
            duration,
            output_len=manifest["output_len"],
            max_concurrency=1,
            async_scheduling="default",
        )
        report.update(
            metrics=metrics,
            records=records,
            prompt_token_ids=ids,
            output_token_ids=outputs,
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print("BENCH", json.dumps(metrics), flush=True)
        return
    for entry in manifest["cases"]:
        start = time.perf_counter()
        result = llm.generate(
            [dict(prompt_token_ids=entry["prompt_token_ids"])], sampling
        )[0]
        seconds = time.perf_counter() - start
        out = result.outputs[0]
        report["cases"].append(
            dict(
                name=entry["name"],
                prompt_token_ids=result.prompt_token_ids,
                output_token_ids=list(out.token_ids),
                text=out.text,
                seconds=seconds,
                logprobs=[
                    {str(k): v.logprob for k, v in row.items()} for row in out.logprobs
                ],
            )
        )
        print("PRIMARY PUBLIC", entry["name"], list(out.token_ids), flush=True)
    if args.output is None:
        raise ValueError("capture requires --output")
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
