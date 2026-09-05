#!/usr/bin/env python3
# ORACLE-VLLM-GGUF-QWEN35 / #2624
# Does the PINNED vLLM, plus the first-party vllm-gguf-plugin, load our
# Qwen3.8-27B Q4_K_M GGUF and EMIT TOKENS? AGENTS.md makes that the
# gateability bar: constructing a config proves nothing.
#
# Correctness only. NO timing is taken and none may be read out of this file.
import json
import os
import sys

MODEL = os.environ["MODEL"]
TOK = os.environ["TOK"]
MMPROJ = os.environ["MMPROJ"]
OUT = os.environ["OUT_JSON"]
GMU = float(os.environ.get("GMU", "0.30"))
MAXLEN = int(os.environ.get("MAXLEN", "2048"))

# The SIX prompts of the recorded Q4_K_M token gate, verbatim from
# docs/bench-evidence/qwen38-27b-q4km-token-gate-20260823.md. Raw completion,
# no chat template, greedy, 48 tokens, ignore_eos, concurrency 1.
PROMPTS = [
    "The capital city of France is",
    "The three primary colors are",
    "Water boils at a temperature of",
    "The Pythagorean theorem states that",
    "In 1969, humans first walked on",
    "A prime number is a natural number",
]
# The oracle's own PROMPT_IDS lengths, recorded in that evidence file. A
# mismatch here means the tokenizations are NOT the same and no id comparison
# below would be meaningful.
EXPECTED_PROMPT_LENS = [6, 5, 6, 7, 11, 7]


# vLLM V1 spawns its EngineCore with multiprocessing "spawn", which RE-IMPORTS
# this file in the child. Without a __main__ guard the module body runs again
# there and the engine dies with the freeze_support() bootstrap error, printing
# `Engine core initialization failed` — an INSTRUMENT failure that reads like a
# model failure. Measured on thor:gpu0, run 20260903T005757Z.
def main() -> None:
    import vllm

    print("VLLM_FILE    =", vllm.__file__, flush=True)
    print("VLLM_VERSION =", vllm.__version__, flush=True)
    assert "site-packages" in vllm.__file__, vllm.__file__
    assert "555967922" in vllm.__version__, vllm.__version__

    # The plugin must be the INSTALLED distribution, and its CUDA extension must
    # import. Either failing here is an instrument failure, never a verdict about
    # the model.
    import vllm_gguf_plugin  # noqa: E402

    print("PLUGIN_FILE  =", vllm_gguf_plugin.__file__, flush=True)
    import vllm_gguf_plugin._C_gguf as _cg  # noqa: E402

    print("PLUGIN_C_EXT =", _cg.__file__, flush=True)

    from transformers import AutoTokenizer  # noqa: E402

    tk = AutoTokenizer.from_pretrained(TOK)
    ids = [tk(p, add_special_tokens=False)["input_ids"] for p in PROMPTS]
    lens = [len(i) for i in ids]
    print("PROMPT_LENS  =", lens, flush=True)
    print("PROMPT_LENS_EXPECTED =", EXPECTED_PROMPT_LENS, flush=True)
    print("PROMPT_LENS_MATCH =", lens == EXPECTED_PROMPT_LENS, flush=True)
    for i, seq in enumerate(ids):
        print(f"PROMPT_IDS[{i}] = {seq}", flush=True)

    from vllm import LLM, SamplingParams  # noqa: E402
    from vllm.inputs import TokensPrompt  # noqa: E402

    # enforce_eager: this box has OOM-REBOOTED in the step AFTER torch.compile
    # (.agents/specs/mtp-k-gt-1.md). This run asks a CORRECTNESS question only, so
    # skipping compilation costs nothing it needs. AGENTS.md forbids eager as a
    # PERFORMANCE denominator; no performance number is taken here.
    # attention_backend FLASHINFER: vLLM's default FLASH_ATTN carries no sm_12x
    # SASS on this device (/workspace/oracle-vllm/README-WHEELS.md).
    # limit_mm_per_prompt zeroed and max_num_seqs=1: vLLM profiles memory with a
    # DUMMY forward, and on a 27B vision-language model the dummy image is where an
    # unbounded allocation would come from. This asks a TEXT question, so the
    # vision path is loaded (its weights are part of the checkpoint) and never
    # profiled with dummy inputs.
    llm = LLM(
        model=MODEL,
        tokenizer=TOK,
        quantization="gguf",
        attention_backend="FLASHINFER",
        gpu_memory_utilization=GMU,
        max_model_len=MAXLEN,
        max_num_seqs=1,
        max_num_batched_tokens=MAXLEN,
        limit_mm_per_prompt={"image": 0, "video": 0},
        enforce_eager=True,
        trust_remote_code=False,
        model_loader_extra_config={"mm_proj": MMPROJ},
    )
    print("ENGINE_UP", flush=True)

    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=48, ignore_eos=True)
    outs = llm.generate([TokensPrompt(prompt_token_ids=s) for s in ids], sp)

    rec = []
    for i, o in enumerate(outs):
        gen = list(o.outputs[0].token_ids)
        txt = o.outputs[0].text
        rec.append({"i": i, "prompt": PROMPTS[i], "prompt_ids": ids[i],
                    "gen_ids": gen, "gen_text": txt})
        print(f"GEN_IDS[{i}] = {gen}", flush=True)
        print(f"GEN_TEXT[{i}] = {txt!r}", flush=True)
        print(f"GEN_LEN[{i}] = {len(gen)}", flush=True)

    with open(OUT, "w") as fh:
        json.dump({"vllm_version": vllm.__version__, "model": MODEL,
                   "tokenizer": TOK, "records": rec}, fh, indent=1)
    print("WROTE", OUT, flush=True)
    assert all(len(r["gen_ids"]) == 48 for r in rec), "not every prompt emitted 48"
    print("DONE_MARKER_GGUFPLUGIN_GEN", flush=True)


if __name__ == "__main__":
    main()
