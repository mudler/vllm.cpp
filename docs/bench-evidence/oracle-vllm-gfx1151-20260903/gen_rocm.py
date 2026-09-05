#!/usr/bin/env python3
# #2740 -- does the PINNED vLLM emit tokens for Qwen3.8-27B on gfx1151?
# AGENTS.md's gateability bar: constructing a config proves nothing.
#
# CORRECTNESS ONLY. No timing is taken and no number here is a performance
# result: this arm's declared token gate reads FAIL and AGENTS.md Gates admits
# no performance result from it.
import json
import os

MODEL = os.environ["MODEL"]
TOK = os.environ["TOK"]
MMPROJ = os.environ.get("MMPROJ") or None
OUT = os.environ["OUT_JSON"]
QUANT = os.environ.get("QUANT") or None          # "gguf" or unset for bf16
GMU = float(os.environ.get("GMU", "0.60"))
MAXLEN = int(os.environ.get("MAXLEN", "2048"))
EAGER = os.environ.get("EAGER", "1") == "1"

# The SIX prompts of the declared Q4_K_M token gate, byte for byte. Their file
# form (one per line, newline-terminated) hashes to
# c8a5080046c3206a1186b42a320a21e887691eff43d2116364192f58e5776c7e, which is the
# prompts_sha256 the gfx1151 v2 evidence records.
PROMPTS = [
    "The capital city of France is",
    "The three primary colors are",
    "Water boils at a temperature of",
    "The Pythagorean theorem states that",
    "In 1969, humans first walked on",
    "A prime number is a natural number",
]
# The llama.cpp b10451 oracle's own PROMPT_IDS, recorded in
# /workspace/rocm-tokgate-strix-v2/.../oracle_hip.txt. These are FED to the
# engine rather than re-tokenized, so the comparison is about generation and
# cannot be contaminated by a tokenizer difference. They are ALSO re-derived
# from the tokenizer and the two are asserted equal.
ORACLE_PROMPT_IDS = [
    [760, 6511, 3177, 314, 9338, 369],
    [760, 2250, 5839, 7736, 513],
    [27336, 85895, 506, 264, 9039, 314],
    [760, 5187, 92068, 43687, 55877, 5134, 421],
    [623, 220, 16, 24, 21, 24, 11, 12313, 1118, 14428, 383],
    [32, 9944, 1324, 369, 264, 5629, 1324],
]


# vLLM V1 spawns EngineCore with multiprocessing "spawn", which RE-IMPORTS this
# file in the child. Without a __main__ guard the module body runs again there
# and the engine dies with the freeze_support bootstrap error -- an INSTRUMENT
# failure that reads exactly like a model failure (#2624, run 20260903T005757Z).
def main() -> None:
    import vllm

    print("VLLM_FILE    =", vllm.__file__, flush=True)
    print("VLLM_VERSION =", vllm.__version__, flush=True)

    import torch

    print("TORCH        =", torch.__version__, "hip", torch.version.hip, flush=True)
    print("DEVICE       =", torch.cuda.get_device_properties(0).gcnArchName, flush=True)
    assert os.environ.get("HSA_OVERRIDE_GFX_VERSION") is None, "override set"
    from vllm.platforms.rocm import on_gfx1151, on_gfx1x

    print("ON_GFX1151   =", on_gfx1151(), " ON_GFX1X =", on_gfx1x(), flush=True)

    if QUANT == "gguf":
        import vllm_gguf_plugin

        print("PLUGIN_FILE  =", vllm_gguf_plugin.__file__, flush=True)
        import vllm_gguf_plugin._C_gguf as _cg

        print("PLUGIN_C_EXT =", _cg.__file__, flush=True)

    from transformers import AutoTokenizer

    tk = AutoTokenizer.from_pretrained(TOK)
    ids = [tk(p, add_special_tokens=False)["input_ids"] for p in PROMPTS]
    print("PROMPT_LENS          =", [len(i) for i in ids], flush=True)
    print("PROMPT_IDS_MATCH_ORACLE =", ids == ORACLE_PROMPT_IDS, flush=True)
    for i, s in enumerate(ids):
        print(f"TOKENIZER_IDS[{i}] = {s}", flush=True)
    # FEED the oracle's ids. If the tokenizers disagree the line above says so
    # and the generation comparison is still well posed.
    feed = ORACLE_PROMPT_IDS

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    kw = dict(
        model=MODEL,
        tokenizer=TOK,
        gpu_memory_utilization=GMU,
        max_model_len=MAXLEN,
        max_num_seqs=1,
        max_num_batched_tokens=MAXLEN,
        limit_mm_per_prompt={"image": 0, "video": 0},
        enforce_eager=EAGER,
        trust_remote_code=False,
    )
    if QUANT:
        kw["quantization"] = QUANT
    if MMPROJ:
        kw["model_loader_extra_config"] = {"mm_proj": MMPROJ}
    print("LLM_KWARGS   =", {k: v for k, v in kw.items()}, flush=True)
    llm = LLM(**kw)
    print("ENGINE_UP", flush=True)

    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=48, ignore_eos=True)
    outs = llm.generate([TokensPrompt(prompt_token_ids=s) for s in feed], sp)

    rec = []
    for i, o in enumerate(outs):
        gen = list(o.outputs[0].token_ids)
        rec.append({"i": i, "prompt": PROMPTS[i], "prompt_ids": feed[i],
                    "gen_ids": gen, "gen_text": o.outputs[0].text})
        print(f"GEN_IDS[{i}] = {gen}", flush=True)
        print(f"GEN_TEXT[{i}] = {o.outputs[0].text!r}", flush=True)
        print(f"GEN_LEN[{i}] = {len(gen)}", flush=True)

    with open(OUT, "w") as fh:
        json.dump({"vllm_version": vllm.__version__, "model": MODEL,
                   "quantization": QUANT, "enforce_eager": EAGER,
                   "gpu_memory_utilization": GMU, "records": rec}, fh, indent=1)
    print("WROTE", OUT, flush=True)
    assert all(len(r["gen_ids"]) == 48 for r in rec), "not every prompt emitted 48"
    print("DONE_MARKER_VLLM_GFX1151_GEN", flush=True)


if __name__ == "__main__":
    main()
