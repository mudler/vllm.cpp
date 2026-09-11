#!/usr/bin/env python3
"""One timed leg of the PINNED vLLM arm of the multi-engine survey (#2497).

vLLM's production configuration is what counts: AGENTS.md Gates forbids
`--enforce-eager` as a denominator, so EAGER defaults to 0 here and the
compiled arm is the headline. An eager leg is a SECOND configuration and is
labelled as one wherever it is reported.

The prompt token ids are FED rather than re-tokenized, so the workload is the
same token sequence the other two arms decode from and no tokenizer difference
can enter the timing.

`gen_rocm.py`'s __main__ guard is kept for the same reason it has one: vLLM V1
spawns EngineCore with multiprocessing "spawn" and re-imports this file in the
child (#2624).
"""
import json
import os
import time

MODEL = os.environ["MODEL"]
TOK = os.environ["TOK"]
MMPROJ = os.environ.get("MMPROJ") or None
OUT = os.environ["OUT_JSON"]
QUANT = os.environ.get("QUANT") or None
GMU = float(os.environ.get("GMU", "0.75"))
MAXLEN = int(os.environ.get("MAXLEN", "4096"))
EAGER = os.environ.get("EAGER", "0") == "1"
NGEN = int(os.environ.get("NGEN", "64"))
REPEAT = int(os.environ.get("REPEAT", "4"))
PROMPT = os.environ.get("PROMPT", "The capital of France is")
TAG = os.environ.get("LEG_TAG", "unknown")


def main() -> None:
    import vllm

    print("VLLM_FILE    =", vllm.__file__, flush=True)
    print("VLLM_VERSION =", vllm.__version__, flush=True)
    import torch

    print("TORCH        =", torch.__version__, "hip", torch.version.hip, flush=True)
    print("DEVICE       =", torch.cuda.get_device_properties(0).gcnArchName, flush=True)
    # The one knob that would make this measurement a measurement of a different
    # device. It stops the run rather than printing a line somebody has to read.
    assert os.environ.get("HSA_OVERRIDE_GFX_VERSION") is None, "HSA_OVERRIDE_GFX_VERSION set"
    for k in sorted(os.environ):
        if k.startswith(("HSA_", "ROCR_", "PYTORCH_", "HIP_")):
            print("INHERITED_ENV", k, "=", os.environ[k], flush=True)
    from vllm.platforms.rocm import on_gfx1151

    print("ON_GFX1151   =", on_gfx1151(), flush=True)

    if QUANT == "gguf":
        import vllm_gguf_plugin
        import vllm_gguf_plugin._C_gguf as _cg

        print("PLUGIN_FILE  =", vllm_gguf_plugin.__file__, flush=True)
        print("PLUGIN_C_EXT =", _cg.__file__, flush=True)

    from transformers import AutoTokenizer

    tk = AutoTokenizer.from_pretrained(TOK)
    ids = tk(PROMPT, add_special_tokens=False)["input_ids"]
    print("PROMPT       =", repr(PROMPT), flush=True)
    print("PROMPT_IDS   =", ids, "len", len(ids), flush=True)

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    def build(gmu, maxlen):
        kw = dict(
            model=MODEL,
            tokenizer=TOK,
            gpu_memory_utilization=gmu,
            max_model_len=maxlen,
            max_num_seqs=1,
            max_num_batched_tokens=maxlen,
            limit_mm_per_prompt={"image": 0, "video": 0},
            enforce_eager=EAGER,
            trust_remote_code=False,
        )
        if QUANT:
            kw["quantization"] = QUANT
        if MMPROJ:
            kw["model_loader_extra_config"] = {"mm_proj": MMPROJ}
        return kw

    used_gmu, used_maxlen = GMU, MAXLEN
    t_load0 = time.perf_counter()
    try:
        kw = build(GMU, MAXLEN)
        print("LLM_KWARGS   =", kw, flush=True)
        llm = LLM(**kw)
    except Exception as exc:  # noqa: BLE001
        # The fallback is DECLARED, not silent: which pair actually built the
        # engine is written into the result file and reported beside the figure.
        print("PRIMARY_CONFIG_FAILED:", type(exc).__name__, exc, flush=True)
        used_gmu, used_maxlen = 0.60, 2048
        kw = build(used_gmu, used_maxlen)
        print("FALLBACK_LLM_KWARGS =", kw, flush=True)
        llm = LLM(**kw)
    load_secs = time.perf_counter() - t_load0
    print("ENGINE_UP load_secs=%.3f gmu=%s maxlen=%s eager=%s"
          % (load_secs, used_gmu, used_maxlen, EAGER), flush=True)

    tp = TokensPrompt(prompt_token_ids=ids)

    def timed(max_tokens):
        sp = SamplingParams(temperature=0.0, top_p=1.0,
                            max_tokens=max_tokens, ignore_eos=True)
        t0 = time.perf_counter()
        outs = llm.generate([tp], sp, use_tqdm=False)
        secs = time.perf_counter() - t0
        gen = list(outs[0].outputs[0].token_ids)
        return secs, gen

    runs = []
    for i in range(1, REPEAT + 1):
        secs, gen = timed(NGEN)
        rec = {"run": i, "of": REPEAT, "secs": secs,
               "completion_tokens": len(gen),
               "tok_s": len(gen) / secs if secs > 0 else None,
               "gen_ids": gen}
        runs.append(rec)
        print("vllm: run=%d/%d completion_tokens=%d secs=%.3f tok_s=%.3f"
              % (i, REPEAT, len(gen), secs, rec["tok_s"]), flush=True)

    # DECODE ISOLATION, measured two independent ways, because the pilot showed
    # the correction is large: 3.85 s of a 9.50 s 64-token request was not decode.
    # A figure that big cannot rest on one sample of one special case.
    #
    #   (a) the one-token call, repeated, median taken. It is the cheapest probe
    #       and it is also the most suspect, because a 1-token request may take a
    #       different cudagraph shape than a 64-token one.
    #   (b) the SLOPE between two real token counts, NGEN and 2*NGEN. Both points
    #       are ordinary requests on the same path, so no special case enters, and
    #       the slope cancels every fixed per-request cost exactly.
    # (b) is the figure to trust. (a) is reported beside it as a cross-check, and
    # where the two disagree the document says so rather than picking one.
    one_runs = []
    for _ in range(3):
        s1, g1 = timed(1)
        one_runs.append(s1)
        print("vllm: one_token_secs=%.4f completion_tokens=%d" % (s1, len(g1)), flush=True)
    one_secs = sorted(one_runs)[len(one_runs) // 2]

    two_runs = []
    for _ in range(2):
        s2, g2 = timed(2 * NGEN)
        two_runs.append({"secs": s2, "completion_tokens": len(g2)})
        print("vllm: double_secs=%.4f completion_tokens=%d" % (s2, len(g2)), flush=True)

    payload = {
        "tag": TAG,
        "vllm_version": __import__("vllm").__version__,
        "model": MODEL,
        "quantization": QUANT,
        "enforce_eager": EAGER,
        "gpu_memory_utilization": used_gmu,
        "max_model_len": used_maxlen,
        "requested_gpu_memory_utilization": GMU,
        "requested_max_model_len": MAXLEN,
        "config_fallback_used": (used_gmu, used_maxlen) != (GMU, MAXLEN),
        "n_gen": NGEN,
        "repeat": REPEAT,
        "prompt": PROMPT,
        "prompt_ids": ids,
        "load_secs": load_secs,
        "one_token_secs": one_secs,
        "one_token_runs": one_runs,
        "double_runs": two_runs,
        "double_n_gen": 2 * NGEN,
        "runs": runs,
    }
    with open(OUT, "w") as fh:
        json.dump(payload, fh, indent=1)
    print("WROTE", OUT, flush=True)
    print("DONE_MARKER_VLLM_SURVEY_LEG", flush=True)


if __name__ == "__main__":
    main()
