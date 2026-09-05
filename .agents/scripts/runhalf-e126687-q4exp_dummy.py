#!/usr/bin/env python3
"""RUNHALF stretch leg (#2611): does the qwen4_exp GRAPH execute at e126687a9a?

THIS IS NOT A PARITY MEASUREMENT AND NOT A TOKEN GATE. The weights are RANDOM
(load_format="dummy") and the config is SHRUNK, so every token it emits is
meaningless. Upstream's own tests/models/registry.py marks all three Qwen4Exp
architectures is_available_online=False, and every published safetensors arm of
Qwen3.8-Flash-Next exceeds the largest fleet box, so this is the strongest
statement about the model itself that the fleet can currently support.

What a green here says: the qwen4_exp forward, its hybrid linear/full attention
KV path, its MoE routing and the sampler all execute on this device at this
revision. What it cannot say: anything about numerics.

The shrink is recorded field by field so a reader can see exactly which
architecture ran.
"""
import copy
import json
import os
import sys

def main():
    src = sys.argv[1]
    import torch
    import vllm
    print("VLLM_VERSION", vllm.__version__, flush=True)
    print("DEVICE", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), flush=True)

    with open(os.path.join(src, "config.json")) as f:
        cfg = json.load(f)
    orig = copy.deepcopy(cfg)

    t = cfg["text_config"]
    # Keep ONE full_attention layer and the linear_attention layers around it,
    # so both branches of the hybrid stack are exercised.
    t["layer_types"] = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
    t["num_hidden_layers"] = 4
    t["num_experts"] = 8
    t["num_experts_per_tok"] = 2
    # Width is left ALONE. An earlier leg halved head_dim to 128 and tripped
    # `assert sum(self.mrope_section) == rotary_dim // 2` in
    # model_executor/layers/rotary_embedding/mrope.py: mrope_section [11,11,10]
    # sums to 32, and with partial_rotary_factor 0.25 that invariant pins
    # head_dim at 256. Shrinking width means re-deriving mrope_section, the
    # indexer dims and hc_lowrank together, and a config nobody validated is
    # not a better statement about the architecture than a deep one. Only
    # DEPTH, expert COUNT and the two unbounded vocabularies are reduced.
    t["max_position_embeddings"] = 4096
    # ngram_vocab_size_base = 20,000,000 is an embedding no shrink can carry.
    t["ngram_vocab_size_base"] = 20000
    t["ple_layer_ids"] = [2]
    if "mtp" in t:
        t["mtp"]["num_hidden_layers"] = 1
    v = cfg.get("vision_config")
    if v:
        v["depth"] = 2
        v["hidden_size"] = 256
        v["intermediate_size"] = 512
        v["num_heads"] = 4
        v["out_hidden_size"] = 512

    dst = os.environ.get("Q4EXP_DST", "/tmp/q4exp-shrunk")
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        if name == "config.json":
            continue
        s = os.path.join(src, name)
        if os.path.isfile(s):
            with open(s, "rb") as a, open(os.path.join(dst, name), "wb") as b:
                b.write(a.read())
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    for k in ("num_hidden_layers", "num_experts", "hidden_size", "num_attention_heads",
              "ngram_vocab_size_base", "vocab_size", "layer_types"):
        print("Q4 SHRUNK %s: %r -> %r" % (k, orig["text_config"].get(k), cfg["text_config"].get(k)), flush=True)
    print("Q4 ARCHITECTURES", cfg["architectures"], flush=True)

    from vllm import LLM, SamplingParams
    kw = dict(model=dst, dtype="bfloat16", load_format="dummy", enforce_eager=True,
              gpu_memory_utilization=float(os.environ.get("GMU", "0.10")),
              max_model_len=int(os.environ.get("MAXLEN", "512")), max_num_seqs=8,
              trust_remote_code=False)
    print("Q4 CFG", json.dumps({k: str(v) for k, v in kw.items()}), flush=True)
    llm = LLM(**kw)
    sp = SamplingParams(temperature=0.0, max_tokens=8)
    # NBATCH selects the cooperative_topk cluster size, and that is the whole
    # point of running it twice. csrc/libtorch_stable/cooperative_topk.cu:94
    #     const bool supports_cluster16 = get_device_prop()->major >= 10;
    #     if (num_rows <= 4 && supports_cluster16)  -> cluster of 16 CTAs
    #     else if (num_rows <= 8)                   -> cluster of  8 CTAs
    # num_rows is the batch, and the cluster size is a compile-time template
    # parameter, so a batch above 4 takes the 8-CTA arm of the SAME binary.
    n = int(os.environ.get("NBATCH", "1"))
    prompts = ["The capital of France is"] * n
    print("Q4 NBATCH=%d (<=4 selects the 16-CTA cluster, 5..8 selects the 8-CTA one)" % n, flush=True)
    outs = llm.generate(prompts, sp)
    ids = list(outs[0].outputs[0].token_ids)
    print("Q4 OUTPUT_TOKEN_IDS", ids, flush=True)
    print("Q4 TEXT", repr(outs[0].outputs[0].text), flush=True)
    print("Q4 TOKENS n=%d seqs=%d  (RANDOM WEIGHTS: these tokens carry NO information)"
          % (len(ids), len(outs)), flush=True)
    if not ids:
        raise SystemExit("qwen4_exp produced no tokens")
    print("DONE_MARKER_Q4EXP", flush=True)

# vLLM v1 spawns its EngineCore in a CHILD process, and multiprocessing
# spawn re-executes THIS file as __main__ in that child. Without the guard
# the child re-enters main(), the core never registers, and the parent
# reports `Engine core initialization failed ... Failed core proc(s): {}` --
# an empty dict, which reads exactly like a model or kernel failure and is
# not one. Measured on job B, 20260902T220741Z, on all three backend legs.
if __name__ == "__main__":
    main()
