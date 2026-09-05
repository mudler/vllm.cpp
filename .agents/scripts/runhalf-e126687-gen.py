#!/usr/bin/env python3
"""RUNHALF (#2611): make vLLM at e126687a9a GENERATE TOKENS, greedily.

The prompt battery, dtype, sampling parameters and per-prompt batch=1 regime are
copied from scripts/opt-oracle-capture.py so the token ids this prints can be
laid beside the committed golden in tests/parity/goldens/opt_greedy. That
comparison is INFORMATIVE, not a gate: the golden was captured at the PIN on
dgx (sm_121a) and this runs a different revision on different silicon.

Every value that decides what was measured is printed. A leg that cannot run
raises; it never prints a partial success.
"""
import json
import os
import sys

PROMPTS = [
    "The capital of France is",
    "Once upon a time,",
    "The largest planet in our solar system is",
    "The chemical symbol for gold is",
    "In 1969, humans first walked on",
    "Water boils at a temperature of",
]

def main():
    model = os.environ["MODEL"]
    eager = os.environ.get("EAGER", "1") == "1"
    backend = os.environ.get("BACKEND", "default")
    gmu = float(os.environ.get("GMU", "0.10"))
    maxlen = int(os.environ.get("MAXLEN", "2048"))
    T = int(os.environ.get("MAXTOK", "16"))

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    print("VLLM_FILE", vllm.__file__, flush=True)
    print("VLLM_VERSION", vllm.__version__, flush=True)
    print("TORCH_VERSION", torch.__version__, flush=True)
    print("CUDA_AVAIL", torch.cuda.is_available(), flush=True)
    print("DEVICE", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), flush=True)

    kw = dict(
        model=model,
        dtype="bfloat16",
        # NO load_format. The default "auto" resolves to "hf", whose
        # allow_patterns are ["*.safetensors", "*.bin"], which is what
        # facebook/opt-125m ships. An earlier leg passed load_format="pt"
        # on the strength of the field's docstring ("pt will load the weights
        # in the pytorch bin format"); default_loader.py globs "*.pt" for that
        # value and refused the checkpoint with `Cannot find any model
        # weights`. Passing nothing is both correct and the plainer path, and
        # it is what scripts/opt-oracle-capture.py passes.
        enforce_eager=eager,
        # thor is UNIFIED memory: gpu_memory_utilization reserves HOST RAM, and
        # this box reboots rather than OOM-kills (.agents/environment.md). 0.10
        # of 132 GB is ~13 GB, which is far more than a 125M model and its KV
        # pool need, and far below the fraction that has taken a box down.
        gpu_memory_utilization=gmu,
        max_model_len=maxlen,
        max_num_seqs=1,
    )
    if backend != "default":
        kw["attention_backend"] = backend
    print("CFG", json.dumps({k: str(v) for k, v in kw.items()}), flush=True)

    llm = LLM(**kw)
    sp = SamplingParams(temperature=0.0, max_tokens=T)
    print("BACKEND_REQUESTED", backend, flush=True)

    all_ids = []
    for i, p in enumerate(PROMPTS):
        out = llm.generate([p], sp)[0]
        ids = list(out.outputs[0].token_ids)
        all_ids.append(ids)
        print(f"PROMPT[{i}] {p!r}", flush=True)
        print(f"OUTPUT_TOKEN_IDS[{i}] {ids}", flush=True)
        print(f"TEXT[{i}] {out.outputs[0].text!r}", flush=True)
        print(f"TOKENS[{i}] n={len(ids)}", flush=True)

    total = sum(len(x) for x in all_ids)
    print("TOKENS total=%d prompts=%d" % (total, len(PROMPTS)), flush=True)
    if total == 0:
        raise SystemExit("no tokens were generated; this is not a run")
    outdir = os.environ.get("OUTDIR")
    if outdir:
        with open(os.path.join(outdir, "greedy_ids.%s.json" % os.environ.get("VLLM_LEG", "leg")), "w") as f:
            json.dump({"prompts": PROMPTS, "token_ids": all_ids,
                       "vllm_version": vllm.__version__, "cfg": {k: str(v) for k, v in kw.items()}}, f)
    print("DONE_MARKER_GEN", flush=True)

# vLLM v1 spawns its EngineCore in a CHILD process, and multiprocessing
# spawn re-executes THIS file as __main__ in that child. Without the guard
# the child re-enters main(), the core never registers, and the parent
# reports `Engine core initialization failed ... Failed core proc(s): {}` --
# an empty dict, which reads exactly like a model or kernel failure and is
# not one. Measured on job B, 20260902T220741Z, on all three backend legs.
if __name__ == "__main__":
    main()
