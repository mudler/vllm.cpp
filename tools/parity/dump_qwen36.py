# vllm.cpp parity harness; oracle = pip vLLM 0.24.0 (same release family as pin e24d1b24).
"""Dump Qwen3.6 per-layer + full-model goldens from the pinned vLLM model with
REAL weights (M0.9 Task 1 — the M0 exit-criterion oracle).

Unlike dump_gdn.py / dump_moe.py (which run isolated pinned OP modules with
stubbed infra), this script runs the WHOLE model end to end through the vLLM
v1 offline engine so the hard plumbing — GDN mamba state, full-attention KV
cache, attention-metadata build, mrope positions, NVFP4 dequant/GEMM — is done
by vLLM itself. We attach forward hooks to capture per-layer hidden states and
compute logits from the model's own bf16 lm_head. See
.agents/qwen36-forward-notes.md for the full recipe + rationale.

Run on dgx (GB10, 119 GiB unified) in the pip oracle venv (`~/venvs/vllm-oracle`,
pip vLLM 0.24.0 — same release family as pin e24d1b24). The pinned-Python-over-
pip-kernels overlay was TRIED and BLOCKED (the post-0.24.0-tag pinned Python
expects a newer vendored/compiled API than the pip 0.24.0 wheel ships — see
.agents/qwen36-forward-notes.md §3), so pip 0.24.0 is the oracle. Every
forward-math module in these checkpoints is byte-identical to the pin; only
weight-loader plumbing + ROCm code differ. The ACTUAL working command (§4):

    ssh dgx.casa 'cd ~/work/vllm.cpp && VLLM_ENABLE_V1_MULTIPROCESSING=0 \
        PATH=~/venvs/vllm-oracle/bin:$PATH ~/venvs/vllm-oracle/bin/python \
        tools/parity/dump_qwen36.py --model <snapshot_dir> --tag <27b|35b> \
        --out tests/parity/goldens'

VLLM_ENABLE_V1_MULTIPROCESSING=0 forces the engine core in-process so the
forward hooks (and the global CAPT dict they write) live in this process.

Captured, for a fixed short raw-token prompt (no chat template):
  qwen36_embed_<tag>        embed_tokens output [T, H]
  qwen36_gdn_layer_<tag>    first linear_attention decoder layer:
                            residual-stream in [T,H] -> out [T,H] (+ positions)
  qwen36_fullattn_layer_<tag> first full_attention decoder layer (qk-norm +
                            partial mrope + GQA + sigmoid output gate):
                            residual-stream in [T,H] -> out [T,H] (+ positions)
  qwen36_norm_<tag>         final RMSNorm: stream in [T,H] -> normed out [T,H]
  qwen36_logits_<tag>       full-model reference: per-position argmax [T],
                            top-1000 logit values+indices [T,1000], and the
                            greedy (temperature 0) decode continuation ids.

Residual-stream note: vLLM threads (hidden_states, residual) through the fused
add-RMSNorm layers. The true residual stream entering layer i is
`hidden_states` (layer 0, residual None) or `hidden_states + residual`
(i>0); leaving layer i it is `out_hidden + out_residual`. We reconstruct and
store those [T,H] streams so a C++ layer can be tested standalone:
stream_in -> layer -> stream_out. Sums are done in f32 (the fused kernel
accumulates the residual in f32 before rounding; storing f32 keeps the exact
pre-round value).

Budget <= 2 MB: hidden states are [T~8, 5120] f32 (tiny); logits are NOT dumped
full ([T, 248320] f32 would be ~8 MB) — we keep per-position argmax + top-1000
values/indices (the plan's budget option), enough for greedy parity + a logit
gap on the dominant logits.
"""

import argparse
import json
import pathlib

import numpy as np
import torch

UPSTREAM_PIN = "e24d1b24"
TOPK = 1000
N_GREEDY = 16

# Populated by forward hooks (in-process; VLLM_ENABLE_V1_MULTIPROCESSING=0).
CAPT: dict = {}
PROMPT_LEN = None  # set before generate; guards hooks to the prefill pass only

DTYPE_NAMES = {torch.float32: "f32", torch.bfloat16: "bf16",
               torch.float16: "f16", torch.int32: "i32", torch.int64: "i64"}


# ------------------------------------------------------------- hook helpers

def _first_dim(t):
    return t.shape[0] if t.ndim >= 1 else None


def _is_prefill(t):
    # Capture only the single prefill forward (token dim == prompt length),
    # never the T==1 decode steps that follow in the same generate() call.
    return isinstance(t, torch.Tensor) and _first_dim(t) == PROMPT_LEN


def _stream(hidden, residual):
    if residual is None:
        return hidden.detach().float().cpu()
    return (hidden.detach().float() + residual.detach().float()).cpu()


def register_hooks(model, gdn_idx, full_idx):
    lm = model.language_model            # Qwen3_5(Moe)ForCausalLM
    mdl = lm.model                       # Qwen3_5Model
    layers = mdl.layers

    def embed_hook(_m, _inp, out):
        if _is_prefill(out) and "embed" not in CAPT:
            CAPT["embed"] = out.detach().float().cpu()
    mdl.embed_tokens.register_forward_hook(embed_hook)

    def mk_pre(name):
        def pre(_m, args, kwargs):
            hs = kwargs["hidden_states"]
            if _is_prefill(hs) and name + "_in" not in CAPT:
                CAPT[name + "_in"] = _stream(hs, kwargs.get("residual"))
                pos = kwargs.get("positions")
                if pos is not None:
                    CAPT[name + "_positions"] = pos.detach().to(torch.int64).cpu()
        return pre

    def mk_post(name):
        def post(_m, args, kwargs, output):
            hs, res = output
            if _is_prefill(hs) and name + "_out" not in CAPT:
                CAPT[name + "_out"] = (hs.detach().float() + res.detach().float()).cpu()
        return post

    for name, idx in (("gdn", gdn_idx), ("full", full_idx)):
        L = layers[idx]
        L.register_forward_pre_hook(mk_pre(name), with_kwargs=True)
        L.register_forward_hook(mk_post(name), with_kwargs=True)

    # Final RMSNorm is called positionally: norm(hidden_states, residual).
    def norm_pre(_m, args):
        hs = args[0]
        res = args[1] if len(args) > 1 else None
        if _is_prefill(hs) and "norm_in" not in CAPT:
            CAPT["norm_in"] = _stream(hs, res)
    mdl.norm.register_forward_pre_hook(norm_pre)

    def norm_post(_m, _inp, output):
        normed = output[0] if isinstance(output, tuple) else output
        if _is_prefill(normed) and "norm_out" not in CAPT:
            CAPT["norm_out"] = normed.detach().float().cpu()
    mdl.norm.register_forward_hook(norm_post)


# ------------------------------------------------------------------ saving

def save_case(root, name, op, tensors, args, inputs, outputs, tol, meta):
    case = root / name
    case.mkdir(parents=True, exist_ok=True)
    manifest = {
        "op": op, "args": args, "tensors": {}, "inputs": inputs,
        "outputs": outputs, "tol": tol, "oracle": meta,
    }
    total = 0
    for key, t in tensors.items():
        if t.dtype in (torch.bfloat16, torch.float16):
            arr = t.contiguous().view(torch.uint16).cpu().numpy()
        else:
            arr = t.contiguous().cpu().numpy()
        np.save(case / f"{key}.npy", arr)
        total += arr.nbytes
        manifest["tensors"][key] = {"file": f"{key}.npy",
                                    "dtype": DTYPE_NAMES[t.dtype],
                                    "shape": list(t.shape)}
    (case / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"wrote {case}  ({total / 1024:.1f} KiB)")
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True, help="e.g. 27b / 35b")
    ap.add_argument("--prompt",
                    default="The capital of France is Paris, and the")
    ap.add_argument("--gpu-mem", type=float, default=0.5)
    args = ap.parse_args()

    import os
    assert os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") == "0", \
        "set VLLM_ENABLE_V1_MULTIPROCESSING=0 so hooks run in-process"

    import vllm
    print("vllm from:", vllm.__file__, "version", vllm.__version__)
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    ids = tok(args.prompt, add_special_tokens=True)["input_ids"]
    global PROMPT_LEN
    PROMPT_LEN = len(ids)
    print("PROMPT_IDS", ids, "len", PROMPT_LEN)

    llm = LLM(model=args.model, enforce_eager=True, tensor_parallel_size=1,
              max_model_len=256, gpu_memory_utilization=args.gpu_mem,
              max_num_seqs=1, dtype="bfloat16")

    # config + layer dispatch
    hf = llm.llm_engine.vllm_config.model_config.hf_text_config
    layer_types = list(hf.layer_types)
    gdn_idx = layer_types.index("linear_attention")
    full_idx = layer_types.index("full_attention")
    print("model_type", hf.model_type, "layers", hf.num_hidden_layers,
          "gdn_idx", gdn_idx, "full_idx", full_idx)

    # oracle provenance. NOTE (M0.9 Task 1): the pinned checkout e24d1b24 is
    # post-0.24.0-tag and its Python expects newer vendored/compiled APIs
    # (vllm_flash_attn.compile_flash_attn_varlen_func_from_specs) than the pip
    # 0.24.0 wheel ships, so a pinned-Python-over-pip-kernels overlay does not
    # run (blocker recorded in .agents/qwen36-forward-notes.md). We therefore
    # use the pip vLLM 0.24.0 engine — the SAME release family as the pin — as
    # the end-to-end oracle; it loads these exact checkpoints and greedy-decodes
    # coherently ("The capital of France is" -> " Paris"). Deviation from the
    # "pinned-only" rule is documented in the notes.
    src = "pinned" if "pinenv" in vllm.__file__ else "pip-vllm"
    meta = {
        "source": f"{src}:{vllm.__version__}",
        "pin_reference": UPSTREAM_PIN,
        "detail": ("vLLM v1 offline engine (pip vllm "
                   f"{vllm.__version__}, same release family as pin "
                   f"{UPSTREAM_PIN}). Quantized linears run the real NVFP4 "
                   "path (cutlass FP4 for compressed-tensors W4A4 [27B] / "
                   "modelopt W4A16 [35B]); GDN in_proj, embed, norms, lm_head "
                   "are bf16."),
        "torch": torch.__version__.split("+")[0],
        "device": torch.cuda.get_device_name(0),
        "model": args.model,
        "prompt": args.prompt,
        "prompt_ids": ids,
        "layer_types_head": layer_types[:8],
        "gdn_layer_idx": gdn_idx,
        "full_attn_layer_idx": full_idx,
        "note": ("Residual-stream in/out reconstructed as hidden+residual "
                 "(f32). Full-attn layer exercises qk-norm + partial mrope "
                 "RoPE + GQA + sigmoid output gate. Logits computed via the "
                 "model's own compute_logits (LogitsProcessor -> lm_head) on "
                 "the captured post-norm hidden; only argmax + top-1000 kept "
                 "(budget)."),
    }

    # register hooks in-process, then run greedy generate (prefill captured;
    # decode steps skipped by the prompt-length guard).
    llm.apply_model(lambda m: register_hooks(m, gdn_idx, full_idx))
    sp = SamplingParams(temperature=0.0, max_tokens=N_GREEDY)
    out = llm.generate({"prompt_token_ids": ids}, sp)
    greedy_ids = list(out[0].outputs[0].token_ids)
    print("GREEDY_TOKEN_IDS", greedy_ids)
    print("GREEDY_TEXT", repr(out[0].outputs[0].text))

    for k in ("embed", "gdn_in", "gdn_out", "full_in", "full_out",
              "norm_in", "norm_out"):
        assert k in CAPT, f"missing capture {k}; CAPT={list(CAPT)}"
        assert CAPT[k].shape[0] == PROMPT_LEN, (k, CAPT[k].shape)

    # logits via the model's OWN compute_logits (LogitsProcessor -> lm_head),
    # on the captured post-norm hidden states. Works for the bf16 lm_head (27B)
    # and the quantized lm_head (35B) alike — reproduces the real logit path.
    def compute_logits(m):
        h = CAPT["norm_out"].to("cuda", torch.bfloat16)
        lg = m.language_model.compute_logits(h)
        return lg.float().cpu()  # [T, vocab]
    logits = llm.apply_model(compute_logits)[0]
    argmax = logits.argmax(-1).to(torch.int32)
    topv, topi = torch.topk(logits, TOPK, dim=-1)
    print("PREFILL_ARGMAX", argmax.tolist())
    assert int(argmax[-1].item()) == greedy_ids[0], \
        (int(argmax[-1]), greedy_ids[0], "last-pos argmax must equal 1st greedy token")

    tokt = torch.tensor(ids, dtype=torch.int32)
    root = pathlib.Path(args.out)
    tag = args.tag
    total = 0
    total += save_case(
        root, f"qwen36_embed_{tag}", "qwen36_embed",
        {"token_ids": tokt, "embed": CAPT["embed"]},
        {"hidden": CAPT["embed"].shape[1], "seqlen": PROMPT_LEN},
        ["token_ids"], ["embed"], {"atol": 1e-3, "rtol": 1e-3}, meta)
    for name, tag_op in (("gdn", "qwen36_gdn_layer"),
                         ("full", "qwen36_fullattn_layer")):
        t = {"hidden_in": CAPT[name + "_in"], "out": CAPT[name + "_out"]}
        if name + "_positions" in CAPT:
            t["positions"] = CAPT[name + "_positions"]
        total += save_case(
            root, f"{tag_op}_{tag}", tag_op, t,
            {"layer_idx": gdn_idx if name == "gdn" else full_idx,
             "layer_type": "linear_attention" if name == "gdn"
             else "full_attention", "seqlen": PROMPT_LEN,
             "positions_layout": "mrope [3,T] (text: rows equal); "
             "partial_rotary_factor 0.25"},
            ["hidden_in", "positions"], ["out"],
            {"atol": 5e-2, "rtol": 5e-2}, meta)
    total += save_case(
        root, f"qwen36_norm_{tag}", "qwen36_norm",
        {"hidden_in": CAPT["norm_in"], "out": CAPT["norm_out"]},
        {"seqlen": PROMPT_LEN, "eps": float(hf.rms_norm_eps)},
        ["hidden_in"], ["out"], {"atol": 1e-3, "rtol": 1e-3}, meta)
    total += save_case(
        root, f"qwen36_logits_{tag}", "qwen36_logits",
        {"token_ids": tokt, "argmax": argmax,
         "topk_values": topv.contiguous(), "topk_indices": topi.to(torch.int32).contiguous(),
         "greedy_ids": torch.tensor(greedy_ids, dtype=torch.int32)},
        {"vocab_size": int(hf.vocab_size), "topk": TOPK,
         "n_greedy": N_GREEDY, "seqlen": PROMPT_LEN,
         "note": "argmax/top-k over per-position logits = lm_head(norm_out); "
                 "greedy_ids = temperature-0 decode continuation."},
        ["token_ids"], ["argmax", "greedy_ids"],
        {"argmax": "exact", "logits_atol": 5e-2, "logits_rtol": 5e-2}, meta)

    print(f"TOTAL golden bytes: {total / 1024:.1f} KiB "
          f"({total / (1024 * 1024):.2f} MiB)")
    assert total <= 2 * 1024 * 1024, "over the 2MB golden budget"


if __name__ == "__main__":
    main()
