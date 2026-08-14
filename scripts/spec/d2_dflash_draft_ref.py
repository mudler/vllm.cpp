#!/usr/bin/env python3
# DFlash D2 — STANDALONE DRAFT-FORWARD reference capture (spec
# .agents/specs/dflash-spec-decode.md §6 D2, port map DF-DRAFT-MODEL). Dumps the
# vLLM in-tree DFlash draft's CONTEXT-FREE block forward so our C++
# Qwen3DFlashModel::ForwardBlockLogits / CombineAuxFeatures can be gated against
# it in ISOLATION (the D2 gate is the draft forward, NOT the propose/verify e2e,
# which is D4).
#
# HOW THE REFERENCE IS BUILT (finalized on-box, vLLM 0.26.0.dev0 @ 555967922):
# we construct the real target+DFlash draft exactly as the D0 capture (mm-off,
# gpu_util 0.30 — the GB10 OOM-reboot fix) and reach the REAL loaded draft module
# `DFlashQwen3ForCausalLM`. Everything below uses vLLM's OWN loaded weights and
# modules:
#   * fc_ref     : vLLM's REAL draft.combine_hidden_states(aux) over a fixed
#                  synthetic aux block (the fc + the eagle3 tap-concat order).
#   * block_ref  : the context-free (1+k) mask-block forward. We call the draft's
#                  REAL submodules per layer (input_layernorm, qkv_proj, q_norm,
#                  k_norm, rotary_emb, o_proj, post_attention_layernorm, mlp,
#                  model.norm) and REAL lm_head, substituting ONLY the paged
#                  attention core `self.attn(q,k,v)` — which reads the pre-inserted
#                  context KV cache (a D3 concern) — with the documented in-block
#                  attention (full layers BIDIRECTIONAL, SWA layers causal-in-
#                  window; qwen3_dflash.py:86-146,238-263). This is exactly the
#                  isolation vt::DFlashBlockAttention computes, so per-layer hidden
#                  + logits are a faithful vLLM-weights reference for our port. The
#                  RED proofs (causal-flip / reversed-tap) are the deterministic
#                  C++ unit test test_qwen3_dflash_forward.cpp.
#   * weights_ref: sha256 + shape of every draft tensor vLLM actually loaded
#                  (embed_tokens, lm_head, model.norm, model.fc, per-layer), so the
#                  C++ loader is confirmed to read byte-identical weights (incl. the
#                  target-SHARED embed_tokens/lm_head the draft ckpt omits).
#
# MEMORY (GB10 119 GiB UNIFIED -> OOM-reboots the box; see
# [[gb10-unified-memory-oom-reboots-box]]): the 27B target is MULTIMODAL, so vLLM
# profiles the vision encoder at max image size on init. limit_mm_per_prompt
# ={image:0,video:0} + gpu_memory_utilization=0.30 (the D0 fix, this tool's
# default). Run under `flock "${GPU_LOCK:-$HOME/gpu.lock}"`, oracle venv on PATH (FlashInfer JIT
# needs ninja), VLLM_USE_V2_MODEL_RUNNER=1, no FLASH_ATTN pin:
#   flock "${GPU_LOCK:-$HOME/gpu.lock}" env PATH=$HOME/venvs/vllm-oracle/bin:$PATH \
#     VLLM_USE_V2_MODEL_RUNNER=1 ~/venvs/vllm-oracle/bin/python \
#     scripts/spec/d2_dflash_draft_ref.py --out-dir tests/parity/goldens/dflash_27b_draft
import argparse
import hashlib
import json
import math
import os

TARGET = os.environ.get("DFLASH_TARGET", "unsloth/Qwen3.6-27B-NVFP4")
DRAFT = os.environ.get("DFLASH_DRAFT", "z-lab/Qwen3.6-27B-DFlash")
BLOCK = int(os.environ.get("DFLASH_BLOCK", "16"))
MASK_TOKEN_ID = int(os.environ.get("DFLASH_MASK_TOKEN_ID", "248070"))
ANCHOR_TOKEN_ID = int(os.environ.get("DFLASH_ANCHOR_TOKEN_ID", "9707"))  # "Hello"


def build_block_inputs():
    """The uniform (1+k) mask block: anchor token then k mask tokens; positions
    0..k; single request so cu_seqlens = [0, 1+k]. Mirrors prepare_dflash_inputs'
    query block (speculator.py:416-563) at ctx_end==0 (empty context)."""
    k = BLOCK - 1
    input_ids = [ANCHOR_TOKEN_ID] + [MASK_TOKEN_ID] * k
    positions = list(range(BLOCK))
    cu_seqlens = [0, BLOCK]
    return input_ids, positions, cu_seqlens


def _dump_worker(worker, out_dir, block, mask_id, anchor_id):
    """Runs INSIDE the vLLM worker process (via collective_rpc) where the model
    lives — the V1 EngineCore runs the model in a subprocess, so this is the only
    way to reach the loaded draft. Navigates worker.model_runner.drafter.model
    (DFlashProposer.model = DFlashQwen3ForCausalLM) and writes all fixtures."""
    import hashlib
    import json
    import os

    import torch

    def sha(t):
        return hashlib.sha256(
            t.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()

    os.makedirs(out_dir, exist_ok=True)
    runner = worker.model_runner
    # V2 model runner stores it as `speculator`; V1 as `drafter`. Both expose
    # `.model` = DFlashQwen3ForCausalLM. Fall back to a __dict__ scan.
    drafter = getattr(runner, "speculator", None) or getattr(runner, "drafter", None)
    if drafter is None:
        for v in vars(runner).values():
            m = getattr(v, "model", None)
            if m is not None and m.__class__.__name__.startswith("DFlash"):
                drafter = v
                break
    if drafter is None:
        raise RuntimeError("could not find the DFlash speculator on the model runner: "
                           + ",".join(sorted(vars(runner).keys())))
    draft = drafter.model  # DFlashQwen3ForCausalLM
    model = draft.model
    dev = next(draft.parameters()).device
    dt = next(draft.parameters()).dtype

    cfg = draft.config
    inner = getattr(cfg, "model", cfg)  # EAGLEConfig wraps the real config in .model

    def gv(name, default=None):
        for c in (cfg, inner):
            if c is not None and hasattr(c, name):
                return getattr(c, name)
        return default

    H = cfg.hidden_size
    Dh = cfg.head_dim
    Hq = cfg.num_attention_heads
    Hkv = cfg.num_key_value_heads
    taps = len(cfg.dflash_config["target_layer_ids"])
    eps = cfg.rms_norm_eps
    nlayers = cfg.num_hidden_layers
    scale = Dh ** -0.5
    # rope_theta / layer_types / sliding_window are NOT proxied by EAGLEConfig; read
    # from the inner config, falling back to the real rotary module's base.
    rope_theta = gv("rope_theta")
    if rope_theta is None:
        rope_theta = float(getattr(model.layers[0].self_attn.rotary_emb, "base", 1e7))
    layer_types = list(gv("layer_types", []))
    sliding_window = gv("sliding_window", None)
    win = sliding_window or 0
    T = block

    # ---- weights_ref: sha256 + shape of every tensor vLLM loaded into the draft.
    wref = {name: {"shape": list(p.shape), "dtype": str(p.dtype), "sha256": sha(p)}
            for name, p in draft.named_parameters()}
    with open(os.path.join(out_dir, "weights_ref.json"), "w") as f:
        json.dump(wref, f, indent=1)
    with open(os.path.join(out_dir, "ckpt_keys.txt"), "w") as f:
        f.write("\n".join(sorted(wref.keys())))
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({
            "hidden_size": H, "head_dim": Dh, "num_attention_heads": Hq,
            "num_key_value_heads": Hkv, "num_hidden_layers": nlayers,
            "intermediate_size": cfg.intermediate_size, "vocab_size": cfg.vocab_size,
            "draft_vocab_size": getattr(cfg, "draft_vocab_size", cfg.vocab_size),
            "rms_norm_eps": eps, "rope_theta": rope_theta, "num_taps": taps,
            "mask_token_id": mask_id, "block_size": block, "anchor_token_id": anchor_id,
            "layer_types": layer_types,
            "target_layer_ids": list(cfg.dflash_config["target_layer_ids"]),
            "sliding_window": sliding_window,
        }, f, indent=2)

    # ---- fc_ref: vLLM's REAL combine_hidden_states over a fixed synthetic aux.
    aux = 0.2 * torch.sin(torch.arange(T * H * taps).float() * 0.3).reshape(T, H * taps)
    with torch.no_grad():
        fc_out = draft.combine_hidden_states(aux.to(device=dev, dtype=dt)).float().cpu()
    with open(os.path.join(out_dir, "fc_ref.json"), "w") as f:
        json.dump({"comb": fc_out.flatten().tolist(), "T": T, "hidden": H, "taps": taps}, f)

    # ---- block_ref: context-free (1+k) block forward via the REAL submodules,
    # substituting ONLY the paged-attn core (a D3 context-KV concern) with the
    # documented in-block attention.
    input_ids = [anchor_id] + [mask_id] * (block - 1)
    positions = list(range(block))
    cu = [0, block]
    causal = draft.get_draft_attn_causal()
    ids = torch.tensor(input_ids, device=dev)
    pos = torch.tensor(positions, device=dev)

    def inblock_attn(q, k, v, layer_causal):
        rep = Hq // Hkv
        kk = k.repeat_interleave(rep, dim=1)
        vv = v.repeat_interleave(rep, dim=1)
        qf, kf, vf = q.float(), kk.float(), vv.float()
        scores = torch.einsum("thd,shd->hts", qf, kf) * scale
        Tq = q.shape[0]
        mask = torch.zeros(Tq, Tq, device=q.device)
        if layer_causal:
            neg = torch.full((Tq, Tq), float("-inf"), device=q.device)
            m = torch.triu(neg, diagonal=1)
            if win and win > 0:
                lo = torch.arange(Tq, device=q.device).unsqueeze(1) - (win - 1)
                jj = torch.arange(Tq, device=q.device).unsqueeze(0)
                m = m + torch.where(jj < lo, neg, torch.zeros_like(neg))
            mask = m
        probs = torch.softmax(scores + mask.unsqueeze(0), dim=-1)
        return torch.einsum("hts,shd->thd", probs, vf).to(q.dtype)

    per_layer = []
    with torch.no_grad():
        hidden = model.embed_input_ids(ids)
        residual = None
        for li, layer in enumerate(model.layers):
            sa = layer.self_attn
            if residual is None:
                residual = hidden
                hn = layer.input_layernorm(hidden)
            else:
                hn, residual = layer.input_layernorm(hidden, residual)
            qkv, _ = sa.qkv_proj(hn)
            q, k, v = qkv.split([sa.q_size, sa.kv_size, sa.kv_size], dim=-1)
            q = sa.q_norm(q.view(-1, Hq, Dh)).view(-1, sa.q_size)
            k = sa.k_norm(k.view(-1, Hkv, Dh)).view(-1, sa.kv_size)
            q, k = sa.rotary_emb(pos, q, k)
            ao = inblock_attn(q.view(-1, Hq, Dh), k.view(-1, Hkv, Dh),
                              v.view(-1, Hkv, Dh), causal[li]).reshape(-1, Hq * Dh)
            attn_out, _ = sa.o_proj(ao)
            hn2, residual = layer.post_attention_layernorm(attn_out, residual)
            hidden = layer.mlp(hn2)
            per_layer.append(hidden.float().cpu().flatten().tolist())
        final, _ = model.norm(hidden, residual)
        logits = draft.logits_processor(draft.lm_head, final)
        if logits is None:
            logits = torch.matmul(final.float(), draft.lm_head.weight.float().t())
        logits = logits.float().cpu()
        tk = torch.topk(logits, k=8, dim=-1)
        final_c = final.float().cpu()

    with open(os.path.join(out_dir, "block_ref.json"), "w") as f:
        json.dump({
            "input_ids": input_ids, "positions": positions, "cu_seqlens": cu,
            "per_layer_causal": list(map(bool, causal)),
            "per_layer_hidden": per_layer,
            "final_hidden": final_c.flatten().tolist(),
            "draft_vocab": int(logits.shape[1]),
            "topk_ids": tk.indices.tolist(),
            "topk_vals": tk.values.tolist(),
        }, f)
    return {"ok": True, "causal": list(map(bool, causal)), "nlayers": nlayers, "T": T}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="tests/parity/goldens/dflash_27b_draft")
    ap.add_argument("--target", default=TARGET)
    ap.add_argument("--draft", default=DRAFT)
    ap.add_argument("--gpu-mem-util", type=float, default=0.30)
    ap.add_argument("--num-spec-tokens", type=int, default=BLOCK)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    import torch
    from vllm import LLM

    llm = LLM(
        model=args.target,
        speculative_config={
            "method": "dflash",
            "model": args.draft,
            "num_speculative_tokens": args.num_spec_tokens,
        },
        gpu_memory_utilization=args.gpu_mem_util,
        limit_mm_per_prompt={"image": 0, "video": 0},
        enforce_eager=True,
        max_model_len=2048,
    )
    out_dir = os.path.abspath(args.out_dir)
    # The V1 EngineCore runs the model in a subprocess; reach the loaded draft via
    # collective_rpc into the worker (where model_runner.drafter.model lives).
    res = llm.collective_rpc(
        _dump_worker, args=(out_dir, BLOCK, MASK_TOKEN_ID, ANCHOR_TOKEN_ID))
    print(f"[d2] worker dump result = {res}")
    print(f"[d2] wrote reference fixtures to {out_dir}")


if __name__ == "__main__":
    main()
