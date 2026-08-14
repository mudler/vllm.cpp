#!/usr/bin/env python3
# DFlash D3 — CONTEXT-KV PRECOMPUTE + prepare_dflash_inputs + BLOCK-PROPOSAL
# reference capture (spec .agents/specs/dflash-spec-decode.md §6 D3, port map
# DF-DRAFT-KV-PREP). Dumps, from the REAL loaded vLLM in-tree DFlash draft, the
# three fixtures our C++ D3 path is gated against IN ISOLATION (the propose/verify
# e2e is D4):
#   * prepare_ref.json  : the prepare_dflash_inputs outputs (input_ids, query
#                         positions/slots, context positions/slots, sample maps,
#                         seq_lens) for a FIXED synthetic target batch, computed by
#                         a line-for-line numpy replica of _prepare_dflash_inputs_kernel
#                         (dflash/speculator.py:472-618). These are INTEGER, so our
#                         host PrepareDflashInputs must match BIT-exact.
#   * ctxkv_ref.json    : the precomputed context K/V (K normed+RoPE'd, V raw) for a
#                         fixed synthetic context, produced by the REAL loaded draft's
#                         precompute buffers (_build_fused_kv_buffers +
#                         _project_context_kv + _normalize_context_k + the fused RoPE,
#                         qwen3_dflash.py:440-619). Envelope-exact vs our
#                         PrecomputeContextKV (bf16/f32-softmax envelope).
#   * propose_ref.json  : the (1+k) block proposal — per-layer hidden, final hidden,
#                         top-k proposed ids — from the REAL submodules with the
#                         precomputed context K/V prepended to each layer's attention
#                         (the D3 context-aware forward our ForwardBlockLogitsWithContext
#                         computes). STRICT-or-ratified-near-tie on the k proposed ids.
#
# HOW IT RUNS (finalized for the dgx GPU-promotion session, vLLM 0.26.0.dev0 @
# 555967922). MEMORY: the 27B target is MULTIMODAL -> vision-encoder profiling
# OOM-reboots the GB10 (see [[gb10-unified-memory-oom-reboots-box]]); use
# limit_mm_per_prompt={image:0,video:0} + gpu_memory_utilization=0.30 (the D0 fix).
# Run under `flock "${GPU_LOCK:-$HOME/gpu.lock}"`, oracle venv on PATH (FlashInfer JIT needs ninja),
# VLLM_USE_V2_MODEL_RUNNER=1, no FLASH_ATTN pin:
#   flock "${GPU_LOCK:-$HOME/gpu.lock}" env PATH=$HOME/venvs/vllm-oracle/bin:$PATH \
#     VLLM_USE_V2_MODEL_RUNNER=1 ~/venvs/vllm-oracle/bin/python \
#     scripts/spec/d3_dflash_kvprep_ref.py --out-dir tests/parity/goldens/dflash_27b_kvprep
import argparse
import os

TARGET = os.environ.get("DFLASH_TARGET", "unsloth/Qwen3.6-27B-NVFP4")
DRAFT = os.environ.get("DFLASH_DRAFT", "z-lab/Qwen3.6-27B-DFlash")
BLOCK = int(os.environ.get("DFLASH_BLOCK", "16"))
MASK_TOKEN_ID = int(os.environ.get("DFLASH_MASK_TOKEN_ID", "248070"))
ANCHOR_TOKEN_ID = int(os.environ.get("DFLASH_ANCHOR_TOKEN_ID", "9707"))  # "Hello"
CTX_LEN = int(os.environ.get("DFLASH_CTX_LEN", "12"))
BLOCK_SIZE = int(os.environ.get("DFLASH_KV_BLOCK_SIZE", "16"))  # paged KV block size


def prepare_numpy_ref(ctx_len, block, mask_id, anchor_id, kv_block_size):
    """Line-for-line numpy replica of _prepare_dflash_inputs_kernel
    (dflash/speculator.py:508-618) for ONE request, no rejection, first prefill.
    Integer only — the bit-exact oracle for our host PrepareDflashInputs."""
    nqpr = block  # 1 + (block-1)
    nspec = block - 1
    # Single request, contiguous prefill: positions 0..ctx_len-1, block table maps
    # each position to a distinct slot via block_size.
    target_positions = list(range(ctx_len))
    ctx_start, ctx_end = 0, ctx_len
    num_rejected = 0
    valid_ctx_end = ctx_end - num_rejected
    last_valid_pos = target_positions[valid_ctx_end - 1]
    num_blocks = (ctx_len + nqpr + kv_block_size - 1) // kv_block_size
    block_table = list(range(100, 100 + num_blocks))  # arbitrary distinct block ids

    def slot(pos):
        bn = min(pos // kv_block_size, len(block_table) - 1)
        return block_table[bn] * kv_block_size + (pos % kv_block_size)

    context_positions = list(target_positions)
    context_slot_mapping = [slot(p) for p in context_positions]
    input_ids, query_positions, query_slot_mapping = [], [], []
    sample_indices, sample_pos, sample_idx_mapping = [], [], []
    for off in range(nqpr):
        qpos = last_valid_pos + 1 + off
        input_ids.append(anchor_id if off == 0 else mask_id)
        query_positions.append(qpos)
        query_slot_mapping.append(slot(qpos))
        if off >= 1:
            sample_indices.append(off)
            sample_pos.append(qpos)
            sample_idx_mapping.append(0)
    return {
        "ctx_len": ctx_len, "block": block, "kv_block_size": kv_block_size,
        "block_table": block_table,
        "input_ids": input_ids, "query_positions": query_positions,
        "query_slot_mapping": query_slot_mapping,
        "context_positions": context_positions,
        "context_slot_mapping": context_slot_mapping,
        "sample_indices": sample_indices, "sample_pos": sample_pos,
        "sample_idx_mapping": sample_idx_mapping,
        "seq_lens": [last_valid_pos + 1 + nqpr],
    }


def _dump_worker(worker, out_dir, block, mask_id, anchor_id, ctx_len, kv_block_size):
    """Runs INSIDE the vLLM worker (via collective_rpc). Reaches the loaded
    DFlashQwen3ForCausalLM and dumps ctxkv_ref + propose_ref using vLLM's OWN
    weights/modules; writes prepare_ref from the numpy replica."""
    import json
    import os

    import torch

    os.makedirs(out_dir, exist_ok=True)
    runner = worker.model_runner
    drafter = getattr(runner, "speculator", None) or getattr(runner, "drafter", None)
    if drafter is None:
        for v in vars(runner).values():
            m = getattr(v, "model", None)
            if m is not None and m.__class__.__name__.startswith("DFlash"):
                drafter = v
                break
    if drafter is None:
        raise RuntimeError("could not find the DFlash speculator on the model runner")
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
    eps = cfg.rms_norm_eps
    nlayers = cfg.num_hidden_layers
    scale = Dh ** -0.5
    causal = draft.get_draft_attn_causal()
    T = block
    rope_theta = gv("rope_theta")
    if rope_theta is None:
        rope_theta = float(getattr(model.layers[0].self_attn.rotary_emb, "base", 1e7))
    layer_types = list(gv("layer_types", []))
    sliding_window = gv("sliding_window", None)

    # ---- config.json (self-contained fixture for the C++ parity test) ----
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({
            "hidden_size": H, "head_dim": Dh, "num_attention_heads": Hq,
            "num_key_value_heads": Hkv, "num_hidden_layers": nlayers,
            "intermediate_size": cfg.intermediate_size, "vocab_size": cfg.vocab_size,
            "draft_vocab_size": getattr(cfg, "draft_vocab_size", cfg.vocab_size),
            "rms_norm_eps": eps, "rope_theta": rope_theta, "num_taps": len(
                cfg.dflash_config["target_layer_ids"]),
            "mask_token_id": mask_id, "block_size": block, "anchor_token_id": anchor_id,
            "layer_types": layer_types,
            "target_layer_ids": list(cfg.dflash_config["target_layer_ids"]),
            "sliding_window": sliding_window,
            "ctx_len": ctx_len, "kv_block_size": kv_block_size,
        }, f, indent=2)

    # ---- prepare_ref: launch the REAL Triton _prepare_dflash_inputs_kernel on GPU
    # (single-request contiguous prefill, no rejection) and read its outputs back,
    # then cross-check them against the line-for-line numpy replica. The fixture we
    # commit is vLLM's ACTUAL kernel output (not just the replica); the assert proves
    # the replica == the kernel so the C++ host port is gated bit-exact vs both. ----
    import triton
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
        _prepare_dflash_inputs_kernel, PAD_SLOT_ID)

    npref = prepare_numpy_ref(ctx_len, block, mask_id, anchor_id, kv_block_size)
    nqpr = block
    nspec = block - 1
    max_num_reqs = 1
    max_num_tokens = nqpr  # single request, no CG padding beyond the active block
    max_model_len = 2048
    num_blocks_bt = (ctx_len + nqpr + kv_block_size - 1) // kv_block_size
    dl = dev

    t_target_positions = torch.arange(ctx_len, dtype=torch.int64, device=dl)
    t_tqsl = torch.tensor([0, ctx_len], dtype=torch.int32, device=dl)
    t_idx_mapping = torch.tensor([0], dtype=torch.int32, device=dl)
    t_last_sampled = torch.tensor([anchor_id], dtype=torch.int32, device=dl)
    t_next_prefill = torch.tensor([0], dtype=torch.int32, device=dl)
    t_num_sampled = torch.tensor([1], dtype=torch.int32, device=dl)
    t_num_rejected = torch.tensor([0], dtype=torch.int32, device=dl)
    t_block_table = torch.tensor(
        [list(range(100, 100 + num_blocks_bt))], dtype=torch.int32, device=dl)

    o_input_ids = torch.zeros(max_num_tokens, dtype=torch.int32, device=dl)
    o_query_positions = torch.zeros(max_num_tokens, dtype=torch.int64, device=dl)
    o_query_start_loc = torch.zeros(max_num_reqs + 1, dtype=torch.int32, device=dl)
    o_seq_lens = torch.zeros(max_num_reqs, dtype=torch.int32, device=dl)
    o_query_slot = torch.zeros(max_num_tokens, dtype=torch.int64, device=dl)
    o_ctx_positions = torch.zeros(ctx_len, dtype=torch.int64, device=dl)
    o_ctx_slot = torch.zeros(ctx_len, dtype=torch.int64, device=dl)
    o_sample_indices = torch.zeros(max_num_reqs * nspec, dtype=torch.int64, device=dl)
    o_sample_pos = torch.zeros(max_num_reqs * nspec, dtype=torch.int64, device=dl)
    o_sample_idx_map = torch.zeros(max_num_reqs * nspec, dtype=torch.int32, device=dl)

    max_tokens_per_req = ctx_len + nqpr
    BS = min(256, triton.next_power_of_2(max(1, max_tokens_per_req)))
    n_blk = triton.cdiv(max_tokens_per_req, BS)
    _prepare_dflash_inputs_kernel[(1, n_blk)](
        o_input_ids, o_query_positions, o_query_start_loc, o_seq_lens, o_query_slot,
        o_ctx_positions, o_ctx_slot, o_sample_indices, o_sample_pos, o_sample_idx_map,
        t_target_positions, t_tqsl, t_idx_mapping, t_last_sampled, t_next_prefill,
        t_num_sampled, t_num_rejected, t_block_table, t_block_table.stride(0),
        mask_id, kv_block_size, nqpr, nspec, max_num_reqs, max_num_tokens, max_model_len,
        SAMPLE_FROM_ANCHOR=False, PAD_SLOT_ID=PAD_SLOT_ID, BLOCK_SIZE=BS)
    torch.cuda.synchronize()

    kern = {
        "input_ids": o_input_ids.cpu().tolist(),
        "query_positions": o_query_positions.cpu().tolist(),
        "query_start_loc": o_query_start_loc.cpu().tolist(),
        "seq_lens": o_seq_lens.cpu().tolist(),
        "query_slot_mapping": o_query_slot.cpu().tolist(),
        "context_positions": o_ctx_positions.cpu().tolist(),
        "context_slot_mapping": o_ctx_slot.cpu().tolist(),
        "sample_indices": o_sample_indices.cpu().tolist(),
        "sample_pos": o_sample_pos.cpu().tolist(),
        "sample_idx_mapping": o_sample_idx_map.cpu().tolist(),
    }
    # Cross-check: numpy replica == real Triton kernel (active region).
    kernel_matches_numpy = (
        kern["input_ids"] == npref["input_ids"]
        and kern["query_positions"] == npref["query_positions"]
        and kern["query_slot_mapping"] == npref["query_slot_mapping"]
        and kern["context_positions"] == npref["context_positions"]
        and kern["context_slot_mapping"] == npref["context_slot_mapping"]
        and kern["seq_lens"][0] == npref["seq_lens"][0]
        and kern["sample_indices"][:nspec] == npref["sample_indices"]
        and kern["sample_pos"][:nspec] == npref["sample_pos"]
        and kern["sample_idx_mapping"][:nspec] == npref["sample_idx_mapping"])
    assert kernel_matches_numpy, (
        "numpy replica DIVERGES from the real Triton kernel:\n"
        f"kernel={kern}\nnumpy={npref}")
    out_prepare = dict(npref)
    out_prepare["kernel"] = kern
    out_prepare["kernel_matches_numpy"] = kernel_matches_numpy
    out_prepare["max_num_reqs"] = max_num_reqs
    out_prepare["max_num_tokens"] = max_num_tokens
    out_prepare["max_model_len"] = max_model_len
    out_prepare["mask_token_id"] = mask_id
    out_prepare["anchor_token_id"] = anchor_id
    with open(os.path.join(out_dir, "prepare_ref.json"), "w") as f:
        json.dump(out_prepare, f)

    # ---- ctxkv_ref: precompute context K/V from a fixed synthetic context via the
    # REAL loaded precompute buffers (qwen3_dflash.py:440-619). ----
    if not hasattr(model, "_num_attn_layers"):
        model._build_fused_kv_buffers()
    context_states = (0.35 * torch.sin(
        torch.arange(ctx_len * H).float() * 0.21).reshape(ctx_len, H)).to(dev, dt)
    context_positions = torch.arange(ctx_len, device=dev)
    with torch.no_grad():
        all_k, all_v = model._project_context_kv(
            context_states, ctx_len, model._num_attn_layers,
            model._num_kv_heads, model._head_dim)
        all_k_normed = model._normalize_context_k(all_k)
        all_k_flat = all_k_normed.view(model._num_attn_layers * ctx_len, model._kv_size)
        pos_rep = context_positions.repeat(model._num_attn_layers)
        cs = model._rope_cos_sin_cache
        if cs.dtype != all_k_flat.dtype:
            cs = cs.to(dtype=all_k_flat.dtype)
        from vllm import _custom_ops as ops
        ops.rotary_embedding(pos_rep, all_k_flat, None, model._rope_head_size, cs,
                             model._rope_is_neox)
        k_final = all_k_flat.view(model._num_attn_layers, ctx_len, Hkv, Dh).float().cpu()
        v_final = all_v.view(model._num_attn_layers, ctx_len, Hkv, Dh).float().cpu()
    with open(os.path.join(out_dir, "ctxkv_ref.json"), "w") as f:
        json.dump({
            "ctx_len": ctx_len, "num_layers": nlayers, "Hkv": Hkv, "Dh": Dh,
            "context_states": context_states.float().cpu().flatten().tolist(),
            "context_positions": context_positions.cpu().tolist(),
            "k": k_final.flatten().tolist(), "v": v_final.flatten().tolist(),
        }, f)

    # ---- propose_ref: the (1+k) block forward with the precomputed context K/V
    # prepended to each layer's attention (the D3 context-aware attention). ----
    input_ids = [anchor_id] + [mask_id] * (block - 1)
    block_positions = list(range(ctx_len, ctx_len + block))  # anchor at ctx_len
    ids = torch.tensor(input_ids, device=dev)
    bpos = torch.tensor(block_positions, device=dev)

    def ctx_attn(q, k, v, ck, cv, layer_causal):
        # q [B,Hq,Dh]; k/v [B,Hkv,Dh] (block); ck/cv [C,Hkv,Dh] (context).
        rep = Hq // Hkv
        kk = torch.cat([ck, k], dim=0).repeat_interleave(rep, dim=1)  # [C+B,Hq,Dh]
        vv = torch.cat([cv, v], dim=0).repeat_interleave(rep, dim=1)
        qf, kf, vf = q.float(), kk.float(), vv.float()
        scores = torch.einsum("thd,shd->hts", qf, kf) * scale  # [Hq,B,C+B]
        B = q.shape[0]
        C = ck.shape[0]
        neg = float("-inf")
        mask = torch.zeros(B, C + B, device=q.device)
        if layer_causal:  # SWA: causal over [context; block], context all in past
            for i in range(B):
                for j in range(C + B):
                    if j > C + i:
                        mask[i, j] = neg
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
            q, k = sa.rotary_emb(bpos, q, k)
            ck = k_final[li].to(dev, q.dtype)
            cv = v_final[li].to(dev, q.dtype)
            ao = ctx_attn(q.view(-1, Hq, Dh), k.view(-1, Hkv, Dh),
                          v.view(-1, Hkv, Dh), ck, cv, causal[li]).reshape(-1, Hq * Dh)
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
    with open(os.path.join(out_dir, "propose_ref.json"), "w") as f:
        json.dump({
            "input_ids": input_ids, "block_positions": block_positions,
            "ctx_len": ctx_len, "per_layer_causal": list(map(bool, causal)),
            "per_layer_hidden": per_layer,
            "final_hidden": final.float().cpu().flatten().tolist(),
            "draft_vocab": int(logits.shape[1]),
            "topk_ids": tk.indices.tolist(), "topk_vals": tk.values.tolist(),
        }, f)
    return {"ok": True, "nlayers": nlayers, "ctx_len": ctx_len, "block": block}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="tests/parity/goldens/dflash_27b_kvprep")
    ap.add_argument("--target", default=TARGET)
    ap.add_argument("--draft", default=DRAFT)
    ap.add_argument("--gpu-mem-util", type=float, default=0.30)
    ap.add_argument("--num-spec-tokens", type=int, default=BLOCK)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

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
    res = llm.collective_rpc(
        _dump_worker,
        args=(out_dir, BLOCK, MASK_TOKEN_ID, ANCHOR_TOKEN_ID, CTX_LEN, BLOCK_SIZE))
    print(f"[d3] worker dump result = {res}")
    print(f"[d3] wrote reference fixtures to {out_dir}")


if __name__ == "__main__":
    main()
