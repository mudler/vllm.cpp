# vllm.cpp parity harness; oracle = upstream vLLM (see .agents/upstream-sync.md pin).
"""Dump op-level goldens using upstream vLLM forward_native implementations.

Run inside the oracle venv:  python tools/parity/dump_ops.py --out tests/parity/goldens
"""
import argparse
import pathlib

import torch

from dump_common import save_case

TIGHT = {"atol": 1e-5, "rtol": 1e-5}
BF16 = {"atol": 8e-3, "rtol": 8e-3}


def dump_rmsnorm(root):
    # vLLM >= 0.24: CustomOp layers must be instantiated inside a vLLM config
    # context (same pattern as upstream's own op unit tests). Math is still
    # upstream forward_native — this only satisfies the dispatch machinery.
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm
    with set_current_vllm_config(VllmConfig()):
        torch.manual_seed(0)
        for dtype, tolname, tol in ((torch.float32, "f32", TIGHT), (torch.bfloat16, "bf16", BF16)):
            x = torch.randn(8, 128, dtype=dtype)
            layer = RMSNorm(128, eps=1e-6).to(dtype)
            layer.weight.data.normal_()
            out = layer.forward_native(x.clone())
            save_case(root, f"rmsnorm_{tolname}_8x128", "rmsnorm",
                      {"x": x, "weight": layer.weight.data, "out": out.float()},
                      {"eps": 1e-6, "gemma": False, "fused_residual": False},
                      ["x", "weight"], ["out"], tol)
        # gemma variant (bf16 — the gate models' dtype)
        xg = torch.randn(8, 128, dtype=torch.bfloat16)
        gl = GemmaRMSNorm(128, eps=1e-6).to(torch.bfloat16)
        gl.weight.data.normal_()
        outg = gl.forward_native(xg.clone())
        save_case(root, "rmsnorm_gemma_bf16_8x128", "rmsnorm",
                  {"x": xg, "weight": gl.weight.data, "out": outg.float()},
                  {"eps": 1e-6, "gemma": True, "fused_residual": False},
                  ["x", "weight"], ["out"], BF16)
        # fused residual (f32)
        xr = torch.randn(8, 128)
        res = torch.randn(8, 128)
        lr = RMSNorm(128, eps=1e-6)
        lr.weight.data.normal_()
        out2, res2 = lr.forward_native(xr.clone(), residual=res.clone())
        save_case(root, "rmsnorm_fused_residual_f32_8x128", "rmsnorm",
                  {"x": xr, "weight": lr.weight.data, "residual_in": res,
                   "out": out2.float(), "residual_out": res2.float()},
                  {"eps": 1e-6, "gemma": False, "fused_residual": True},
                  ["x", "weight", "residual_in"], ["out", "residual_out"], TIGHT)


def dump_matmul(root):
    # Oracle is pure torch (no vLLM layer code): f32-accumulation reference
    # over bf16-rounded inputs, matching our kernel's accumulation exactly.
    torch.manual_seed(0)
    a = torch.randn(16, 64, dtype=torch.bfloat16)
    b = torch.randn(64, 32, dtype=torch.bfloat16)
    out = a.float() @ b.float()
    save_case(root, "matmul_bf16_16x64x32", "matmul",
              {"a": a, "b": b, "out": out}, {}, ["a", "b"], ["out"], TIGHT)


def dump_silu_and_mul(root):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.activation import SiluAndMul
    with set_current_vllm_config(VllmConfig()):
        torch.manual_seed(0)
        x = torch.randn(8, 256, dtype=torch.bfloat16)
        out = SiluAndMul().forward_native(x)
        save_case(root, "silu_and_mul_bf16_8x256", "silu_and_mul",
                  {"x": x, "out": out.float()}, {}, ["x"], ["out"], BF16)


def dump_embedding(root):
    # Oracle is pure torch F.embedding: exact row gather.
    torch.manual_seed(0)
    table = torch.randn(64, 32, dtype=torch.bfloat16)
    ids = torch.randint(0, 64, (32,), dtype=torch.int64)
    ids[0], ids[1], ids[2], ids[3] = 0, 63, 7, 7  # boundary ids + guaranteed dup
    out = torch.nn.functional.embedding(ids, table)
    save_case(root, "embedding_bf16_32ids", "embedding",
              {"table": table, "ids": ids, "out": out.float()},
              {}, ["table", "ids"], ["out"], TIGHT)


def dump_rope(root):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.rotary_embedding import get_rope
    with set_current_vllm_config(VllmConfig()):
        torch.manual_seed(0)
        rope = get_rope(head_size=64, max_position=200000, is_neox_style=True,
                        rope_parameters={"rope_type": "default", "rope_theta": 10000,
                                         "rope_dim": 32}, dtype=torch.float32)
        args = {"base": float(rope.base), "rotary_dim": 32, "head_size": 64,
                "num_q_heads": 4, "num_kv_heads": 2,
                "get_rope_call": ("get_rope(head_size=64, max_position=200000, "
                                  "is_neox_style=True, rope_parameters={'rope_type': "
                                  "'default', 'rope_theta': 10000, 'rope_dim': 32}, "
                                  "dtype=torch.float32)")}
        for name, pos, tol in (
                ("rope_f32_pos_short", torch.arange(0, 64, 8, dtype=torch.int64), TIGHT),
                ("rope_f32_pos_131k", torch.arange(131040, 131104, 8, dtype=torch.int64),
                 {"atol": 2e-2, "rtol": 0.0})):
            q = torch.randn(8, 4 * 64, dtype=torch.float32)  # [T, Hq*D] flattened
            k = torch.randn(8, 2 * 64, dtype=torch.float32)  # [T, Hk*D] flattened
            q_out, k_out = rope.forward_native(pos, q.clone(), k.clone())
            save_case(root, name, "rope",
                      {"q_in": q, "k_in": k, "positions": pos,
                       "q_out": q_out, "k_out": k_out},
                      args, ["q_in", "k_in", "positions"], ["q_out", "k_out"], tol)


def _attention_case(root, name, T, Hq, Hk, D, rotary_dim, base, eps, tol):
    """Dense causal attention op golden (qwen36-forward-notes.md §5).

    Oracle: vLLM GemmaRMSNorm.forward_native (qk-norm, gemma (1+w), f32) +
    get_rope(...).forward_native (partial NeoX, same API as dump_rope) +
    pure-torch causal GQA scaled-dot-product (f32) + torch.sigmoid gate. Every
    piece is pinned-equivalent to Qwen3NextAttention's core. Tensors: q/gate/k/v
    are the PRE-norm projection split (q = pre-norm query, gate = pre-sigmoid
    output gate, both per-head [T,Hq,D]); `attn` is the pre-gate attention output
    [T,Hq,D]; `out` is the gated output flattened [T,Hq*D].
    """
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm
    from vllm.model_executor.layers.rotary_embedding import get_rope
    with set_current_vllm_config(VllmConfig()):
        torch.manual_seed(0)
        q = torch.randn(T, Hq, D, dtype=torch.float32)
        gate = torch.randn(T, Hq, D, dtype=torch.float32)
        k = torch.randn(T, Hk, D, dtype=torch.float32)
        v = torch.randn(T, Hk, D, dtype=torch.float32)
        positions = torch.arange(T, dtype=torch.int64)

        q_norm = GemmaRMSNorm(D, eps=eps).to(torch.float32)
        k_norm = GemmaRMSNorm(D, eps=eps).to(torch.float32)
        q_norm.weight.data.normal_()
        k_norm.weight.data.normal_()
        qn = q_norm.forward_native(q.clone())  # per-head RMSNorm over D
        kn = k_norm.forward_native(k.clone())

        rope = get_rope(head_size=D, max_position=200000, is_neox_style=True,
                        rope_parameters={"rope_type": "default", "rope_theta": base,
                                         "rope_dim": rotary_dim}, dtype=torch.float32)
        qf, kf = rope.forward_native(positions, qn.reshape(T, Hq * D).clone(),
                                     kn.reshape(T, Hk * D).clone())
        qr = qf.reshape(T, Hq, D)
        kr = kf.reshape(T, Hk, D)

        scale = D ** -0.5
        rep = Hq // Hk
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        attn = torch.empty(T, Hq, D, dtype=torch.float32)
        for h in range(Hq):
            g = h // rep
            scores = (qr[:, h, :] @ kr[:, g, :].T) * scale  # [T,T]
            scores = scores.masked_fill(mask, float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            attn[:, h, :] = probs @ v[:, g, :]  # v is NOT normed/roped
        gated = (attn * torch.sigmoid(gate)).reshape(T, Hq * D)

        args = {"eps": eps, "base": float(rope.base), "rotary_dim": rotary_dim,
                "scale": scale, "head_dim": D, "num_q_heads": Hq, "num_kv_heads": Hk,
                "causal": True}
        save_case(root, name, "dense_attention",
                  {"q": q, "gate": gate, "k": k, "v": v, "positions": positions,
                   "q_norm_weight": q_norm.weight.data, "k_norm_weight": k_norm.weight.data,
                   "attn": attn, "out": gated},
                  args,
                  ["q", "gate", "k", "v", "positions", "q_norm_weight", "k_norm_weight"],
                  ["attn", "out"], tol)


def dump_attention(root):
    TOL = {"atol": 2e-4, "rtol": 2e-4}
    # small hand-scale case (GQA ratio 2), and real head_dim/rotary few-heads.
    _attention_case(root, "dense_attention_f32_small", T=6, Hq=4, Hk=2, D=8,
                    rotary_dim=4, base=10000.0, eps=1e-6, tol=TOL)
    _attention_case(root, "dense_attention_f32_realdims", T=9, Hq=4, Hk=2, D=256,
                    rotary_dim=64, base=1e7, eps=1e-6, tol=TOL)


DUMPERS = {"rmsnorm": dump_rmsnorm, "matmul": dump_matmul,
           "silu_and_mul": dump_silu_and_mul, "embedding": dump_embedding,
           "rope": dump_rope, "attention": dump_attention}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    root = pathlib.Path(a.out)
    for name, fn in DUMPERS.items():
        if a.only is None or name in a.only:
            fn(root)
