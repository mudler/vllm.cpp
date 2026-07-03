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
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm
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


DUMPERS = {"rmsnorm": dump_rmsnorm}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    root = pathlib.Path(a.out)
    for name, fn in DUMPERS.items():
        if a.only is None or name in a.only:
            fn(root)
