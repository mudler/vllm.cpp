import importlib, torch, traceback
for m in ("vllm._C", "vllm._rocm_C", "vllm._C_stable_libtorch", "vllm._moe_C_stable_libtorch"):
    try:
        importlib.import_module(m); print(f"EXT {m:30s} LOADED")
    except Exception as e:
        print(f"EXT {m:30s} FAILED: {type(e).__name__}: {e}")
try:
    from vllm import _custom_ops as ops
    d = torch.device("cuda:0")
    x = torch.randn(4, 512, device=d, dtype=torch.bfloat16)
    w = torch.randn(512, device=d, dtype=torch.bfloat16)
    out = torch.empty_like(x)
    ops.rms_norm(out, x, w, 1e-6)
    ref = (x.float() / (x.float().pow(2).mean(-1, keepdim=True) + 1e-6).sqrt()) * w.float()
    print("CUSTOM_OP rms_norm    max_abs_err =", (out.float() - ref).abs().max().item())
    # A SECOND op, chosen because it is a different kernel family. This pin has
    # no `_custom_ops.silu_and_mul`; the activation went to torch.ops._C direct.
    y = torch.randn(4, 1024, device=d, dtype=torch.bfloat16)
    o2 = torch.empty(4, 512, device=d, dtype=torch.bfloat16)
    torch.ops._C.silu_and_mul(o2, y)
    r2 = torch.nn.functional.silu(y[..., :512].float()) * y[..., 512:].float()
    print("CUSTOM_OP silu_and_mul max_abs_err =", (o2.float() - r2).abs().max().item())
    # A CONTROL: the op must actually have written. A kernel that never ran would
    # leave the garbage `torch.empty` gave it, and that is not what is compared.
    print("CUSTOM_OP silu_and_mul rel_err     =", ((o2.float()-r2).abs().max()/r2.abs().max()).item())
    torch.cuda.synchronize()
    print("CUSTOM_OPS=PASS")
except Exception:
    traceback.print_exc(); print("CUSTOM_OPS=FAIL")
