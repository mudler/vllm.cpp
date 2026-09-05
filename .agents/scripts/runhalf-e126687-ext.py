"""EXT_PRESENT, measured by EXECUTING a compiled kernel rather than by a name.

Job B probed `import vllm._C` and printed EXT_RC=1. That name is what every
prior HEADPIN job probed too, and at THIS revision it does not exist: the
`csrc/libtorch_stable/` refactor ships `vllm._C_stable_libtorch` and
`vllm._moe_C_stable_libtorch`. A missing module is not a missing extension.
"""
import importlib
import torch

for m in ("vllm._C", "vllm._C_stable_libtorch", "vllm._moe_C_stable_libtorch",
          "vllm.cumem_allocator", "vllm.fs_io_C", "vllm.spinloop",
          "vllm.vllm_flash_attn._vllm_fa2_C", "vllm.vllm_flash_attn._vllm_fa3_C"):
    try:
        mod = importlib.import_module(m)
        print("EXTMOD %-40s PRESENT %s" % (m, getattr(mod, "__file__", "builtin")))
    except Exception as e:
        print("EXTMOD %-40s ABSENT  %s" % (m, type(e).__name__))

# A name is not an execution. Run a compiled custom op on the device.
import vllm._custom_ops as ops
x = torch.randn(4, 128, dtype=torch.bfloat16, device="cuda")
w = torch.ones(128, dtype=torch.bfloat16, device="cuda")
out = torch.empty_like(x)
ops.rms_norm(out, x, w, 1e-6)
torch.cuda.synchronize()
ref = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
err = (out.float() - ref).abs().max().item()
print("EXTOP rms_norm EXECUTED on", torch.cuda.get_device_name(0),
      "max_abs_err_vs_torch=%.4g" % err)
assert torch.isfinite(out).all(), "compiled op produced non-finite output"
assert err < 0.05, "compiled op disagrees with a torch reference: %g" % err
print("EXT_PRESENT=True")
