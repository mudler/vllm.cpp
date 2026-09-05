#!/usr/bin/env python3
"""Does Qwen3.5's layernorm reach vLLM's _C kernels at all?

The question decides the `no kernel image` attribution. If it does, then
vLLM's `_C` has no sm_110 image for `rms_norm_kernel` either and the forward
could never have reached the MLP; if it does not, then `silu_and_mul` is the
FIRST `_C` launch of the whole forward.

This does not transcribe the predicate. It reads the two constructs out of the
pinned vLLM checkout by regex and `exec`s those exact source bytes, so a change
upstream makes the probe fail to find them rather than silently answer for a
version that no longer exists. Run it with --vllm <pinned checkout>.

Positive control: the same predicate with a dtype-matched weight, which is what
a plain `RMSNorm` passes, must return True.
"""
import argparse, re, sys
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--vllm", required=True, help="pinned vllm checkout (5559679229)")
a = ap.parse_args()

kern = open(f"{a.vllm}/vllm/kernels/vllm_c.py").read()
lnorm = open(f"{a.vllm}/vllm/model_executor/layers/layernorm.py").read()
qnext = open(f"{a.vllm}/vllm/model_executor/models/qwen3_next.py").read()

# 1. Qwen3.5's norm IS GemmaRMSNorm, under an alias.
alias = re.search(r"^\s*GemmaRMSNorm as Qwen3NextRMSNorm,$", qnext, re.M)
assert alias, "qwen3_next no longer aliases GemmaRMSNorm"
print("qwen3_next.py:", alias.group(0).strip())

# 2. GemmaRMSNorm widens the weight to f32 before handing it to the IR op.
widen = re.search(r"^        weight = self\.weight\.float\(\) \+ 1\.0$", lnorm, re.M)
assert widen, "GemmaRMSNorm no longer widens its weight"
print("layernorm.py:", widen.group(0).strip())

# 3. The vllm_c impls are admissible only for a dtype-MATCHED weight.
ns = {}
for name in ("rms_no_var_size", "rms_add_no_var_size"):
    m = re.search(r"^%s = lambda .*?\n\)\n" % name, kern, re.S | re.M)
    assert m, f"vllm_c.py no longer defines {name}"
    exec(m.group(0), ns)
print("vllm_c.py: exec'd rms_no_var_size, rms_add_no_var_size")

class _P: pass
p = _P()
p.weight = torch.zeros(8, dtype=torch.bfloat16)   # nn.Parameter under a bf16 model dtype
loc = {}
exec(widen.group(0).strip(), {"self": p, "torch": torch}, loc)
w = loc["weight"]
x = torch.zeros(4, 8, dtype=torch.bfloat16)
res = torch.zeros(4, 8, dtype=torch.bfloat16)

print(f"\nx.dtype={x.dtype}  weight.dtype={w.dtype}")
r1 = ns["rms_no_var_size"](x, w, 1e-6, None)
r2 = ns["rms_add_no_var_size"](x, res, w, 1e-6, None)
ctl = ns["rms_add_no_var_size"](x, res, w.to(torch.bfloat16), 1e-6, None)
print(f"vllm_c rms_norm            supports_args = {r1}")
print(f"vllm_c fused_add_rms_norm  supports_args = {r2}")
print(f"CONTROL, dtype-matched weight            = {ctl}")
ok = (r1 is False) and (r2 is False) and (ctl is True)
print("\nVERDICT:", "vllm_c is INADMISSIBLE for every Qwen3.5 norm; ir.op.dispatch "
      "falls through to native, so no _C layernorm kernel is ever launched"
      if ok else "UNEXPECTED — re-read the pin")
sys.exit(0 if ok else 1)
