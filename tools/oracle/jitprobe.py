#!/usr/bin/env python3
"""#1864 Triton JIT probe: make the compiler's own message reachable.

Attempt 5 died 20 s into the render inside triton's runtime, which JIT-builds a
CPython extension (`cuda_utils`) and calls

    subprocess.check_call(cc_cmd, stdout=subprocess.DEVNULL)
    # triton/runtime/build.py:48, in the triton the WORKER installed (3.7.1);
    # the line is quoted above rather than only cited, because that path is an
    # installed package and not a pinned tree, so nothing here can resolve it.

with no `stderr=` argument at all. `check_call` raises CalledProcessError
carrying only the argv, so gcc's diagnostic went to the process's stderr in a
context nothing was reading, and five attempts produced a return code without a
reason. This wraps that one call, prints the argv, the translation unit and BOTH
streams, and only then re-raises. It changes no compiler flag and no computation.

Exercised afterwards, in the order the render meets them:
  1. the driver init that failed (`driver.active`, which builds `cuda_utils`),
  2. the exact dispatch the render hit -- a [B,D,1] @ [B,1,S] matmul, which
     torch._native routes to `bmm_outer_product`'s triton impl, from
     `modeling_gemma4_unified.py` in the transformers the WORKER installed --
     a version Phase A never recorded, so the line number that dispatch sat on
     is not resolvable against any pin and is deliberately not written here,
  3. an ordinary triton kernel, so a failure that is ptxas rather than gcc is
     distinguishable from one that is not.

Exit 0 only when all three pass.
"""
import os
import subprocess
import sys
import traceback

_REAL_CHECK_CALL = subprocess.check_call


def _loud_check_call(cmd, *args, **kwargs):
    print("=== JITPROBE cc_cmd ===", flush=True)
    print(" ".join(map(str, cmd)), flush=True)
    src = next((t for t in cmd if isinstance(t, str) and t.endswith(".c")), None)
    if src and os.path.exists(src):
        print("=== JITPROBE translation unit %s (first 30 lines) ===" % src, flush=True)
        with open(src) as fh:
            for i, line in enumerate(fh):
                if i >= 30:
                    break
                print("%4d| %s" % (i + 1, line.rstrip()), flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("=== JITPROBE cc rc=%d ===" % res.returncode, flush=True)
    if res.stdout:
        print("--- cc stdout ---\n" + res.stdout, flush=True)
    if res.stderr:
        print("--- cc stderr ---\n" + res.stderr, flush=True)
    if res.returncode:
        raise subprocess.CalledProcessError(res.returncode, cmd)
    return 0


subprocess.check_call = _loud_check_call

rc = 0
import torch  # noqa: E402
import triton  # noqa: E402

print("JITPROBE torch", torch.__version__, "triton", getattr(triton, "__version__", "?"),
      "cuda_avail", torch.cuda.is_available(), flush=True)
print("JITPROBE TRITON_CACHE_DIR", os.environ.get("TRITON_CACHE_DIR", "(unset)"), flush=True)

try:
    from triton.runtime import driver
    print("JITPROBE driver OK, current device", driver.active.get_current_device(), flush=True)
except Exception:
    traceback.print_exc()
    rc = 1

try:
    a = torch.randn(1, 128, 1, device="cuda", dtype=torch.float32)
    b = torch.randn(1, 1, 77, device="cuda", dtype=torch.float32)
    c = (a @ b).transpose(1, 2)
    torch.cuda.synchronize()
    print("JITPROBE bmm_outer_product dispatch OK", tuple(c.shape),
          float(c.abs().sum()), flush=True)
except Exception:
    traceback.print_exc()
    rc = 1

try:
    import triton.language as tl

    @triton.jit
    def _double(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = off < n
        tl.store(y_ptr + off, tl.load(x_ptr + off, mask=m) * 2.0, mask=m)

    x = torch.randn(1024, device="cuda")
    y = torch.empty_like(x)
    _double[(1,)](x, y, 1024, BLOCK=1024)
    torch.cuda.synchronize()
    assert torch.allclose(y, x * 2), "triton kernel produced the wrong values"
    print("JITPROBE triton kernel compile+launch OK", flush=True)
except Exception:
    traceback.print_exc()
    rc = 1

# Read-only: record what escape hatches torch._native offers, so the question
# "can the render avoid the JIT" is answered from the source rather than guessed.
# Nothing here is set. An eager fallback would change which kernel computes the
# reference, and this run exists to be a faithful reference.
try:
    import torch._native as _n
    root = os.path.dirname(_n.__file__)
    print("=== JITPROBE torch._native env switches (read-only) ===", flush=True)
    out = subprocess.run(
        ["grep", "-rhoE", r"(TORCH|TRITON)[A-Z0-9_]*", root],
        capture_output=True, text=True)
    names = sorted(set(out.stdout.split()))
    print("  " + " ".join(names[:60]), flush=True)
except Exception:
    traceback.print_exc()

print("JITPROBE rc=%d" % rc, flush=True)
sys.exit(rc)
