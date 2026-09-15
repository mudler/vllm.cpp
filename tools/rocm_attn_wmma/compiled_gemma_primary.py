"""Export compiled Gemma expressions and RoPE samples into one fixture bundle."""

import ast, hashlib, importlib.util, json
from pathlib import Path
import torch
from torch._inductor.async_compile import AsyncCompile
import argparse
import vllm
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.model_executor.layers.rotary_embedding.linear_scaling_rope import (
    LinearScalingRotaryEmbedding,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--layer-module", type=Path, required=True)
parser.add_argument("--initial-module", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if "e126687a9" not in vllm.__version__:
    raise RuntimeError(f"wrong primary pin: {vllm.__version__}")
dst = args.output
dst.mkdir(exist_ok=False)
sources = {"layer": args.layer_module, "initial": args.initial_module}
expected_sources = {
    "layer": "e4ef8a4e67080a024bf231d82f6eeedeac71a3c8785b055833f4f3a3f0e337f6",
    "initial": "5bf4b6b5af1feaab7f9c1b8cf57cd098847c5af518006990479c12c6028ca900",
}
for name, path in sources.items():
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sources[name]:
        raise RuntimeError(f"generated primary module differs: {name}")
mods = {}
manifest = {}
for name, path in sources.items():
    text = path.read_text()
    tree = ast.parse(text)
    defs = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "triton"
        ):
            defs[node.targets[0].id] = ast.get_source_segment(text, node)
    target = dst / (name + ".py")
    target.write_text(
        "from torch._inductor.async_compile import AsyncCompile\nasync_compile=AsyncCompile()\n"
        + "\n".join(defs.values())
        + "\nasync_compile.wait(globals())\n"
    )
    spec = importlib.util.spec_from_file_location("compiled_" + name, target)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mods[name] = mod
    manifest[name] = {
        "source": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    torch.cuda.synchronize()
torch.manual_seed(91015)
payload = bytearray()


def rnd(shape, scale=1.0):
    return (torch.randn(shape, device="cuda") * scale).bfloat16()


def store(t):
    data = t.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    offset = len(payload)
    payload.extend(data)
    return {
        "offset": offset,
        "bytes": len(data),
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


stream = torch.cuda.current_stream().cuda_stream
report = {
    "pin": "e126687a9a828d513c01a07cd69f025f27d63280",
    "sources": manifest,
    "cases": [],
    "origin": "Frozen executing torch.compile kernels, ROCm production Gemma3 at the "
    "recorded vLLM pin. Harness extraction preserves generated expressions and stores.",
}
for rows in (1, 3):
    m = mods["layer"]
    a = rnd((rows, 2560))
    base = rnd((rows, 2560), 3)
    delta = rnd((rows, 2560))
    wa = rnd((2560,), 0.5)
    wd = rnd((2560,), 0.5)
    w = rnd((2560,), 0.5)
    sums = torch.empty((rows, 1), device="cuda", dtype=torch.float32)
    out2 = torch.empty_like(a)
    out3 = torch.empty_like(a)
    res = torch.empty_like(a)
    scratch = torch.empty_like(a, dtype=torch.float32)
    m.triton_red_fused__to_copy_add_fused_add_rms_norm_rms_norm_0.run(
        a, wa, base, w, sums, out2, rows, 2560, stream=stream
    )
    m.triton_red_fused__to_copy_add_fused_add_rms_norm_rms_norm_2.run(
        delta, wd, a, sums, wa, base, w, scratch, res, out3, rows, 2560, stream=stream
    )
    c = {"name": f"sandwich{rows}", "kind": "sandwich", "rows": rows, "width": 2560}
    for name, t in dict(
        a=a, base=base, delta=delta, wa=wa, wd=wd, w=w, out2=out2, out3=out3, res=res
    ).items():
        c[name] = store(t)
    report["cases"].append(c)
    qkv = rnd((rows, 4096))
    qw = rnd((256,), 0.5)
    kw = rnd((256,), 0.5)
    cs = rnd((64, 256))
    pos = torch.tensor(list(range(17, 17 + rows)), device="cuda", dtype=torch.int64)
    qs = torch.empty((rows, 8, 1), device="cuda", dtype=torch.float32)
    ks = torch.empty((rows, 4, 1), device="cuda", dtype=torch.float32)
    qo = torch.empty((rows, 8, 256), device="cuda", dtype=torch.bfloat16)
    ko = torch.empty((rows, 4, 256), device="cuda", dtype=torch.bfloat16)
    m.triton_red_fused_3.run(qkv, qs, ks, rows * 8, rows * 4, stream=stream)
    m.triton_poi_fused_4.run(
        qkv,
        ks,
        kw,
        pos,
        cs,
        qs,
        qw,
        ko,
        ko[:, :, 128:],
        qo,
        qo[:, :, 128:],
        rows * 4 * 128,
        rows * 8 * 128,
        stream=stream,
    )
    c = {"name": f"qk{rows}", "kind": "qk", "rows": rows}
    for name, t in dict(qkv=qkv, qw=qw, kw=kw, cs=cs, pos=pos, qo=qo, ko=ko).items():
        c[name] = store(t)
    report["cases"].append(c)
    if rows == 1:
        x = rnd((rows, 20480), 2)
        out = torch.empty((rows, 10240), device="cuda", dtype=torch.bfloat16)
        m.triton_poi_fused_gelu_mul_slice_1.run(x, out, rows * 10240, stream=stream)
        report["cases"].append(
            {
                "name": "gelu",
                "kind": "gelu",
                "x": store(x),
                "out": store(out),
            }
        )
for rows in (1, 3):
    table = rnd((16, 2560), 0.03)
    ids = torch.arange(rows, device="cuda", dtype=torch.int32)
    scale = torch.tensor([50.5], device="cuda", dtype=torch.bfloat16)
    w = rnd((2560,), 0.5)
    res = torch.empty((rows, 2560), device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(res)
    mods["initial"].triton_red_fused__to_copy_add_embedding_mul_rms_norm_0.run(
        ids, table, scale, w, res, out, rows, 2560, stream=stream
    )
    c = {
        "name": f"scaled{rows}",
        "kind": "scaled",
        "scale": 50.5,
        "rows": rows,
        "width": 2560,
    }
    for name, t in dict(x=table[:rows], w=w, res=res, out=out).items():
        c[name] = store(t)
    report["cases"].append(c)
report["rope_cache_source"] = (
    "vllm/model_executor/layers/rotary_embedding/base.py:89-112 and "
    "linear_scaling_rope.py:107-127; executing Gemma3 layer buffers on gfx1100"
)
report["rope_cache"] = []
torch.set_default_device("cuda")
with set_current_vllm_config(VllmConfig()):
    for name, base, factor, positions in [
        ("local", 10000.0, 1.0, [0, 42, 53, 64, 86, 106, 518, 1207, 4095, 131071]),
        ("global", 1000000.0, 8.0, [0, 82, 107, 212, 518, 1207, 4095, 131071, 1048575]),
    ]:
        if factor == 1.0:
            rope = RotaryEmbedding(256, 256, 131072, base, True, torch.bfloat16)
        else:
            rope = LinearScalingRotaryEmbedding(
                256, 256, 131072, base, True, factor, torch.bfloat16
            )
        sample = rope.cos_sin_cache[positions].contiguous()
        report["rope_cache"].append(
            dict(
                name=name,
                base=base,
                factor=factor,
                positions=positions,
                **store(sample),
            )
        )
report["data_file"] = "cases.bin"
report["data_bytes"] = len(payload)
(dst / report["data_file"]).write_bytes(payload)
(dst / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
print("EXPORTED", len(report["cases"]), flush=True)
