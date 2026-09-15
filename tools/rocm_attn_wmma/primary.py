#!/usr/bin/env python3
"""Export and compare identical arrays through pinned vLLM and native attention."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def bf16(array):
    return torch.from_numpy(array.astype(np.float32)).to(torch.bfloat16).view(torch.uint16).numpy()


def read_bf16(path, shape):
    return torch.from_numpy(np.fromfile(path, dtype=np.uint16).reshape(shape)).view(torch.bfloat16)


def generate(output):
    output.mkdir(exist_ok=False)
    cases = []
    rng = np.random.default_rng(0)
    # Prefix-prefill e126687a9a test_contexted_kv_attention uses uniform
    # [-1e-3,1e-3], seed zero, windows 0/16/2048, and atol=1e-4, rtol=0.
    # BF16 and single-request tails extend its F16 multi-request fixtures to
    # the reachable SharedK specialization. Stress inputs retain the frozen
    # P1 oracle's independent abs=1.5e-2, rel=1e-2 bar.
    for tokens, prefix, kv_heads, window, amplitude, softcap, causal in [
        (64, 0, 1, -1, .001, 0., True),
        (79, 19, 4, 15, .001, 0., True),
        (257, 19, 4, 2047, .001, 0., True),
        (64, 0, 1, -1, 2., 0., True),
        (79, 19, 4, 15, 2., 0., True),
        (257, 19, 4, 1023, 2., 0., True),
        (79, 19, 1, 15, 2., 30., True),
        (79, 19, 1, -1, 2., 0., False),
        (1024, 0, 4, 1023, .001, 0., True),
        (2048, 0, 4, -1, .001, 0., True),
    ]:
        dim, block, heads = 256, 16, 2 * kv_heads
        seq_len = tokens + prefix
        blocks = (seq_len + block - 1) // block
        name = f"case{len(cases):02d}-t{tokens}-p{prefix}-h{heads}-w{window}"
        directory = output / name
        directory.mkdir()
        arrays = {
            "q.bf16": bf16(rng.uniform(-amplitude, amplitude, (tokens, heads, dim))),
            "k.bf16": bf16(rng.uniform(-amplitude, amplitude, (blocks, block, kv_heads, dim))),
            "v.bf16": bf16(rng.uniform(-amplitude, amplitude, (blocks, block, kv_heads, dim))),
            "blocks.i32": rng.permutation(blocks).astype(np.int32),
            "length.i32": np.array([seq_len], dtype=np.int32),
            "starts.i32": np.array([0, tokens], dtype=np.int32),
            "poison.bf16": bf16(np.full((tokens, heads, dim), 64.)),
        }
        hashes = {}
        for filename, array in arrays.items():
            array.tofile(directory / filename)
            hashes[filename] = hashlib.sha256(array.tobytes()).hexdigest()
        cases.append(dict(name=name, tokens=tokens, heads=heads, kv_heads=kv_heads,
                          dim=dim, blocks=blocks, block=block, seq_len=seq_len,
                          scale=1. / np.sqrt(dim), causal=causal, softcap=softcap,
                          window_left=window, window_right=0 if window >= 0 else -1,
                          amplitude=amplitude, hashes=hashes))
    (output / "manifest.json").write_text(json.dumps({"cases": cases}, indent=2) + "\n")


def primary(manifest_path, output):
    import vllm
    from vllm.v1.attention.ops.prefix_prefill import context_attention_fwd
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention
    if "e126687a9" not in vllm.__version__:
        raise RuntimeError(f"wrong primary pin: {vllm.__version__}")
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    output.mkdir(exist_ok=False)
    cases = json.loads(manifest_path.read_text())["cases"]
    report = {"vllm_version": vllm.__version__, "cases": []}
    for c in cases:
        directory = manifest_path.parent / c["name"]
        for filename, digest in c["hashes"].items():
            if hashlib.sha256((directory / filename).read_bytes()).hexdigest() != digest:
                raise ValueError(f"fixture checksum differs: {filename}")
        t, h, hk, d, nb, b = (c[k] for k in ("tokens", "heads", "kv_heads", "dim", "blocks", "block"))
        q = read_bf16(directory / "q.bf16", (t, h, d)).cuda()
        k = read_bf16(directory / "k.bf16", (nb, b, hk, d)).cuda()
        v = read_bf16(directory / "v.bf16", (nb, b, hk, d)).cuda()
        blocks = torch.from_numpy(np.fromfile(directory / "blocks.i32", dtype=np.int32)).reshape(1, nb).cuda()
        lens = torch.tensor([c["seq_len"]], dtype=torch.int32, device="cuda")
        starts = torch.tensor([0, t], dtype=torch.int32, device="cuda")
        out = torch.full_like(q, 64.)
        if c["softcap"]:
            # Unified attention is the primary API that exposes softcap.
            unified_attention(q=q, k=k, v=v, out=out, cu_seqlens_q=starts,
                              max_seqlen_q=t, seqused_k=lens, max_seqlen_k=c["seq_len"],
                              softmax_scale=c["scale"], causal=c["causal"],
                              window_size=(c["window_left"], c["window_right"]),
                              block_table=blocks, softcap=c["softcap"],
                              q_descale=None, k_descale=None, v_descale=None)
            kernel = "unified_attention"
        else:
            kp = k.reshape(nb, b, hk, d // 8, 8).permute(0, 2, 3, 1, 4).contiguous()
            vp = v.permute(0, 2, 3, 1).contiguous()
            one = torch.ones((), dtype=torch.float32, device="cuda")
            context_attention_fwd(q, None, None, out, "auto", kp, vp, blocks, starts,
                                  lens, c["seq_len"], t, one, one,
                                  sliding_window=c["window_left"] + 1 if c["window_left"] >= 0 else 0,
                                  sm_scale=c["scale"], causal=c["causal"])
            kernel = "context_attention_fwd"
        torch.cuda.synchronize()
        out.cpu().view(torch.uint16).numpy().tofile(output / (c["name"] + ".bf16"))
        report["cases"].append({"name": c["name"], "kernel": kernel})
        print("PRIMARY", c["name"], kernel, flush=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


def compare(manifest_path, reference, native, output):
    rows = []
    for c in json.loads(manifest_path.read_text())["cases"]:
        shape = (c["tokens"], c["heads"], c["dim"])
        ref = read_bf16(reference / (c["name"] + ".bf16"), shape).float().numpy()
        got = read_bf16(native / (c["name"] + ".bf16"), shape).float().numpy()
        difference = np.abs(got - ref)
        atol, rtol = (1e-4, 0.) if c["amplitude"] == .001 else (1.5e-2, 1e-2)
        valid = bool(np.isfinite(got).all() and np.isfinite(ref).all())
        violations = int(np.count_nonzero(difference > atol + rtol * np.abs(ref)))
        rows.append(dict(name=c["name"], max_abs=float(difference.max()),
                         differing_words=int(np.count_nonzero(got != ref)),
                         finite=valid, violations=violations, atol=atol, rtol=rtol,
                         passed=valid and violations == 0))
    report = {"reference": str(reference), "native": str(native), "cases": rows,
              "passed": all(r["passed"] for r in rows)}
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("generate")
    p.add_argument("output", type=Path)
    p = sub.add_parser("primary")
    p.add_argument("manifest", type=Path)
    p.add_argument("output", type=Path)
    p = sub.add_parser("compare")
    for name in ("manifest", "reference", "native", "output"):
        p.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.mode == "generate":
        generate(args.output)
    elif args.mode == "primary":
        primary(args.manifest, args.output)
    else:
        compare(args.manifest, args.reference, args.native, args.output)


if __name__ == "__main__":
    main()
