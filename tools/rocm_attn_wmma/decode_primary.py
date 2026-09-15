#!/usr/bin/env python3
"""Gate single-query RDNA3 attention against the executing pinned vLLM kernel."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def generate(output):
    from primary import bf16

    output.mkdir(exist_ok=False)
    rng = np.random.default_rng(917)
    cases = []
    for block in (16, 32, 64, 19):
        for length, window, heads in (
            (1, -1, 1),
            (17, 0, 4),
            (79, 15, 1),
            (1217, 1022, 4),
            (1217, -1, 4),
        ):
            count = (length + block - 1) // block
            name = f"decode-b{block}-s{length}-w{window}-kv{heads}"
            directory = output / name
            directory.mkdir()
            order = rng.permutation(count).astype(np.int32)
            k = bf16(rng.normal(0, 1, (count, block, heads, 256)))
            v = bf16(rng.normal(0, 1, (count, block, heads, 256)))
            # The unused tail deliberately carries BF16 NaNs. A masked score
            # alone does not protect the PV product from 0 * NaN.
            for position in range(length, count * block):
                k[order[position // block], position % block] = 0x7FC0
                v[order[position // block], position % block] = 0x7FC0
            arrays = {
                "q.bf16": bf16(rng.normal(0, 1, (1, heads * 2, 256))),
                "k.bf16": k,
                "v.bf16": v,
                "blocks.i32": order,
                "length.i32": np.array([length], np.int32),
                "starts.i32": np.array([0, 1], np.int32),
                "poison.bf16": bf16(np.full((1, heads * 2, 256), 64.0)),
            }
            hashes = {}
            for filename, array in arrays.items():
                array.tofile(directory / filename)
                hashes[filename] = hashlib.sha256(array.tobytes()).hexdigest()
            cases.append(
                dict(
                    name=name,
                    tokens=1,
                    heads=heads * 2,
                    kv_heads=heads,
                    dim=256,
                    blocks=count,
                    block=block,
                    seq_len=length,
                    scale=0.0625,
                    causal=True,
                    softcap=0.0,
                    window_left=window,
                    window_right=0 if window >= 0 else -1,
                    amplitude=1.0,
                    hashes=hashes,
                )
            )
    (output / "manifest.json").write_text(json.dumps({"cases": cases}, indent=2) + "\n")


def capture(manifest, output):
    import torch
    from primary import read_bf16

    import vllm
    from vllm.v1.attention.ops.chunked_prefill_paged_decode import (
        kernel_paged_attention_2d,
    )

    if "e126687a9" not in vllm.__version__:
        raise RuntimeError("wrong vLLM revision")
    output.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    for c in json.loads(manifest.read_text())["cases"]:
        directory = manifest.parent / c["name"]
        for filename, digest in c["hashes"].items():
            if (
                hashlib.sha256((directory / filename).read_bytes()).hexdigest()
                != digest
            ):
                raise ValueError(f"fixture checksum differs: {filename}")
        n, b, h = c["blocks"], c["block"], c["kv_heads"]
        q = read_bf16(directory / "q.bf16", (1, h * 2, 256)).cuda()
        k = read_bf16(directory / "k.bf16", (n, b, h, 256)).cuda()
        v = read_bf16(directory / "v.bf16", (n, b, h, 256)).cuda()
        k = k.reshape(n, b, h, 32, 8).permute(0, 2, 3, 1, 4).contiguous()
        v = v.permute(0, 2, 3, 1).contiguous()
        blocks = (
            torch.from_numpy(np.fromfile(directory / "blocks.i32", np.int32))
            .cuda()
            .reshape(1, n)
        )
        lens = torch.tensor([c["seq_len"]], dtype=torch.int32, device="cuda")
        starts = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
        one = torch.ones((), device="cuda")
        out = torch.empty_like(q)
        # chunked_prefill_paged_decode.py:461-514 at e126687a9a.
        kernel_paged_attention_2d[(1, h)](
            out,
            q,
            k,
            v,
            None,
            blocks,
            lens,
            None,
            c["scale"],
            one,
            one,
            1.0,
            h * 2,
            2,
            16,
            blocks.stride(0),
            q.stride(0),
            q.stride(1),
            out.stride(0),
            out.stride(1),
            b if b & (b - 1) == 0 else 32,
            b,
            256,
            256,
            False,
            c["window_left"] + 1 if c["window_left"] >= 0 else 0,
            8,
            *k.stride(),
            *v.stride(),
            True,
            starts,
            False,
            False,
        )
        torch.cuda.synchronize()
        out.cpu().view(torch.uint16).numpy().tofile(output / (c["name"] + ".bf16"))
        print("PRIMARY DECODE", c["name"], flush=True)
    (output / "revision.txt").write_text(vllm.__version__ + "\n")


def compare(manifest, reference, native):
    results = []
    for c in json.loads(manifest.read_text())["cases"]:
        name = c["name"] + ".bf16"
        a = np.fromfile(reference / name, np.uint16)
        b = np.fromfile(native / name, np.uint16)
        results.append(
            {
                "name": c["name"],
                "words": len(a),
                "different": (
                    int(np.count_nonzero(a != b)) if a.shape == b.shape else -1
                ),
                "exact": np.array_equal(a, b),
            }
        )
    print(json.dumps(results, indent=2))
    if not all(r["exact"] for r in results):
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("generate", "capture", "compare"))
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    {"generate": generate, "capture": capture, "compare": compare}[args.mode](
        *args.paths
    )


if __name__ == "__main__":
    main()
