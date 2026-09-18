"""Run and export the pinned upstream single-device BF16 MoE cases (#3094).

The original test function executes without reduced shapes, changed seeds,
tolerances, references, padding modes, or graph settings. Only its inputs and
outputs are exported for the independent C++ provider comparison. The existing
local pointer-array ABI materializes logical [K,N] matrices contiguously. Padded
oracle calls remain distinct upstream cases; they do not test native padded strides.
"""
import argparse
import hashlib
import importlib
import itertools
import json
import sys
from pathlib import Path

PIN = "e126687a9a828d513c01a07cd69f025f27d63280"
SHAPES = [(1, 128, 128), (1, 2048, 128), (33, 2048, 128),
          (32768, 2048, 511), (40000, 1024, 1024)]
CASES = list(itertools.product(SHAPES, (8, 64, 192), (2, 6), (False, True)))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", type=int, required=True, choices=range(len(CASES)))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import torch
    import vllm
    assert vllm.__version__ == "0.28.1rc1.dev132+ge126687a9"
    # Import the installed pinned runtime first, then its exact source tests.
    sys.path.insert(0, str(args.source))
    upstream = importlib.import_module("tests.kernels.moe.test_moe")
    import pytest
    from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager
    init_workspace_manager(torch.device("cuda:0"))
    shape, experts, top_k, padding = CASES[args.case]
    m, n, k = shape
    files = {}
    runtime_calls = []
    captured_reference = [False]

    def save(name, tensor):
        cpu = tensor.detach().contiguous().cpu()
        raw = cpu.view(torch.uint16).numpy() if cpu.dtype == torch.bfloat16 else cpu.numpy()
        path = args.output / name
        raw.tofile(path)
        files[name] = {"shape": list(tensor.shape), "source_stride": list(tensor.stride()),
                       "export_stride": list(cpu.stride()),
                       "dtype": str(tensor.dtype), "bytes": path.stat().st_size,
                       "sha256": sha256(path)}

    original = upstream.run_moe_test

    def export_run(baseline, moe_fn, **kwargs):
        outputs = []
        inputs = []

        def capture(*positional, **keywords):
            inputs[:] = positional
            output = moe_fn(*positional, **keywords)
            outputs[:] = [output]
            return output

        reference = original(baseline, capture, **kwargs)
        a, w1, w2, score = inputs[:4]
        pad = 128 if padding else 0
        assert list(w1.stride()) == [2 * n * (k + pad), k + pad, 1]
        assert list(w2.stride()) == [k * (n + pad), n + pad, 1]
        runtime_calls.append({"implementation": getattr(moe_fn, "__name__", str(moe_fn)),
                              "padding": kwargs["padding"],
                              "w1_stride": list(w1.stride()),
                              "w2_stride": list(w2.stride()),
                              "use_compile": kwargs.get("use_compile", False),
                              "use_cudagraph": kwargs.get("use_cudagraph", False),
                              "atol": 0.02, "rtol": 0})
        if not captured_reference[0]:
            captured_reference[0] = True
            routes, ids, _ = upstream.fused_topk(a, score, top_k, False)
            save("a.bf16", a)
            save("score.bf16", score)
            # Preserve the actual original padded storage as evidence. The
            # dense exports below are the explicit loader/ABI adaptation.
            assert w1.storage_offset() == w2.storage_offset() == 0
            save("oracle-w1-storage.bf16", w1.as_strided((experts, 2 * n, k + pad), w1.stride()))
            save("oracle-w2-storage.bf16", w2.as_strided((experts, k, n + pad), w2.stride()))
            # Required storage adaptation: [E,N,K] checkpoint matrices become
            # one [K,N] Matmul-B pointer per expert. Logical values stay BF16.
            save("gate.bf16", w1[:, :n, :].transpose(1, 2))
            save("up.bf16", w1[:, n:, :].transpose(1, 2))
            save("down.bf16", w2.transpose(1, 2))
            save("ids.i32", ids.to(torch.int32).reshape(-1))
            save("routes.f32", routes.float().reshape(-1))
            save("reference.bf16", reference)
        # The last call is modular Triton, the selected production backend.
        save("triton.bf16", outputs[0])
        return reference

    upstream.run_moe_test = export_run
    try:
        with pytest.MonkeyPatch.context() as monkeypatch:
            upstream.test_fused_moe(m=m, n=n, k=k, e=experts, topk=top_k,
                                   ep_size=1, dtype=torch.bfloat16, padding=padding,
                                   use_td=False, monkeypatch=monkeypatch, workspace_init=None)
        torch.cuda.synchronize()
    finally:
        upstream.run_moe_test = original
        reset_workspace_manager()
    assert len(runtime_calls) == 3
    expected_graph = n >= 1024 and k >= 1024
    assert all(call["use_cudagraph"] == expected_graph for call in runtime_calls[1:])
    record = {"case": args.case, "pin": PIN, "M": m, "N": n, "K": k,
              "experts": experts, "top_k": top_k, "padding": padding,
              "seed": 7, "renormalize": False, "use_compile": False,
              "use_cudagraph": expected_graph, "atol": 0.02, "rtol": 0,
              "upstream_test_sha256": sha256(args.source / "tests/kernels/moe/test_moe.py"),
              "upstream_calls": runtime_calls, "files": files,
              "storage_adaptation": "Actual padded upstream weight storage is retained. Native expert pointers address contiguous logical [K,N] matrices, as the existing shared ABI requires. This is differential padded-oracle coverage, not native padded-stride coverage.",
              "exclusions": {
                  "ep_size=4": "This row adds no distributed execution path",
                  "use_td=true": "gfx1100 is excluded by fused_moe/utils.py:665-684"}}
    (args.output / "case.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"case": args.case, "shape": shape, "experts": experts,
                      "top_k": top_k, "padding": padding,
                      "upstream": "PASS", "graph": expected_graph}))


if __name__ == "__main__":
    main()
