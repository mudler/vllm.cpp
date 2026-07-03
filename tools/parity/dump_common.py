# vllm.cpp parity harness; oracle = upstream vLLM (see .agents/upstream-sync.md pin).
import json
import pathlib

import numpy as np
import torch

UPSTREAM_PIN = "e24d1b24"

DTYPE_NAMES = {torch.float32: "f32", torch.bfloat16: "bf16", torch.float16: "f16",
               torch.int32: "i32", torch.int64: "i64"}


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    t = t.detach()
    if t.dtype in (torch.bfloat16, torch.float16):
        return t.contiguous().view(torch.uint16).cpu().numpy()
    return t.contiguous().cpu().numpy()


def save_case(root: pathlib.Path, name: str, op: str, tensors: dict, args: dict,
              inputs: list, outputs: list, tol: dict) -> None:
    import vllm
    case = root / name
    case.mkdir(parents=True, exist_ok=True)
    manifest = {"op": op, "args": args, "tensors": {}, "inputs": inputs,
                "outputs": outputs, "tol": tol,
                "oracle": {"vllm_version": vllm.__version__, "upstream_pin": UPSTREAM_PIN,
                           "torch": torch.__version__.split("+")[0]}}
    for key, t in tensors.items():
        np.save(case / f"{key}.npy", _to_numpy(t))
        manifest["tensors"][key] = {"file": f"{key}.npy", "dtype": DTYPE_NAMES[t.dtype],
                                    "shape": list(t.shape)}
    (case / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {case}")
