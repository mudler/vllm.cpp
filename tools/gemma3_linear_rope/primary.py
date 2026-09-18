#!/usr/bin/env python3
"""Export pinned vLLM linear-RoPE caches and native rotations on gfx1100."""
import argparse
import json
from pathlib import Path

import torch
import vllm
from vllm.model_executor.layers.rotary_embedding.linear_scaling_rope import (
    LinearScalingRotaryEmbedding,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    if "e126687a9" not in vllm.__version__:
        raise RuntimeError(f"wrong vLLM pin: {vllm.__version__}")
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    cases = []
    # Extend the primary module-cache linear arm (factor tuple (1,)) with
    # Gemma's factor 8 and multi-factor/fractional-length coverage.
    for factors in ([1.], [8.], [1., 1.5, 8.]):
        for dtype in (torch.float32, torch.bfloat16):
            for neox in (False, True):
                rope = LinearScalingRotaryEmbedding(16, 8, 17, 1000000., neox, factors, dtype)
                rows = rope.cos_sin_cache.shape[0]
                positions = torch.tensor([0, 1, 7, rows - 1], dtype=torch.long)
                q = torch.randn((4, 2, 16), dtype=dtype)
                k = torch.randn((4, 1, 16), dtype=dtype)
                qo, ko = rope.forward_native(positions, q, k)
                q_only, absent_key = rope.forward_native(positions, q, None)
                assert absent_key is None
                torch.testing.assert_close(q_only, qo)
                def floats(t):
                    return t.float().cpu().flatten().tolist()
                cases.append(dict(factors=factors, dtype=str(dtype).split(".")[1],
                    neox=neox, rows=rows, positions=positions.cpu().tolist(),
                    offsets={str(f): v for f, v in rope.scaling_factor_to_offset.items()},
                    cache=floats(rope.cos_sin_cache), q=floats(q), k=floats(k),
                    q_out=floats(qo), k_out=floats(ko)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(vllm_version=vllm.__version__, seed=0,
        upstream="tests/kernels/core/test_pos_encoding.py:125-173", cases=cases),
        separators=(",", ":")) + "\n")
    print("EXPORTED", len(cases), "cases", flush=True)


if __name__ == "__main__":
    from vllm.config import VllmConfig, set_current_vllm_config
    # Match the primary suite's default_vllm_config fixture for CustomOp.
    with set_current_vllm_config(VllmConfig()):
        main()
