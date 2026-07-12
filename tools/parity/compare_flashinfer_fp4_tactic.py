#!/usr/bin/env python3
"""Compare one vllm.cpp NVFP4 raw dump with an exact FlashInfer tactic.

The dump is produced by the focused C++ test with VT_FP4_TEST_DUMP_DIR. This
script calls FlashInfer's raw SM120 runner (not its autotuner) on the identical
packed operands, 128x4 scale layouts, alpha and output shape, then reports both
BF16 bit equality and numerical deltas.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from flashinfer.gemm.gemm_base import (
    DEFAULT_WORKSPACE_SIZE,
    _MM_FP4_TUNING_CONFIG_128x4,
    get_gemm_sm120_module_cutlass_fp4,
)
from flashinfer.autotuner import AutoTuner, autotune


def load(path: Path, dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
    array = np.fromfile(path, dtype=dtype)
    expected = int(np.prod(shape))
    if array.size != expected:
        raise RuntimeError(f"{path}: expected {expected} elements, found {array.size}")
    return array.reshape(shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump", type=Path)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--tactic", type=int)
    selection.add_argument("--tune", action="store_true")
    args = parser.parse_args()

    m, n, k, mp, npad, kp = map(
        int, (args.dump / "shape.txt").read_text(encoding="utf-8").split()
    )
    device = torch.device("cuda:0")
    a = torch.from_numpy(load(args.dump / "a_packed.bin", np.uint8, (m, k // 2))).to(
        device
    )
    b_nk = torch.from_numpy(
        load(args.dump / "b_packed.bin", np.uint8, (n, k // 2))
    ).to(device)
    a_sf = torch.from_numpy(load(args.dump / "a_sf_sw.bin", np.uint8, (mp, kp))).to(
        device
    )
    b_sf = torch.from_numpy(
        load(args.dump / "b_sf_sw.bin", np.uint8, (npad, kp))
    ).to(device)
    alpha = torch.from_numpy(load(args.dump / "alpha.bin", np.float32, (1,))).to(device)
    expected_bits = load(args.dump / "out_bf16.bin", np.uint16, (m, n))

    out = torch.empty((m, n), dtype=torch.bfloat16, device=device)
    workspace = torch.empty(DEFAULT_WORKSPACE_SIZE, dtype=torch.uint8, device=device)
    runner = get_gemm_sm120_module_cutlass_fp4().cutlass_fp4_gemm_runner()
    inputs = [
        a,
        b_nk.T,
        a_sf,
        b_sf.T,
        alpha,
        torch.bfloat16,
        out,
        16,
        True,
        workspace,
    ]
    tactic = args.tactic
    if args.tune:
        with autotune(tune_mode=True):
            runner, tactic = AutoTuner.get().choose_one(
                "fp4_gemm", [runner], _MM_FP4_TUNING_CONFIG_128x4, inputs
            )
        print(f"flashinfer_selected_tactic={tactic}")
    assert tactic is not None
    runner.forward(inputs=inputs, tactic=tactic)
    torch.cuda.synchronize()

    actual_bits = out.view(torch.uint16).cpu().numpy()
    expected = torch.from_numpy(expected_bits).view(torch.bfloat16).float().numpy()
    actual = out.float().cpu().numpy()
    differing = int(np.count_nonzero(actual_bits != expected_bits))
    maximum = float(np.max(np.abs(actual - expected)))
    print(
        f"shape={m}x{n}x{k} tactic={tactic} "
        f"bf16_differing={differing}/{expected_bits.size} max_abs={maximum:.9g}"
    )
    if differing:
        first = tuple(int(index) for index in np.argwhere(actual_bits != expected_bits)[0])
        print(
            f"first={first} ours_bits=0x{int(expected_bits[first]):04x} "
            f"flashinfer_bits=0x{int(actual_bits[first]):04x} "
            f"ours={float(expected[first]):.9g} flashinfer={float(actual[first]):.9g}"
        )


if __name__ == "__main__":
    main()
