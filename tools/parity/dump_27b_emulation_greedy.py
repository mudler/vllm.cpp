# vllm.cpp parity harness; oracle = pip vLLM 0.24.0 (same release family as pin e24d1b24).
"""Capture the 27B greedy continuation from the vLLM ORACLE forced to EMULATION.

This is the FAITHFUL reference our C++ fp4 W4A4 forward mirrors token-for-token
(ledger row 56). It is the correctness bar for test_qwen27_paged_engine.cpp — NOT
the native production-198 stream (greedy_ids.npy), which is vLLM's full-stack
flashinfer/cutlass-sm120a result on a razor near-tie (tok6 198 "\\n" vs 271 "\\n\\n").

Unlike dump_qwen36.py (which dumps the FULL logits golden case and would clobber
the production-198 golden), this writes ONLY greedy_ids_emulation.npy alongside it.

vLLM's NVFP4 kernel auto-selection (kernels/linear/__init__.py @ e24d1b24) walks
_POSSIBLE_NVFP4_KERNELS[CUDA]; on sm_121 it normally picks the
FlashInfer-cutlass / cutlass sm120a fp4×fp4 kernel (-> tok6=198). Setting
VLLM_DISABLED_KERNELS to the 7 higher-priority CUDA NVFP4 kernels leaves only
EmulationNvFp4LinearKernel -> the emulation path -> tok6=271. The selected kernel
is echoed by vLLM's "Using <Kernel> for NVFP4 GEMM" log; we also assert tok6==271.

Run on dgx (GB10) in the pip oracle venv (~/venvs/vllm-oracle, vLLM 0.24.0). The
ACTUAL working command:

    VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    VLLM_DISABLED_KERNELS=FlashInferCuteDslNvFp4LinearKernel,FlashInferCutlassNvFp4LinearKernel,CutlassNvFp4LinearKernel,MarlinNvFp4LinearKernel,FlashInferTrtllmNvFp4LinearKernel,FlashInferCudnnNvFp4LinearKernel,FbgemmNvFp4LinearKernel \
    ~/venvs/vllm-oracle/bin/python tools/parity/dump_27b_emulation_greedy.py \
        --model <27B snapshot dir> \
        --out tests/parity/goldens/qwen36_logits_27b/greedy_ids_emulation.npy
"""

import argparse
import os
import pathlib

import numpy as np
import torch

N_GREEDY = 16
EXPECT_TOK6 = 271  # emulation near-tie side (native/production is 198)

# The 7 higher-priority CUDA NVFP4 kernels above EmulationNvFp4LinearKernel in
# _POSSIBLE_NVFP4_KERNELS[CUDA] (kernels/linear/__init__.py:408-419 @ e24d1b24).
# Disabling all of them forces the emulation path. (FlashInferB12x is already
# excluded from auto-selection upstream.)
DISABLED_KERNELS = ",".join([
    "FlashInferCuteDslNvFp4LinearKernel",
    "FlashInferCutlassNvFp4LinearKernel",
    "CutlassNvFp4LinearKernel",
    "MarlinNvFp4LinearKernel",
    "FlashInferTrtllmNvFp4LinearKernel",
    "FlashInferCudnnNvFp4LinearKernel",
    "FbgemmNvFp4LinearKernel",
])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True,
                    help="path to greedy_ids_emulation.npy")
    ap.add_argument("--prompt",
                    default="The capital of France is Paris, and the")
    ap.add_argument("--gpu-mem", type=float, default=0.5)
    args = ap.parse_args()

    assert os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") == "0", \
        "set VLLM_ENABLE_V1_MULTIPROCESSING=0"
    disabled = os.environ.get("VLLM_DISABLED_KERNELS", "")
    assert "EmulationNvFp4LinearKernel" not in disabled, \
        "do NOT disable EmulationNvFp4LinearKernel"
    for k in ("FlashInferCutlassNvFp4LinearKernel", "CutlassNvFp4LinearKernel"):
        assert k in disabled, (
            f"VLLM_DISABLED_KERNELS must disable {k} to force emulation; "
            f"expected: VLLM_DISABLED_KERNELS={DISABLED_KERNELS}")

    import vllm
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    print("vllm from:", vllm.__file__, "version", vllm.__version__)
    print("VLLM_DISABLED_KERNELS:", disabled)

    tok = AutoTokenizer.from_pretrained(args.model)
    ids = tok(args.prompt, add_special_tokens=True)["input_ids"]
    print("PROMPT_IDS", ids, "len", len(ids))

    llm = LLM(model=args.model, enforce_eager=True, tensor_parallel_size=1,
              max_model_len=256, gpu_memory_utilization=args.gpu_mem,
              max_num_seqs=1, dtype="bfloat16")

    sp = SamplingParams(temperature=0.0, max_tokens=N_GREEDY)
    out = llm.generate({"prompt_token_ids": ids}, sp)
    greedy_ids = list(out[0].outputs[0].token_ids)
    print("EMULATION GREEDY_TOKEN_IDS", greedy_ids)
    print("EMULATION GREEDY_TEXT", repr(out[0].outputs[0].text))
    print("tok6 =", greedy_ids[6])

    assert len(greedy_ids) == N_GREEDY, (len(greedy_ids), N_GREEDY)
    assert greedy_ids[6] == EXPECT_TOK6, (
        f"tok6={greedy_ids[6]} != {EXPECT_TOK6}; the emulation path was NOT "
        f"forced (kernel auto-selection likely picked cutlass/flashinfer). "
        f"Check VLLM_DISABLED_KERNELS and the 'Using <Kernel> for NVFP4 GEMM' log.")

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, np.asarray(greedy_ids, dtype=np.int32))
    print(f"wrote {out_path} (16 i32 emulation greedy ids, tok6={EXPECT_TOK6})")


if __name__ == "__main__":
    main()
