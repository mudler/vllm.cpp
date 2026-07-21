#!/usr/bin/env python3
# vllm.cpp original (ADDITIVE-MODEL W0 tooling for OPT-125m, `OPTForCausalLM`).
#
# Materialize facebook/opt-125m as a BF16 *safetensors* model dir.
#
# The HF hub snapshot ships `pytorch_model.bin` (a torch pickle, fp16) and a
# GPT2-style vocab.json/merges.txt pair with NO `tokenizer.json`. vllm.cpp's
# loader reads safetensors + a fast `tokenizer.json`, so this script performs a
# one-time, container-level materialization:
#
#   * torch pickle -> safetensors          (container change only)
#   * fp16         -> bf16                 (ONE rounding — exactly what the vLLM
#                                           oracle does under `--dtype bfloat16`,
#                                           the arm the SACRED gate is run on;
#                                           see .agents/specs/sweep-opt-125m.md
#                                           "Risks/decisions" D1)
#   * slow GPT2 BPE -> fast tokenizer.json (`AutoTokenizer.save_pretrained`)
#
# config.json / generation_config.json are copied verbatim; the ON-DISK tensor
# names are preserved verbatim (`decoder.*`) so our loader mirrors vLLM's
# WeightsMapper (opt.py:328-338) rather than a pre-baked rename.
#
# Usage:
#   python3 scripts/opt-materialize-checkpoint.py <hf_snapshot_dir> <out_dir>
import pathlib
import shutil
import sys

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer


def main() -> int:
    src = pathlib.Path(sys.argv[1])
    dst = pathlib.Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)

    sd = torch.load(src / "pytorch_model.bin", map_location="cpu", weights_only=True)
    out = {}
    seen = set()
    for k, v in sd.items():
        t = v.to(torch.bfloat16).contiguous()
        if t.data_ptr() in seen:  # tied lm_head/embed share one storage
            t = t.clone()
        seen.add(t.data_ptr())
        out[k] = t

    save_file(out, str(dst / "model.safetensors"), metadata={"format": "pt"})

    for f in ("config.json", "generation_config.json"):
        if (src / f).exists():
            shutil.copy(src / f, dst / f)

    tok = AutoTokenizer.from_pretrained(str(src), use_fast=True)
    tok.save_pretrained(str(dst))

    print("tensors:", len(out))
    for k, v in sorted(out.items()):
        print(f"  {k:60s} {str(list(v.shape)):20s} {v.dtype}")
    print("wrote", dst)
    print("tokenizer.json:", (dst / "tokenizer.json").exists())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
