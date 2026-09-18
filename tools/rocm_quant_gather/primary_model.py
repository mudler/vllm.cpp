#!/usr/bin/env python3
"""Run one bounded model request through the pinned plugin production path.

Each process makes one fresh engine. Repeat for both prompt indices and all
three repeats only after the first model load succeeds. A refusal is recorded
with its traceback and a nonzero exit; operation parity cannot replace it.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import traceback

from primary import seal


def capture_cache_memory(worker):
    """Read the executing worker's cache views and allocator blocks.

    This callback observes both runner versions at e126687a9a. It does not
    replace an allocator or change a tensor. rocprofv3 records the backing
    allocations independently on the native and primary processes.
    """
    import torch
    import json

    torch.cuda.synchronize()
    runner = worker.model_runner
    snapshot = torch.cuda.memory_snapshot()
    tensors = []
    for index, tensor in enumerate(runner.kv_caches):
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError("bounded full-attention cache is not a tensor")
        storage = tensor.untyped_storage()
        address = storage.data_ptr()
        allocation = None
        for segment in snapshot:
            offset = segment["address"]
            for block in segment["blocks"]:
                start = block.get("address", offset)
                if start <= address < start + block["size"]:
                    allocation = {
                        "block_address": start, "block_bytes": block["size"],
                        "requested_bytes": block.get("requested_size"),
                        "state": block["state"],
                        "segment_address": segment["address"],
                        "segment_bytes": segment["total_size"],
                        "segment_allocated_bytes": segment["allocated_size"],
                        "segment_active_bytes": segment["active_size"],
                    }
                offset = start + block["size"]
        tensors.append({
            "index": index, "dtype": str(tensor.dtype),
            "shape": list(tensor.shape), "stride_elements": list(tensor.stride()),
            "element_bytes": tensor.element_size(), "numel": tensor.numel(),
            "payload_bytes": tensor.numel() * tensor.element_size(),
            "storage_bytes": storage.nbytes(), "storage_address": address,
            "storage_offset_elements": tensor.storage_offset(),
            "contiguous": tensor.is_contiguous(), "device": str(tensor.device),
            "allocator": allocation,
        })
    if not tensors:
        raise RuntimeError("executing worker exposes no full-attention cache")
    layout = worker.vllm_config.cache_config.get_resolved_kv_cache_layout()
    report = {
        "runner_class": type(runner).__module__ + "." + type(runner).__name__,
        "physical_blocks": runner.kv_cache_config.num_blocks,
        "resolved_layout": layout.name,
        "layout_stride_order": list(layout.stride_order),
        "resolved_model_dtype": str(runner.dtype),
        "cache_tensors": tensors,
        "process_allocated_bytes": torch.cuda.memory_allocated(),
        "process_reserved_bytes": torch.cuda.memory_reserved(),
        "process_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "measurement": "live tensor views and PyTorch allocator snapshot; backing allocations require rocprofv3",
    }
    # The resolved layout is an Enum with a tuple value at this pin. Serialize
    # a complete JSON value in the worker, so RPC sees only a bounded string.
    return json.dumps(report, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prompt-index", type=int, choices=(0, 1), default=0)
    parser.add_argument("--repeat", type=int, choices=(0, 1, 2), default=0)
    args = parser.parse_args()
    prompt = ([1, 0, 63, 127, 63], [1, 127, 0, 127])[args.prompt_index]
    report = {"model": seal(args.model), "config": seal(args.config / "config.json"),
              "primary_pin": "e126687a9a828d513c01a07cd69f025f27d63280",
              "plugin_pin": "d4c1f0d082fc7cd4350da56689109a01c1f29d6c",
              "prompt": prompt, "repeat": args.repeat, "tokens": [],
              "requested": {"model_dtype": "auto from bfloat16 config", "block_size": 16,
                            "num_gpu_blocks_override": 16, "max_model_len": 64,
                            "max_num_seqs": 1, "kv_cache_dtype": "auto",
                            "seed": 0x524F434D, "greedy": True, "ignore_eos": True,
                            "max_tokens": 4, "stop": [], "stop_token_ids": []},
              "harness_adaptations": ["pre-tokenized IDs with skip_tokenizer_init=True",
                                      "explicit local HF text config mirrors GGUF geometry",
                                      "development model_class_overrides selects the pinned text class for the plugin architecture key",
                                      "top-level partial_rotary_factor=1.0 preserves GGUF rotary width 64 during HF normalization",
                                      "16 physical KV blocks match secondary 256-cell allocation; one primary null block is reserved; logical limit remains 64",
                                      "no oracle patch or eager-mode override"],
              "status": "PENDING"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        import torch
        import vllm
        from vllm import LLM, SamplingParams
        from vllm.inputs import TokensPrompt

        report.update(torch=torch.__version__, torch_git=torch.version.git_version,
                      hip=torch.version.hip, vllm=vllm.__version__,
                      plugin=importlib.metadata.version("vllm-gguf-plugin"))
        model = LLM(model=str(args.model), hf_config_path=str(args.config),
                    model_class_overrides={"Qwen3_5ForConditionalGeneration":
                        "vllm.model_executor.models.qwen3_5:Qwen3_5ForCausalLM"},
                    hf_overrides={"partial_rotary_factor": 1.0},
                    skip_tokenizer_init=True, dtype="auto", block_size=16,
                    num_gpu_blocks_override=16, max_model_len=64, max_num_seqs=1,
                    kv_cache_dtype="auto", seed=0x524F434D)
        config = model.llm_engine.vllm_config
        report["resolved_model_dtype"] = str(config.model_config.dtype)
        report["resolved_cache_config"] = str(config.cache_config)
        report["memory_before_generation"] = [json.loads(item) for item in
            model.collective_rpc(capture_cache_memory, timeout=30)]
        report["cache_capacity_contract"] = {"block_size": 16, "physical_blocks": 16,
            "physical_cells": 256, "reserved_null_blocks": 1, "usable_blocks": 15,
            "usable_cells": 240, "logical_max_model_len": 64, "bf16_kv_payload_bytes": 65536,
            "source": "vllm/v1/core/kv_cache_utils.py:2304-2309 at e126687a9a"}
        params = SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True,
                                seed=0x524F434D, stop=[], stop_token_ids=[])
        output = model.generate([TokensPrompt(prompt_token_ids=prompt)], params, use_tqdm=False)
        if len(output) != 1 or list(output[0].prompt_token_ids) != prompt:
            raise RuntimeError("the oracle changed the request or prompt IDs")
        report["tokens"] = list(output[0].outputs[0].token_ids)
        report["memory_after_generation"] = [json.loads(item) for item in
            model.collective_rpc(capture_cache_memory, timeout=30)]
        if len(report["tokens"]) != 4:
            raise RuntimeError("the oracle did not generate exactly four tokens")
        report["status"] = "EXECUTED; token comparison remains required"
        print(json.dumps(report, sort_keys=True), flush=True)
    except Exception as error:
        report["status"] = "PENDING: primary model refused before qualification"
        report["exception_type"] = type(error).__name__
        report["exception"] = str(error)
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
