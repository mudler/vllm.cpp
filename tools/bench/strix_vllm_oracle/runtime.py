#!/usr/bin/env python3
"""Current-pin runtime adapter; invoked only inside the worker's isolated venv."""
import argparse
import dataclasses
import importlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys

# Exact historical prompt bytes and llama.cpp prompt IDs, gen_rocm.py at #2740.
PROMPTS = ["The capital city of France is", "The three primary colors are",
           "Water boils at a temperature of", "The Pythagorean theorem states that",
           "In 1969, humans first walked on", "A prime number is a natural number"]
PROMPT_IDS = [[760, 6511, 3177, 314, 9338, 369], [760, 2250, 5839, 7736, 513],
              [27336, 85895, 506, 264, 9039, 314], [760, 5187, 92068, 43687, 55877, 5134, 421],
              [623, 220, 16, 24, 21, 24, 11, 12313, 1118, 14428, 383],
              [32, 9944, 1324, 369, 264, 5629, 1324]]
ENGINE_KWARGS = dict(gpu_memory_utilization=0.60, max_model_len=2048, max_num_seqs=1,
                     max_num_batched_tokens=2048, limit_mm_per_prompt={"image": 0, "video": 0},
                     enforce_eager=False, trust_remote_code=False, quantization="gguf")


def identity():
    import torch
    import torchvision
    import triton
    import vllm
    import vllm_gguf_plugin
    from vllm.platforms import current_platform
    from vllm_gguf_plugin import ops

    extensions = [importlib.import_module(name).__file__ for name in
                  ("vllm._C", "vllm._rocm_C", "vllm_gguf_plugin._C_gguf")]
    paths = [vllm.__file__, vllm_gguf_plugin.__file__, triton.__file__, *extensions]
    if not all(Path(p).resolve().is_relative_to(Path(sys.prefix).resolve()) for p in paths):
        raise ValueError("runtime imported outside isolated venv")
    if not torch.version.hip or not current_platform.is_rocm():
        raise ValueError("runtime platform is not ROCm")
    arch = torch.cuda.get_device_properties(0).gcnArchName
    if arch.split(":")[0] != "gfx1151":
        raise ValueError("runtime device is not gfx1151")
    entries = [(e.name, e.value) for e in metadata.entry_points(group="vllm.general_plugins")]
    if not any(name == "gguf" for name, _ in entries):
        raise ValueError("GGUF plugin is not registered")
    predicates = {name: ops._cuda_kernel_available("ggml_mul_mat_vec_a8", getattr(ops, "GGML_TYPE_" + name))
                  for name in ("Q4_K", "Q5_K", "Q6_K", "Q8_0")}
    try:
        triton_distribution = metadata.version("triton")
    except metadata.PackageNotFoundError:
        triton_distribution = None
    return dict(runtime_version=vllm.__version__, distribution_version=metadata.version("vllm"),
                platform="rocm", device_arch=arch, torch=torch.__version__, torchvision=torchvision.__version__,
                triton=triton.__version__, triton_rocm_distribution=metadata.version("triton-rocm"),
                triton_distribution=triton_distribution, triton_path=triton.__file__,
                vllm_path=vllm.__file__, plugin_path=vllm_gguf_plugin.__file__, extensions=extensions,
                plugin_predicates=predicates, plugin_entry_points=entries,
                plugin_version=metadata.version("vllm-gguf-plugin"))


def projection_metadata(model):
    # Pinned LLM.apply_model at entrypoints/llm.py:599. No forward replacement.
    records = []
    for name, module in model.named_modules():
        if "in_proj" in name:
            records.append(dict(name=name, module=type(module).__name__,
                                quant_method=type(getattr(module, "quant_method", None)).__name__,
                                parameters=[dict(name=n, dtype=str(p.dtype), shape=list(p.shape))
                                            for n, p in module.named_parameters(recurse=False)]))
    return {"projections": records, "projection_output_dtype": "PENDING: parameter metadata is not a forward capture"}


def generate(assets, output):
    runtime = identity()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tokenizer = AutoTokenizer.from_pretrained(str(assets / "tokenizer"), trust_remote_code=False, local_files_only=True)
    ids = [tokenizer(prompt, add_special_tokens=False)["input_ids"] for prompt in PROMPTS]
    if ids != PROMPT_IDS:
        raise ValueError("tokenization differs from explicit oracle prompt IDs")
    kwargs = dict(ENGINE_KWARGS, model=str(assets / "model/Qwen3.8-27B-Q4_K_M.gguf"),
                  tokenizer=str(assets / "tokenizer"),
                  model_loader_extra_config={"mm_proj": str(assets / "mmproj/mmproj-BF16.gguf")})
    llm = LLM(**kwargs)
    config = llm.llm_engine.vllm_config
    compilation = config.compilation_config
    resolved = dict(dtype=str(config.model_config.dtype),
                    compilation_config=dataclasses.asdict(compilation) if dataclasses.is_dataclass(compilation) else vars(compilation))
    data = dict(identity=runtime, engine_kwargs=ENGINE_KWARGS, resolved_engine_kwargs=kwargs,
                resolved_config=resolved, projection_metadata=llm.apply_model(projection_metadata), records=[])
    # Persist loaded identity/config before generation, retaining partial progress.
    output.write_text(json.dumps(data, default=str, indent=2) + "\n")
    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=48, ignore_eos=True)
    outputs = llm.generate([TokensPrompt(prompt_token_ids=ids) for ids in PROMPT_IDS], sampling)
    if len(outputs) != 6:
        raise ValueError("six complete outputs required")
    for i, result in enumerate(outputs):
        ids = list(result.outputs[0].token_ids)
        if list(result.prompt_token_ids) != PROMPT_IDS[i] or len(ids) != 48:
            raise ValueError("six complete 48-token outputs required")
        data["records"].append(dict(prompt=PROMPTS[i], prompt_ids=PROMPT_IDS[i],
                                    gen_ids=ids, gen_text=result.outputs[0].text))
        output.write_text(json.dumps(data, default=str, indent=2) + "\n")
        print(f"PROMPT {i}: output_ids={ids}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("identity", "generate"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path)
    args = parser.parse_args()
    if args.mode == "identity":
        args.output.write_text(json.dumps(identity(), indent=2) + "\n")
    else:
        if args.assets is None:
            parser.error("generation requires --assets")
        generate(args.assets, args.output)


if __name__ == "__main__":
    main()
