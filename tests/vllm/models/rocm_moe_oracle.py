"""Pinned vLLM production oracle for BACKEND-ROCM-BF16-MOE (#3094).

This is an evidence harness, not a denominator timing run. It keeps the default
engine compilation and backend policy and records the executing native tensors.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_and_capture(model, output):
    import importlib
    import torch

    output = Path(output)
    metadata = []
    for name, module in model.named_modules():
        method = getattr(module, "quant_method", None)
        backend = getattr(method, "unquantized_backend", None)
        if backend is None:
            continue
        record = {"module": name, "backend": str(backend),
                  "expert_class": str(getattr(method, "experts_cls", None)),
                  "weights": {}}
        for weight_name in ("w13_weight", "w2_weight"):
            weight = getattr(module, weight_name, None)
            if weight is not None:
                record["weights"][weight_name] = {
                    "shape": list(weight.shape), "stride": list(weight.stride()),
                    "dtype": str(weight.dtype), "element_size": weight.element_size()}
        metadata.append(record)
    assert metadata, "No executing unquantized MoE method found"
    assert all(item["backend"] == "UnquantizedMoeBackend.TRITON" for item in metadata), metadata

    module = importlib.import_module("vllm.model_executor.layers.fused_moe.experts.triton_moe")
    original = module.invoke_fused_moe_triton_kernel
    count = [0]

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        if count[0] >= 32 or torch.cuda.is_current_stream_capturing():
            return result
        index = count[0]
        count[0] += 1
        names = ("A", "B", "C", "A_scale", "B_scale", "topk_weights",
                 "sorted_token_ids", "expert_ids", "num_tokens_post_padded")
        record = {"index": index, "mul_routed_weight": args[9],
                  "top_k": args[10], "config": args[11],
                  "compute_type": str(kwargs.get("compute_type")), "tensors": {}}
        tensors = {}
        for name, value in zip(names, args):
            if isinstance(value, torch.Tensor):
                record["tensors"][name] = {"shape": list(value.shape),
                    "stride": list(value.stride()), "dtype": str(value.dtype),
                    "element_size": value.element_size()}
                tensors[name] = value.detach().cpu()
        torch.save(tensors, output / f"kernel-{index:03d}.pt")
        (output / f"kernel-{index:03d}.json").write_text(json.dumps(record, indent=2) + "\n")
        return result

    module.invoke_fused_moe_triton_kernel = capture
    return metadata


def install_schedule_capture(worker):
    """Observe CPU scheduler metadata, including calls that replay a GPU graph."""
    assert worker.use_v2_model_runner
    assert type(worker.model_runner).__module__ == "vllm.v1.worker.gpu.model_runner"
    worker._moe_schedule_records = []
    original = worker.execute_model

    def capture(scheduler_output):
        record = {"total_tokens": scheduler_output.total_num_scheduled_tokens,
                  "scheduled": dict(scheduler_output.num_scheduled_tokens),
                  "new": [{"request_id": request.req_id,
                           "prompt_token_ids": request.prompt_token_ids,
                           "computed_tokens": request.num_computed_tokens}
                          for request in scheduler_output.scheduled_new_reqs]}
        result = original(scheduler_output)
        # gpu/model_runner.py:1799-1811 publishes the executed V2 batch here.
        # The legacy runner's input_batch member does not exist on this path.
        record["runner_class"] = str(type(worker.model_runner))
        record["runner_request_ids"] = (
            list(worker.model_runner.execute_model_state.input_batch.req_ids)
            if record["total_tokens"] else [])
        worker._moe_schedule_records.append(record)
        return result

    worker.execute_model = capture
    return True


def take_schedule_capture(worker):
    records = worker._moe_schedule_records
    worker._moe_schedule_records = []
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import torch
    import vllm
    from vllm import LLM, SamplingParams

    properties = torch.cuda.get_device_properties(0)
    assert "gfx1100" in properties.gcnArchName
    assert vllm.__version__ == "0.28.1rc1.dev132+ge126687a9", vllm.__version__
    artifacts = {name: {"bytes": (args.fixture / name).stat().st_size,
                        "sha256": sha256(args.fixture / name)}
                 for name in ("config.json", "model.safetensors")}
    llm = LLM(model=str(args.fixture), tokenizer=None, skip_tokenizer_init=True,
              dtype="bfloat16", seed=7, max_model_len=256, max_num_seqs=2,
              max_num_batched_tokens=256, kv_cache_memory_bytes=128 * 1024 * 1024,
              max_logprobs=128)  # Capture-only: retain the complete vocabulary.
    metadata = llm.apply_model(lambda model: inspect_and_capture(model, str(args.output)))
    assert llm.collective_rpc(install_schedule_capture) == [True]
    print(json.dumps({"selected_after_load": metadata}), flush=True)
    params = SamplingParams(temperature=0, top_p=1, top_k=-1, min_p=0,
                            repetition_penalty=1, presence_penalty=0,
                            frequency_penalty=0, seed=7, ignore_eos=True,
                            min_tokens=8, max_tokens=8, logprobs=128)
    runs = []
    # A prefill of 33 tokens executes the module after capture hooks are installed.
    for length in (33, 1, 3):
        for concurrency in (1, 2):
            prompts = [{"prompt_token_ids": [1 + ((11 + 29 * r + 17 * i) % 126)
                                              for i in range(length)]}
                       for r in range(concurrency)]
            reference = None
            for repeat in range(3):
                # Prefix caching is a production default. Reset request cache state
                # between repeats to keep the compared workloads independent.
                llm.reset_prefix_cache()
                # Pinned llm.py:727-731 documents this simultaneous-arrival
                # sequence. Level zero pauses scheduling while accepting work.
                llm.sleep(level=0)
                request_ids = llm.enqueue(prompts, params, use_tqdm=False)
                # input_processor.py:260-279 assigns internal IDs. The output
                # processor keeps their authoritative external mapping until
                # completion (output_processor.py:543-572).
                states = llm.llm_engine.output_processor.request_states
                external_ids = [states[request_id].external_req_id for request_id in request_ids]
                assert [states[request_id].prompt_token_ids for request_id in request_ids] == [
                    item["prompt_token_ids"] for item in prompts]
                llm.wake_up(tags=["scheduling"])
                outputs = llm.wait_for_completion(use_tqdm=False)
                schedule, = llm.collective_rpc(take_schedule_capture)
                schedule_path = args.output / f"schedule-L{length}-C{concurrency}-R{repeat}.json"
                schedule_path.write_text(json.dumps({"request_ids": request_ids,
                    "external_request_ids": external_ids,
                    "output_request_ids": [item.request_id for item in outputs],
                    "schedule": schedule}, indent=2) + "\n")
                assert [item.request_id for item in outputs] == external_ids
                assert [item.prompt_token_ids for item in outputs] == [
                    item["prompt_token_ids"] for item in prompts]
                active = [item for item in schedule if item["total_tokens"]]
                assert len(active) == 8, active
                assert active[0]["total_tokens"] == length * concurrency, active[0]
                assert [item["prompt_token_ids"] for item in active[0]["new"]] == [
                    item["prompt_token_ids"] for item in prompts], active[0]
                assert all(item["computed_tokens"] == 0 for item in active[0]["new"])
                internal_ids = [item["request_id"] for item in active[0]["new"]]
                assert internal_ids == request_ids
                for step, item in enumerate(active):
                    assert list(item["scheduled"]) == internal_ids, item
                    assert item["runner_request_ids"] == internal_ids, item
                    assert list(item["scheduled"].values()) == [length if step == 0 else 1] * concurrency, item
                tokens = [list(item.outputs[0].token_ids) for item in outputs]
                assert all(len(row) == 8 for row in tokens)
                if reference is None:
                    reference = tokens
                assert tokens == reference
                runs.append({"length": length, "concurrency": concurrency,
                             "repeat": repeat, "tokens": tokens,
                             "request_ids": request_ids, "external_request_ids": external_ids,
                             "schedule": schedule,
                             "logprobs": [[{str(k): v.logprob for k, v in step.items()}
                                           for step in item.outputs[0].logprobs]
                                          for item in outputs]})
    result = {"pin": "e126687a9a828d513c01a07cd69f025f27d63280",
              "vllm": vllm.__version__, "torch": torch.__version__,
              "hip": torch.version.hip, "device": properties.name,
              "architecture": properties.gcnArchName, "artifacts": artifacts,
              "selected": metadata, "runs": runs,
              "purpose": "instrumented production correctness and selection evidence"}
    (args.output / "production.json").write_text(json.dumps(result, indent=2) + "\n")
    cache = Path(os.environ["TRITON_CACHE_DIR"])
    kernels = [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
               for path in sorted(cache.rglob("*"))
               if path.is_file() and path.suffix in (".amdgcn", ".llir", ".ttir", ".ttgir", ".json")]
    (args.output / "generated-kernels.json").write_text(json.dumps(kernels, indent=2) + "\n")
    assert kernels, "No generated Triton kernel evidence"
    captured = [json.loads(path.read_text()) for path in sorted(args.output.glob("kernel-*.json"))]
    assert any(not item["mul_routed_weight"] for item in captured), "No executing gate/up capture"
    assert any(item["mul_routed_weight"] for item in captured), "No executing weighted-down capture"
    for item in captured:
        assert item["compute_type"] == "bf16", item
        for name in ("A", "B", "C"):
            assert item["tensors"][name]["dtype"] == "torch.bfloat16", item
    print(json.dumps({"selected": metadata, "runs": len(runs), "artifacts": artifacts,
                      "generated_files": len(kernels)}))


if __name__ == "__main__":
    main()
