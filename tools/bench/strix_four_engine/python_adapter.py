"""Persistent public-API clients for the pinned vLLM and patched SGLang."""
import asyncio
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tools.bench.strix_four_engine.qualify import PROMPTS, RESOLVED, parse, tokens


class PythonEngine:
    def __init__(self, config):
        from transformers import AutoTokenizer
        self.name = config["engine"]
        tokenizer = AutoTokenizer.from_pretrained(config["model"], local_files_only=True,
                                                   trust_remote_code=False)
        self.prompts = [tokenizer.encode(p, add_special_tokens=False) for p in PROMPTS]
        if config["prompts"] != PROMPTS or (config["prompt_ids"] is not None and self.prompts != config["prompt_ids"]):
            raise ValueError("canonical tokenizer mismatch")
        for prompt in self.prompts:
            tokens(prompt)
        import torch
        if not torch.version.hip or torch.version.cuda is not None or not torch.cuda.is_available():
            raise ValueError("ROCm backend required")
        if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx1151":
            raise ValueError("gfx1151 required")
        if self.name == "vLLM":
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.v1.engine.async_llm import AsyncLLM
            args = AsyncEngineArgs(model=config["model"], dtype="bfloat16", kv_cache_dtype="auto",
                                   max_model_len=2048, max_num_seqs=4, enable_prefix_caching=False,
                                   enforce_eager=False, kv_cache_memory_bytes=1207959552,
                                   trust_remote_code=False)
            self.engine = AsyncLLM.from_engine_args(args)
            actual = self.engine.vllm_config
            if (str(actual.model_config.dtype) != "torch.bfloat16" or actual.model_config.enforce_eager or
                    actual.model_config.max_model_len != 2048 or actual.scheduler_config.max_num_seqs != 4 or
                    actual.cache_config.enable_prefix_caching or actual.cache_config.cache_dtype != "auto" or
                    actual.cache_config.kv_cache_memory_bytes != 1207959552):
                raise ValueError("vLLM resolved configuration mismatch")
            self.info = dict(model_dtype=str(actual.model_config.dtype), kv_cache_dtype=actual.cache_config.cache_dtype,
                             max_model_len=actual.model_config.max_model_len,
                             kv_cache_memory_bytes=actual.cache_config.kv_cache_memory_bytes,
                             enforce_eager=actual.model_config.enforce_eager)
        elif self.name == "patched SGLang":
            from sglang import Engine
            self.engine = Engine(model_path=config["model"], dtype="bfloat16", kv_cache_dtype="bf16",
                                 context_length=2048, max_running_requests=4, max_total_tokens=8192,
                                 disable_radix_cache=True, attention_backend="triton", disable_cuda_graph=False,
                                 trust_remote_code=False)
            # Engine creates its loop when constructed without a running loop.
            # Server-info starts its receiver there. Reuse it for every corpus.
            self.loop = self.engine.loop
            self.info = self.engine.get_server_info()
            for key, value in dict(dtype="bfloat16", kv_cache_dtype="bf16", context_length=2048,
                                   max_running_requests=4, disable_radix_cache=True,
                                   attention_backend="triton", disable_cuda_graph=False).items():
                if self.info.get(key) != value:
                    raise ValueError("SGLang resolved configuration mismatch: " + key)
            if self.info.get("max_total_num_tokens", 0) < 8192:
                raise ValueError("SGLang resolved KV capacity below 8192")
        else:
            raise ValueError("unknown Python engine")
        self.run_number = 0

    async def one(self, index, run_number):
        prompt = self.prompts[index]
        started = time.monotonic()
        request_id = f"strix-{run_number}-{index}"
        if self.name == "vLLM":
            from vllm import SamplingParams
            from vllm.sampling_params import RequestOutputKind
            sampling = SamplingParams(temperature=0, top_p=1, max_tokens=128, ignore_eos=True,
                                      min_tokens=0, output_kind=RequestOutputKind.CUMULATIVE)
            previous = []
            final = None
            async for output in self.engine.generate({"prompt_token_ids": prompt}, sampling, request_id):
                if output.request_id != request_id or output.prompt_token_ids != prompt or len(output.outputs) != 1:
                    raise ValueError("vLLM request identity mismatch")
                sequence = output.outputs[0]
                ids = list(sequence.token_ids)
                if ids[:len(previous)] != previous or len(ids) < len(previous) or len(ids) > 128:
                    raise ValueError("invalid cumulative token stream")
                previous = ids
                if output.finished:
                    if final is not None:
                        raise ValueError("duplicate final output")
                    final = sequence
            if final is None:
                raise ValueError("vLLM missing final output")
            ids, finish = list(final.token_ids), final.finish_reason
        else:
            output = await self.engine.async_generate(input_ids=prompt, rid=request_id,
                sampling_params=dict(temperature=0, top_p=1, max_new_tokens=128, ignore_eos=True), stream=False)
            ids = output["output_ids"]
            meta = output["meta_info"]
            finish = meta["finish_reason"]["type"]
            if meta.get("id") != request_id or meta["prompt_tokens"] != len(prompt) or meta["completion_tokens"] != len(ids):
                raise ValueError("SGLang request identity mismatch")
            if finish == "length":
                finish = "length"
        return dict(index=index, prompt_ids=prompt, output_ids=ids, status="ok", finish_reason=finish,
                    dispatched=started, completed=time.monotonic())

    async def corpus(self, concurrency):
        results = [None] * 6
        next_index = 0
        async def worker():
            nonlocal next_index
            while next_index < 6:
                index = next_index
                next_index += 1
                results[index] = await self.one(index, self.run_number)
        started = time.monotonic()
        await asyncio.gather(*(worker() for _ in range(concurrency)))
        return dict(started=started, completed=time.monotonic(), requests=results)

    def run(self, command):
        concurrency = command["concurrency"]
        if type(concurrency) is not int or concurrency not in (1, 4):
            raise ValueError("concurrency must be 1 or 4")
        self.run_number += 1
        return self.loop.run_until_complete(self.corpus(concurrency))

    def close(self):
        self.engine.shutdown()
        self.loop.close()


def main():
    # Keep library prints off the protocol descriptor, including native writes.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    engine = None
    identifier = 0
    try:
        for line in sys.stdin:
            command = parse(line)
            if type(command.get("schema")) is not int or command["schema"] != 1 or type(command.get("id")) is not int or command["id"] != identifier + 1:
                raise ValueError("command identity mismatch")
            identifier = command["id"]
            action = command["command"]
            if action == "configure" and engine is None:
                engine = PythonEngine(command)
                result = dict(prompt_ids=engine.prompts, requested=RESOLVED, runtime_info=engine.info)
            elif action == "run" and engine is not None:
                if command["phase"] not in ("qualification", "warmup", "measured") or type(command["repetition"]) is not int:
                    raise ValueError("invalid run command")
                result = engine.run(command)
            elif action == "shutdown" and engine is not None:
                engine.close()
                engine = None
                protocol.write(json.dumps(dict(schema=1, id=identifier, status="ok")) + "\n")
                return 0
            else:
                raise ValueError("out-of-order command")
            protocol.write(json.dumps(dict(result, schema=1, id=identifier, status="ok"), default=str) + "\n")
        raise ValueError("protocol EOF before shutdown")
    except Exception as error:
        protocol.write(json.dumps(dict(schema=1, id=identifier, status="error", error=str(error))) + "\n")
        return 1
    finally:
        if engine is not None:
            engine.close()


if __name__ == "__main__":
    sys.exit(main())
