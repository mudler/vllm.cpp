#!/usr/bin/env python3
# CPU-only wave-boundary scheduler-composition oracle for the ITL tail-stall
# discriminator (row SERVE-GATE-ONLINE, spec
# .agents/specs/tail-stall-analysis-2026-07-16.md).
#
# Drives the REAL vllm.v1.core.sched Scheduler (sync, depth-1) and AsyncScheduler
# (depth-2) at the online-gate binding shape (max_num_seqs=32,
# max_num_batched_tokens=2048, chunked prefill on, no prefix caching) and records
# the per-step scheduling composition. No GPU, no model weights. It is the ORACLE
# for tests/vllm/v1/test_scheduler_wave.cpp: the C++ test drives OUR schedulers
# through the byte-identical script and asserts the same composition.
#
# Two modes:
#   script     : the analysis scenario -- N running mid-decode, 2 finish the SAME
#                step, 2 fresh 1024-token prefills waiting. Compact per-step table;
#                this is the golden the C++ test pins.
#   closedloop : a full closed loop (all clients re-issue on completion) at the
#                binding shape; shows the emergent regime (both arms self-stagger
#                to ~2048-token admission steps -- the attractor evidence).
#
# Requires the pinned vLLM oracle (~/venvs/vllm-oracle == vllm 0.25.0 == 702f481;
# the sched loop is drift-free vs e24d1b24 for this scenario -- async_scheduler.py
# and request_queue.py are byte-identical and scheduler.py differs only in
# spec-decode / structured-output / KV-connector paths, all inert here). Run on a
# host with the oracle venv, e.g.:
#   ~/venvs/vllm-oracle/bin/python tools/bench/scheduler_wave_diff.py \
#       script --out tests/fixtures/scheduler_wave/wave_script_oracle.json
#
# vllm/torch are imported lazily (inside build_scheduler) so the module stays
# importable and py_compile-clean without the oracle installed.
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque

BLOCK_SIZE = 16
BUDGET = 2048
MAX_NUM_SEQS = 32
FRESH_PROMPT_LEN = 1024  # one prompt fills half the 2048 budget (the knife-edge)
FILL_PROMPT_LEN = 4      # short prompt for the running set
NUM_BLOCKS = 200000
MAX_MODEL_LEN = 2048

_NONE_HASH_DONE = False


def build_scheduler(async_sched: bool):
    """Construct the REAL vLLM Scheduler/AsyncScheduler at the binding shape."""
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
    import torch
    from vllm.config import (
        CacheConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        VllmConfig,
    )
    from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
    )
    from vllm.v1.structured_output import StructuredOutputManager

    mc = ModelConfig(model="facebook/opt-125m", trust_remote_code=True,
                     dtype="float16", seed=42, skip_tokenizer_init=True)
    sc = SchedulerConfig(
        max_num_seqs=MAX_NUM_SEQS, max_num_batched_tokens=BUDGET,
        max_model_len=MAX_MODEL_LEN, long_prefill_token_threshold=0,
        enable_chunked_prefill=True, async_scheduling=async_sched,
        is_encoder_decoder=mc.is_encoder_decoder, watermark=0.0)
    cc = CacheConfig(block_size=BLOCK_SIZE, gpu_memory_utilization=0.9,
                     cache_dtype="auto", enable_prefix_caching=False)
    vc = VllmConfig(scheduler_config=sc, model_config=mc, cache_config=cc,
                    parallel_config=ParallelConfig(pipeline_parallel_size=1))
    kv = KVCacheConfig(
        num_blocks=NUM_BLOCKS, kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(
            ["layer"], FullAttentionSpec(block_size=BLOCK_SIZE, num_kv_heads=1,
                                         head_size=1, dtype=torch.float32))])
    cc.num_gpu_blocks = NUM_BLOCKS
    register_all_kvcache_specs(vc)
    cls = AsyncScheduler if async_sched else Scheduler
    s = cls(vllm_config=vc, kv_cache_config=kv, block_size=BLOCK_SIZE,
            log_stats=False, structured_output_manager=StructuredOutputManager(vc))
    s.use_v2_model_runner = True
    return s


class Harness:
    """Drives a real scheduler; records per-step composition."""

    def __init__(self, scheduler):
        global _NONE_HASH_DONE
        from vllm.utils.hashing import sha256
        from vllm.v1.core.kv_cache_utils import (get_request_block_hasher,
                                                 init_none_hash)
        if not _NONE_HASH_DONE:
            init_none_hash(sha256)
            _NONE_HASH_DONE = True
        self.s = scheduler
        self.block_hasher = get_request_block_hasher(BLOCK_SIZE, sha256)
        self.uid = 0
        self.recs = []
        self.step = 0

    def add(self, rid, prompt_len, max_tokens):
        from vllm.sampling_params import SamplingParams
        from vllm.v1.request import Request
        self.uid += 1
        params = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
        req = Request(request_id=rid,
                      prompt_token_ids=[1 + (self.uid % 40000)] * prompt_len,
                      sampling_params=params, pooling_params=None,
                      block_hasher=self.block_hasher)
        self.s.add_request(req)

    def schedule_and_snapshot(self):
        cb = {rid: r.num_computed_tokens for rid, r in self.s.requests.items()}
        pl = {rid: r.num_prompt_tokens for rid, r in self.s.requests.items()}
        so = self.s.schedule()
        sampling = [rid for rid in so.num_scheduled_tokens
                    if not self.s.requests[rid].is_prefill_chunk]
        pcounts, ptok, dtok, chunk, ndec = [], 0, 0, False, 0
        for rid, n in so.num_scheduled_tokens.items():
            if cb.get(rid, 0) < pl.get(rid, 0):
                ptok += n
                pcounts.append(n)
                if cb.get(rid, 0) + n < pl.get(rid, 0):
                    chunk = True
            else:
                dtok += n
                ndec += 1
        pcounts.sort(reverse=True)
        self.recs.append({
            "step": self.step, "total": so.total_num_scheduled_tokens,
            "prefill_tokens": ptok, "decode_tokens": dtok,
            "n_prefill": len(pcounts), "n_decode": ndec,
            "prefill_counts": pcounts, "chunked": chunk,
            "running": len(self.s.running)})
        self.step += 1
        return so, sampling

    def make_mro(self, so, sampling):
        from vllm.v1.outputs import ModelRunnerOutput
        ids = list(so.num_scheduled_tokens.keys())
        idx = {r: i for i, r in enumerate(ids)}
        samp = [[7] if (r in sampling and r in self.s.requests) else []
                for r in ids]
        return ModelRunnerOutput(req_ids=ids, req_id_to_index=idx,
                                 sampled_token_ids=samp)

    def process(self, so, sampling):
        before = set(so.num_scheduled_tokens.keys())
        self.s.update_from_output(so, self.make_mro(so, sampling))
        return [r for r in before if r not in self.s.requests]


def run_script(async_sched, N, inject_at=4, steps=14):
    """N fillers (short prompt); f0/f1 finish (max_tokens=3), the rest stay
    (max_tokens=60). Two fresh FRESH_PROMPT_LEN prefills W0/W1 are added right
    before the schedule() that becomes step `inject_at`."""
    depth = 2 if async_sched else 1
    hz = Harness(build_scheduler(async_sched))
    for i in range(N):
        hz.add(f"f{i}", FILL_PROMPT_LEN, 3 if i < 2 else 60)
    injected = [False]

    def maybe_inject():
        if not injected[0] and hz.step >= inject_at:
            hz.add("W0", FRESH_PROMPT_LEN, 200)
            hz.add("W1", FRESH_PROMPT_LEN, 200)
            injected[0] = True

    if depth == 1:
        for _ in range(steps):
            maybe_inject()
            so, sampling = hz.schedule_and_snapshot()
            if not so.num_scheduled_tokens:
                break
            hz.process(so, sampling)
    else:
        q = deque()
        for _ in range(depth):
            maybe_inject()
            q.append(hz.schedule_and_snapshot())
        for _ in range(steps):
            if not q:
                break
            so, sampling = q.popleft()
            hz.process(so, sampling)
            maybe_inject()
            nso, nsamp = hz.schedule_and_snapshot()
            if nso.num_scheduled_tokens:
                q.append((nso, nsamp))
    return hz.recs


def run_closedloop(async_sched, C, prompt_len=1024, max_tokens=128, steps=320):
    """Closed loop: C clients re-issue a fresh prompt_len prompt on completion."""
    depth = 2 if async_sched else 1
    hz = Harness(build_scheduler(async_sched))
    inflight = {}

    def submit(client):
        rid = f"c{client}_{hz.uid + 1}"
        hz.add(rid, prompt_len, max_tokens)
        inflight[rid] = client

    def process(so, sampling):
        for rid in hz.process(so, sampling):
            client = inflight.pop(rid, None)
            if client is not None:
                submit(client)

    for c in range(C):
        submit(c)
    if depth == 1:
        for _ in range(steps):
            so, sampling = hz.schedule_and_snapshot()
            if not so.num_scheduled_tokens:
                break
            process(so, sampling)
    else:
        q = deque()
        for _ in range(depth):
            q.append(hz.schedule_and_snapshot())
        for _ in range(steps):
            if not q:
                break
            so, sampling = q.popleft()
            process(so, sampling)
            nso, nsamp = hz.schedule_and_snapshot()
            if nso.num_scheduled_tokens:
                q.append((nso, nsamp))
    return hz.recs


def summarize_admit(recs):
    admit = [r for r in recs if r["prefill_tokens"] > 0]
    hist = {}
    for r in admit:
        k = r["prefill_tokens"]
        hist[k] = hist.get(k, 0) + 1
    return {"num_admission_steps": len(admit),
            "budget_filled_steps": sum(1 for r in admit if r["total"] == BUDGET),
            "any_chunked": any(r["chunked"] for r in recs),
            "hist_prefill_tokens": {str(k): v for k, v in sorted(hist.items())}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["script", "closedloop"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--closedloop-steps-c8", type=int, default=320)
    ap.add_argument("--closedloop-steps-c32", type=int, default=640)
    args = ap.parse_args()

    result = {"mode": args.mode, "shape": {
        "max_num_seqs": MAX_NUM_SEQS, "max_num_batched_tokens": BUDGET,
        "fresh_prompt_len": FRESH_PROMPT_LEN, "block_size": BLOCK_SIZE},
        "arms": {}}
    for cname, C in (("c8", 8), ("c32", 32)):
        for aname, is_async in (("sync", False), ("async", True)):
            if args.mode == "script":
                recs = run_script(is_async, C)
            else:
                steps = (args.closedloop_steps_c8 if cname == "c8"
                         else args.closedloop_steps_c32)
                recs = run_closedloop(is_async, C, steps=steps)
            key = f"{cname}_{aname}"
            result["arms"][key] = {"concurrency": C, "async": is_async,
                                   "driver_depth": 2 if is_async else 1,
                                   "records": recs,
                                   "summary": summarize_admit(recs)}
            sm = result["arms"][key]["summary"]
            print(f"{key}: {sm}", file=sys.stderr)

    with open(args.out, "w") as f:
        json.dump(result, f, indent=1, sort_keys=True)
    print("wrote", args.out, file=sys.stderr)


if __name__ == "__main__":
    main()
