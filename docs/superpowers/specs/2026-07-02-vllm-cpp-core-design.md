# vllm.cpp — vLLM V1 Engine, Ported to Pure C++ (Core Design)

**Date:** 2026-07-02 (rev 2 — gates redefined by user)
**Status:** Draft — awaiting user review
**Upstream:** `/home/mudler/_git/vllm` @ `e24d1b24` (2026-07-02, master)
**Companion:** `.agents/porting-inventory.md` (full feature inventory, tiers T0–T3),
`AGENTS.md` (canonical state + discipline)

## 1. Goal & gates

A 1:1 port of vLLM to pure C++ — no Python/PyTorch at build or run time —
structured so any future upstream PR can be ported mechanically (mirrored
paths, class names, algorithms; see AGENTS.md discipline).

MVP gates (from AGENTS.md, authoritative): (1) Qwen3.6-35B-A3B-NVFP4 +
Qwen3.6-27B-NVFP4 served on DGX Spark (GB10, sm_121) at vLLM-parity prefill
and decode throughput under large concurrency; (2) the same models load from
GGUF; (3) llama.cpp-style library packaging (C API + example CLI/server);
(4) tool calling + grammars in the MVP feature set; (5) e2e test suites.

ggml is a **design reference, not a dependency**: its static-graph execution
model is architecturally at odds with vLLM's persistent-batch, paged-KV
design; building on it would forfeit the performance we're porting for.

### Non-goals for MVP (tracked as T1–T3 in the inventory)

Multi-GPU (TP/PP/DP/EP), spec decode (incl. Qwen3.6 MTP — first post-gate
perf item), LoRA, multimodal, KV offload/connectors, the long tail of model
architectures and quant methods, torch.compile-equivalent graph compilation.

## 2. Decisions log

- **D1 — No PyTorch** (user).
- **D2 — No ggml dependency; ggml as example** (user).
- **D3 — CUDA at step 1** on `dgx.casa` (GB10, sm_121); development over SSH
  (user). A scalar/SIMD CPU reference backend exists from day one, but only
  correctness-grade: it runs CI and op-parity tests on non-GPU machines.
- **D4 — Gate models define T0**: Qwen3.6 family ⇒ hybrid GDN + gated
  full-attention + 256-expert MoE + NVFP4 are core, not extensions (user).
- **D5 — GGUF is a gate** alongside safetensors; includes NVFP4 GGUF
  extension types from the APEX/killgate tooling (user).
- **D6 — Library-first** like llama.cpp: `libvllm` + stable C API +
  example CLI/openai-server binaries (user).
- **D7 — Tool calling + grammars in MVP** (user). Grammars via vendored
  xgrammar C++ core (upstream's default backend, pure C++), plus GBNF input
  format as an extension.
- **D8 — Track upstream master**; sync point recorded in AGENTS.md, ported
  files carry upstream path + commit headers (user: "port every PR").
- **D9 — Port Model Runner V2** (`vllm/v1/worker/gpu/`), not the legacy
  runner: upstream converges on it; PR-portability demands we sit where
  development happens.
- **D10 — Multi-backend by vLLM's own seams** (user): the architecture must
  accommodate Apple Metal (MLX/ANE explorations), Vulkan, and Intel XPU
  without touching engine code — portability lives exclusively in the
  mirrored `platforms/` + attention-backend registries + vt:: op tables.
  Binding vt:: interface requirements and per-platform strategy:
  `.agents/backends.md`. Implementations are post-MVP; the NVIDIA gate is
  not delayed.

## 3. Approaches considered (compute layer)

**A. Build on ggml** — rejected (D2): graph-rebuild-per-step vs persistent
batch, no paged-KV-shaped memory model, scheduler mismatch.

**B. Own minimal eager tensor runtime (`vt::`) + kernels ported from vLLM's
own sources (chosen).** vLLM's kernel semantics live in three portable
places: `csrc/cpu/` SIMD templates (CPU reference), the FLA Triton kernels
(GDN — port semantics to CUDA), and vLLM-history CUDA kernels + cuBLASLt for
dense/paged work. mudler's killgate/phase llama.cpp patches provide proven
GB10 techniques for NVFP4 W4A4 MMA and GDN. Vendor a specific
CUTLASS/FlashInfer kernel only when a benchmark proves we can't match it.

**C. Assemble from heavyweight deps (full CUTLASS/FlashInfer/TRT-LLM stack)**
— rejected as default posture: build weight, JIT-Python entanglement
(FlashInfer), and it inverts the "we own the code" premise. Selective
vendoring stays available inside B.

## 4. Architecture

Source layout mirrors upstream 1:1 (discipline in AGENTS.md). The only
net-new subsystem is `vt/` (compute), which replaces torch/Triton/compile:

```
vllm.cpp/
├── include/vllm.h              # stable C API (llama.cpp-style, purego-friendly)
├── include/vllm/*.hpp          # C++ API (LLM, AsyncLLM equivalents)
├── src/vt/                     # tensor runtime: Tensor{dtype,shape,strides,device,ptr},
│   ├── alloc.{h,cpp}           #   step arena + persistent buffers (graph-replay friendly)
│   ├── ops.h                   #   explicit-output ops, device-dispatched
│   ├── cuda/                   #   PRIMARY: GB10 sm_121 kernels (see §5)
│   └── cpu/                    #   reference kernels (CI/parity; csrc/cpu-derived)
├── src/vllm/                   # mirrors upstream python paths, e.g.:
│   ├── v1/core/sched/scheduler.{h,cpp}      ← vllm/v1/core/sched/scheduler.py
│   ├── v1/core/{block_pool,kv_cache_manager,kv_cache_coordinator,...}.{h,cpp}
│   ├── v1/worker/gpu/{model_runner,input_batch,block_table,...}.{h,cpp}   (MRV2)
│   ├── v1/attention/backends/{gdn_attn,paged_attn,...}.{h,cpp}
│   ├── v1/sample/…, v1/engine/…, v1/structured_output/…
│   ├── model_executor/models/{registry,qwen3_5,qwen3,llama,...}.{h,cpp}
│   ├── model_executor/layers/…             (linear/norms/rope/fused_moe/…)
│   ├── model_loader/{default_loader,gguf_loader,weight_utils}.{h,cpp}
│   ├── config/…                            (structs mirroring config dataclasses)
│   └── entrypoints/openai/…                (server lib used by examples/server)
├── examples/{cli,server,bench}/
├── tools/parity/               # Python dump scripts run against upstream (test-time only)
├── tests/                      # op-parity, engine-behavioral, model-parity, server-e2e
└── third_party/                # vendored: cpp-httplib, nlohmann/json, xgrammar core
```

Engine mechanics (ported semantics, per the upstream map): unified
token-budget scheduler with chunked prefill and preemption; BlockPool with
hash-chained prefix caching; **hybrid KV cache coordinator** with two groups
for the gate models (paged full-attn blocks + GDN conv/temporal state);
persistent InputBatch with `SchedulerOutput` diffs; split execute/sample;
sampler pipeline in upstream order; grammar bitmask via structured-output
manager; EngineCore busy loop on its own thread behind in-process queues;
OpenAI server with SSE streaming, tool-call parsing, and vLLM metric names.

## 5. The performance plan (gate #1)

GB10 is unified-memory, bandwidth-bound. Parity levers, in dependency order:

1. **NVFP4 W4A4 kernels** for MoE grouped-GEMM and dense GEMM (weights stay
   fp4 in memory — bandwidth is the currency; killgate patch 0034 is prior
   art). cuBLASLt for bf16/fp8 paths.
2. **GDN kernels**: chunked-scan prefill + fused sigmoid-gating recurrence
   decode, ported from FLA Triton semantics to CUDA; fused post-conv prep +
   causal conv1d.
3. **Paged attention** for the 1-in-4 full-attn layers (GQA 16/2, partial
   RoPE): FA2-style varlen prefill + paged decode tuned for sm_121.
4. **CUDA graphs** for decode steps (MoE decode is launch-bound; upstream
   default FULL_AND_PIECEWISE tells us graphs are non-optional) — our own
   capture/replay over persistent buffers; the vt arena design exists for
   this.
5. **Fused small ops**: qk-norm+RoPE+gate (upstream has this exact fusion),
   rmsnorm+residual, MoE routing top-k.
6. Async scheduling / batch-queue pipelining if CPU-side becomes the
   bottleneck at high concurrency (T1 item, promotable).

Benchmark harness (`examples/bench`, `vllm bench serve` semantics) lands
*before* kernel tuning; every optimization is measured against the vLLM
baseline on the same box (vLLM install on dgx.casa is a recorded TODO).

## 6. Error handling

Request-level errors rejected at InputProcessor (OpenAI-shaped 4xx JSON,
matching upstream error shapes). Engine invariants assert in debug; corrupted
engine state is fatal-loud (upstream behavior). KV exhaustion is preemption,
not error. Server: 503 on engine death, graceful drain on shutdown. CUDA
errors: fail the affected requests, mark engine dead if the context is
poisoned (matching upstream EngineDeadError semantics).

## 7. Testing (gate #5)

The five suites from `.agents/porting-inventory.md` §10: op parity (upstream
golden dumps; CPU + CUDA), engine behavioral (ported `tests/v1/core/`
semantics incl. hybrid-group allocation), model parity (logits + greedy
token-for-token on Qwen3-0.6B fast / gate models nightly; safetensors, NVFP4
and GGUF paths), server e2e conformance (incl. tool-call streaming and
grammar constraint tests), and the gate benchmark with per-commit regression
tracking. CI (GitHub Actions, CPU): build + unit + behavioral + 0.6B smoke.
Nightly (dgx.casa): CUDA parity + gate benchmarks.

## 8. Milestones

- **M0 — Model correctness on CUDA.** vt runtime (CUDA + CPU ref), safetensors
  + GGUF containers, tokenizer (HF json + GGUF vocab), Qwen3.6 forward pass
  eager bf16 (NVFP4 dequantized on load — fits unified memory), single-request
  greedy decode parity vs upstream. Everything mirrored-structure from day one.
- **M1 — The engine.** Hybrid KV cache + BlockPool + prefix caching +
  scheduler + persistent InputBatch + sampler + EngineCore; offline C++/C API;
  behavioral suites green; concurrency scales.
- **M2 — Parity performance.** NVFP4 W4A4 MoE + GDN CUDA kernels + tuned paged
  attention + CUDA graphs + bench harness; **gate #1 measured and met**.
- **M3 — Serving MVP.** OpenAI server (streaming, tools, grammars), metrics,
  CLI, library packaging polish, README (house style), docs. Gates #2–#5
  validated end-to-end.
- **Post-MVP:** T1 tier (dense model families, MTP spec decode, fp8, sliding
  window, priority scheduling, structured-output breadth), then T2.

## 9. Risks

- **GDN + NVFP4 kernel effort** is the long pole; mitigated by killgate/FLA
  prior art and by bench-first development. Fallback: vendor FlashInfer's
  Blackwell GDN/attention kernels behind vt ops (deviation §9 of inventory).
- **Parity target moves** (vLLM improves on GB10): gate measured against the
  vLLM version pinned at AGENTS.md sync point; re-baseline consciously.
- **Tokenizer/chat-template scope**: minja-style Jinja subset for Qwen/Llama
  templates; hard-fail with clear errors on unsupported constructs.
- **Upstream drift**: mitigated by the sync-point discipline + mirrored
  structure (AGENTS.md).
- **xgrammar vendoring**: C++ core but sizeable; isolate behind the
  structured-output manager interface so it stays swappable.
