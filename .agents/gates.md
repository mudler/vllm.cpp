# The gates (definition of MVP success)

> **MVP FOCUS (user, 2026-07-04): get BOTH gate models (35B MoE/W4A16 + 27B
> dense/W4A4) running at vLLM's throughput — that IS the MVP.** Gate #1 on both
> models is the bar. Everything below (#2–#5) supports it. Explicitly POST-MVP,
> in order: (a) more model architectures; (b) more/faster CUDA kernels; (c)
> **lift the design so we can import kernels directly from vLLM** (drop-in csrc
> alignment → reduce our maintenance burden); (d) a systematic **sweep of vLLM
> features / architectures / accelerators we lack** to decide follow-up. Do NOT
> start post-MVP work until both models hit parity.

1. **Throughput parity vs vLLM** on `dgx.casa` (DGX Spark, GB10): serve
   **Qwen3.6-35B-A3B (NVFP4)** and **Qwen3.6-27B (NVFP4)** with prefill AND
   decode throughput matching vLLM at large concurrency (request-rate sweeps,
   measured with our `bench serve` equivalent, same box, same models).
   Note: the two gate models are DISTINCT quant/arch paths — **35B** is
   **MoE** (`Qwen3_5MoeForConditionalGeneration`, modelopt **W4A16**), **27B**
   is **dense** (`Qwen3_5ForConditionalGeneration`, compressed-tensors
   **W4A4**). Both remain throughput-parity targets, but they exercise
   different architectures and NVFP4 quant schemes.
2. **GGUF reading**: the same models load and serve from GGUF files (including
   NVFP4 GGUF extension types from the APEX/killgate tooling), not just
   safetensors.
3. **Library-first packaging** (llama.cpp-style): `libvllm` + stable C API
   (`include/vllm.h`, cgo/purego-friendly for LocalAI) + example CLI and
   OpenAI-server binaries.
4. **MVP features**: OpenAI completions/chat, **both streaming (SSE) and
   non-streaming** responses, **tool calling** (Qwen + Hermes parsers, auto
   tool choice), **grammars/structured outputs** (xgrammar C++ core: JSON
   schema/regex/choice/EBNF + GBNF extension), core sampling surface.
   (Prometheus `/metrics` endpoint DROPPED from MVP — user decision 2026-07-04,
   moved to post-MVP.)
5. **E2E test suites**: op parity vs upstream dumps, engine behavioral tests
   (ported from upstream `tests/v1/core/` semantics), model parity
   (logits + greedy token-for-token), server conformance, gate benchmark
   regression tracking. CI-runnable on CPU (0.6B model); nightly on dgx.casa.

## PROTOCOL DIRECTIVE — always compare vs vLLM (the oracle), on the SAME workload

vLLM is the parity oracle for BOTH correctness and performance. Every change
that could affect either MUST be compared against vLLM, apples-to-apples, and
BOTH numbers recorded in the [parity ledger](parity-ledger.md):

- **Correctness:** op dumps + model logits/greedy vs the pinned pip-vLLM oracle
  (`~/venvs/vllm-oracle` on dgx.casa, forward-math-identical to the pin). A new
  op/model change is not "done" until it matches the vLLM oracle (or the
  deviation is recorded as accepted with the reason).
- **Performance (gate #1):** every perf/kernel change is measured with our
  `vllm-bench` AND the vLLM baseline (`vllm bench throughput`) on the **IDENTICAL
  workload** (same model, input-len, output-len, num-prompts, concurrency, same
  box) — never quote our number without the vLLM number on the same config, and
  never re-base the bench config without re-running vLLM on it. Record the ratio
  (ours / vLLM) so gate-#1 progress is always a parity delta, not an absolute.
- Rationale: the prime directive is 1:1 vLLM parity; "faster than before" is
  meaningless without "how far from vLLM." A perf win that doesn't move the vLLM
  ratio (e.g. a banked compute win under a host-bound profile) must say so.
