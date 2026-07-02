# The gates (definition of MVP success)

1. **Throughput parity vs vLLM** on `dgx.casa` (DGX Spark, GB10): serve
   **Qwen3.6-35B-A3B (NVFP4)** and **Qwen3.6-27B (NVFP4)** with prefill AND
   decode throughput matching vLLM at large concurrency (request-rate sweeps,
   measured with our `bench serve` equivalent, same box, same models).
2. **GGUF reading**: the same models load and serve from GGUF files (including
   NVFP4 GGUF extension types from the APEX/killgate tooling), not just
   safetensors.
3. **Library-first packaging** (llama.cpp-style): `libvllm` + stable C API
   (`include/vllm.h`, cgo/purego-friendly for LocalAI) + example CLI and
   OpenAI-server binaries.
4. **MVP features**: OpenAI completions/chat, **both streaming (SSE) and
   non-streaming** responses, **tool calling** (Qwen + Hermes parsers, auto
   tool choice), **grammars/structured outputs** (xgrammar C++ core: JSON
   schema/regex/choice/EBNF + GBNF extension), core sampling surface,
   Prometheus metrics with vLLM's metric names.
5. **E2E test suites**: op parity vs upstream dumps, engine behavioral tests
   (ported from upstream `tests/v1/core/` semantics), model parity
   (logits + greedy token-for-token), server conformance, gate benchmark
   regression tracking. CI-runnable on CPU (0.6B model); nightly on dgx.casa.
