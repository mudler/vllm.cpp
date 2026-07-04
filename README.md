# vllm.cpp

A 1:1 port of [vLLM](https://github.com/vllm-project/vllm) to pure C++ —
no Python, no PyTorch at runtime. Same architecture, same algorithms, same
serving surface; usable as a library (llama.cpp-style) with an example CLI
and OpenAI-compatible server.

> ⚠️ **Pre-release, under heavy development.** Nothing below is usable yet;
> the tables track real, tested support and are kept up to date as work
> lands (see `.agents/workflow.md`).

## Supported model architectures

| Architecture | Families | Safetensors | GGUF | Status |
|---|---|---|---|---|
| Qwen3.5/3.6 hybrid (GDN + gated attention, MoE + dense) | Qwen3.6-35B-A3B, Qwen3.6-27B | 🚧 forward parity (M0.9, greedy-verified on GB10) | 🚧 load path done (k-quant dequant + qwen35moe loader; real-file parity dgx-pending) | 🚧 in development (M0–M3; engine + OpenAI server + tools + grammars on CPU) |
| Qwen3 / Qwen2 dense | Qwen3-32B, Qwen3-0.6B, … | — | — | 🗓 planned (post-MVP T1) |
| Llama-family dense | Llama 3.x, Mistral | — | — | 🗓 planned (post-MVP T1) |
| MoE decoders | Mixtral, Qwen3-MoE | — | — | 🗓 planned (post-MVP T1) |

## Acceleration

| Backend | Hardware | Status |
|---|---|---|
| CUDA | NVIDIA (first target: GB10 / DGX Spark, sm_121) | 🚧 in development (M0.6: baseline ops parity-verified on GB10) |
| CPU | x86-64 reference (correctness/CI grade) | 🚧 in development (M0) |
| Metal | Apple Silicon (MLX vs native MSL under exploration) | 🗓 planned (post-MVP) |
| Vulkan | Portable GPU | 🗓 planned (post-MVP) |
| SYCL / XPU | Intel GPUs | 🗓 planned (post-MVP) |

## Quantization

| Format | Status |
|---|---|
| NVFP4 (W4A4, Blackwell) | 🚧 in development (MVP gate) |
| GGUF quants (Q4_K/Q5_K/Q6_K/Q3_K, Q8_0, F32) | 🚧 dequant + qwen35moe load path done on CPU (real-file greedy parity dgx-pending); IQ2_S/IQ4_XS i-quants pending |
| fp8, MXFP4 | 🗓 planned (post-MVP T1) |

Legend: ✅ supported & parity-tested · 🚧 in development · 🗓 planned.

## Project record

Development is fully documented in [AGENTS.md](AGENTS.md) (canonical index):
gates, porting inventory vs upstream vLLM, parity ledger, roadmap, and the
upstream sync protocol.
