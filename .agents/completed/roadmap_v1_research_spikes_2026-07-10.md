# Completed roadmap v1 research spikes B1-B7

**Closed:** 2026-07-10. This is the frozen execution record for the B-series
parallel research block. Live work and decisions are summarized in
`../roadmap_v1.md` and the area matrices; do not use this archive as the current
status surface.

| # | Track | Question | Closed finding |
|---|---|---|---|
| B1 | Apple Metal / MLX | vllm-metal vs MLX C++ behind `vt::` vs native MSL | MLX selected as the first Apple kernel layer, with custom MSL/MLX primitives for paged gaps. M4/16 GB host is adequate for ops/small models, not 27B/35B scale. Live: `../backend-matrix.md`. |
| B2 | Architectures/vendors | CUDA targets; Triton-to-AMD; SYCL vs Vulkan | Corrected by the later exhaustive sweep: pinned vLLM has 13 numeric CUDA targets while vllm.cpp gates only sm121a. Every other target and ROCm needs explicit build/dispatch/AOT/correctness/trace/perf work. Vulkan remains before SYCL. Live: `../backend-matrix.md`, `../kernel-matrix.md`. |
| B3 | SGLang comparison | portable engine/kernel wins | Async/overlap scheduling is a vLLM default at the pin and remains a mirror obligation. RadixAttention is not adopted as a vLLM-compatibility feature. SGLang is now a low-concurrency benchmark reference. |
| B4 | llama.cpp CPU | vendor only on measured win | vllm.cpp CPU is single-thread scalar. Threadpool precedes compute-in-quant; direct quantized compute is the structural GGUF speed requirement. Live: `../quantization-matrix.md`, `../backend-matrix.md`. |
| B5 | Speculative decode | MTP and DFlash | Both gate checkpoints ship MTP heads; both are GDN hybrids. Route is MTP k=1 including GDN state, then DFlash. Live specs: `../specs/mtp-spec-decode.md`, `../specs/dflash-spec-decode.md`. |
| B6 | Feature parity | one-by-one vLLM surface | Initial feature matrix exposed metrics, async scheduling and serving logprob/control gaps. The record has since expanded into area matrices under the tabular policy. |
| B7 | Multimodal/tools | image/audio/video and tool calls | Gate checkpoints contain full vision towers that are currently omitted. Qwen3-Coder XML/tool-template and reasoning/Jinja gaps remain; parallel tool-call core is present. Live scope: `../specs/mm-tools-scoping-2026-07-10.md`. |

## Source reports

- `../specs/expansion-map-2026-07-10.md`
- `../specs/spec-decode-scoping-2026-07-10.md`
- `../specs/mm-tools-scoping-2026-07-10.md`
- `../feature-matrix.md` at the closing era, superseded in place by the live
  tabular matrices

## Closure rule applied

All seven research questions reported. Their ongoing implementation rows were
carried into roadmap C/D and the area matrices before this block moved under
`completed/`. No open implementation work is represented as completed here.
