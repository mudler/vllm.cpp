# Mission

1:1 port of vLLM to pure C++ — no Python, no PyTorch at build or run time.
Loyal to the upstream codebase: same architecture, same class/file names, same
algorithms, same config/metric/API surfaces, so that **any future upstream vLLM
PR can be ported here mechanically**.

> **Sanctioned exception (User, 2026-07-09; narrowed 2026-07-10):** a single,
> bounded, gated (`VLLM_CPP_TRITON`, default OFF) Triton AOT fast-path is
> allowed for kernels where portable C++ is *measured-exhausted* against vLLM's
> compiler codegen (the GDN chunk kernels). The generated artifacts (cubins
> embedded in C launchers) are **VENDORED per-arch in-repo**
> (`src/vt/cuda/triton_aot_vendored/`), so BUILDING needs no Python/Triton
> either — only a C compiler; Python+Triton is a **MAINTAINER-only,
> regen-time** dependency (`VLLM_CPP_TRITON_REGEN=ON`,
> `scripts/regen-triton-aot.sh`). The RUNTIME stays Python/PyTorch/Triton-free
> (cubins via the CUDA driver API), the CPU reference + portable hand-C++ CUDA
> fallback are preserved, and every other backend still ports from
> `vt::`+CPU-ref. See discipline.md ("SANCTIONED EXCEPTION") and
> porting-inventory.md §9.

ggml is a design reference (minimal deps, explicit kernels), **not** a
dependency — its static-graph execution model conflicts with vLLM's
persistent-batch, paged-KV design.

Packaging is llama.cpp-style: usable as a library (`libvllm` + stable C API),
with example CLI / OpenAI-server binaries shipped from this repo.
