# Mission

1:1 port of vLLM to pure C++ — no Python, no PyTorch at build or run time.
Loyal to the upstream codebase: same architecture, same class/file names, same
algorithms, same config/metric/API surfaces, so that **any future upstream vLLM
PR can be ported here mechanically**.

ggml is a design reference (minimal deps, explicit kernels), **not** a
dependency — its static-graph execution model conflicts with vLLM's
persistent-batch, paged-KV design.

Packaging is llama.cpp-style: usable as a library (`libvllm` + stable C API),
with example CLI / OpenAI-server binaries shipped from this repo.
