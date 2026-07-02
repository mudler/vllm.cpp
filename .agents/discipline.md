# Design discipline (non-negotiable)

- **Mirror upstream structure.** C++ sources live at paths that mirror the
  Python: `vllm/v1/core/sched/scheduler.py` → `src/vllm/v1/core/sched/
  scheduler.{h,cpp}`. Same class names, same method names, same field names
  (C++-case-adjusted only where unavoidable). A vLLM developer must be able to
  navigate this repo blind; a PR touching `scheduler.py` maps to exactly one
  place here.
- **Port, don't reinvent.** Read the upstream implementation before writing
  any subsystem. Algorithms, edge cases, defaults, and even comments' intent
  come from upstream. When upstream has a design quirk, we keep it (and note
  why it exists) rather than "improving" it — divergence kills PR portability.
- **Every ported file carries a header comment**: upstream path + upstream
  commit hash it was ported from. When re-syncing with upstream, diff that
  file against its recorded commit.
- **Deviations are exceptional and recorded** in `docs/porting-inventory.md`
  §9 (currently: compute layer replaces torch/Triton with `vt::`; in-process
  queues replace ZMQ; cpp-httplib replaces FastAPI; C-ABI plugins replace
  Python plugins; GGUF promoted to first-class).
- **No PyTorch, no ggml dependencies.** Header-only third-party deps preferred
  (cpp-httplib, nlohmann/json); xgrammar C++ core vendored for grammars;
  CUDA + cuBLAS for GPU.
- **Parity-first testing.** Nothing is "done" until it matches upstream:
  kernels vs golden dumps, engine behavior vs ported test semantics, models
  vs logits/greedy-decode, server vs OpenAI conformance. Upstream vLLM
  (Python) is a test-time oracle only — never a runtime dependency.
- **We port Model Runner V2** (`vllm/v1/worker/gpu/`), not the legacy runner —
  see [vllm-v1-v2.md](vllm-v1-v2.md).
