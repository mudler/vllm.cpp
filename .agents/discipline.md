# Design discipline (non-negotiable)

- **Mirror upstream structure.** C++ sources live at paths that mirror the
  Python: `vllm/v1/core/sched/scheduler.py` → `src/vllm/v1/core/sched/
  scheduler.{h,cpp}`. Same class names, same method names, same field names
  (C++-case-adjusted only where unavoidable). A vLLM developer must be able to
  navigate this repo blind; a PR touching `scheduler.py` maps to exactly one
  place here.
- **Port, don't reinvent — GROUND EVERY IMPLEMENTATION IN UPSTREAM SOURCE.**
  Read the upstream implementation before writing *any* subsystem OR kernel, and
  mirror it. Algorithms, edge cases, defaults, PTX/intrinsics, tile/config
  values, and even comments' intent come from upstream. When upstream has a
  design quirk, we keep it (and note why it exists) rather than "improving" it —
  divergence kills PR portability.
  - **"Upstream" includes the whole chain, not just the vLLM repo.** The kernels
    vLLM actually runs live in its DEPS — **flashinfer** (CuTe-DSL fp4/fp8 GEMMs,
    fused norm+quant, `blackwell_sm12x`), **cutlass**, **cuBLASLt** (`nvjet`),
    **DeepGEMM**, **TensorRT-LLM**, torch/Inductor codegen. Ground in *that*
    source (read it, and for JIT/compiled code dump the generated kernel), then
    port the same structure. See [parity-lever-protocol.md § Verify the whole
    chain] and § Trace the execution.
  - **Writing from scratch is EXCEPTIONAL — only when strictly necessary**, i.e.
    no upstream (vLLM *or* dep) implementation exists to mirror, or the upstream
    form is genuinely un-portable to eager-C++ (proven, not assumed — cite the
    dep `file:line` / dumped kernel showing why). When you must, record the
    reason in `.agents/porting-inventory.md` §9. A hand-rolled reinvention of
    something upstream already does is a defect, not an optimization.
  - **Every new kernel/implementation cites what it was ported FROM** — the
    upstream `file:line` (vLLM or dep) in the code/commit. "Grounded in upstream"
    means a reviewer can open that source and see we mirrored it. (User,
    2026-07-07: *"every implementation you do should be grounded by upstream
    source code, you should not re-implement things from scratch."*)
- **Every ported file carries a header comment**: upstream path + upstream
  commit hash it was ported from. When re-syncing with upstream, diff that
  file against its recorded commit.
- **Deviations are exceptional and recorded** in `.agents/porting-inventory.md`
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
