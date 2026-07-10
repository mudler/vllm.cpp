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
  - **Kernel-DSL source (Triton / CuTe-DSL / Inductor-generated) is a 1:1 PORTING
    REFERENCE, NEVER a compile-target.** READ the Triton/CuTe-DSL kernel — it is
    the exact fusion spec (the WHAT *and* the HOW at algorithm level) — and hand-
    transcribe it into our **portable** C++/`vt::` (with a CPU reference), citing
    the source `file:line`. Do **NOT** AOT-compile Triton/CuTe-DSL to PTX/cubin or
    otherwise link vendor-*generated* kernels: that re-imports CUDA lock-in and
    breaks the multi-backend contract (the `vt::` op interface + CPU ref + portable
    structure that Vulkan / Metal / ROCm / XPU port FROM). CuTe-DSL is Python sugar
    over CuTe (the C++ CUTLASS library) → its kernels port to C++ CUTLASS/EVT
    near-mechanically; Triton (tile IR: block-pointers, `boundary_check`) ports to
    C++/CUDA with understanding (done: the GDN chunk). Reuse the fusion **PATTERN**
    (the finite chain set vLLM/Inductor fuse — see the parity-lever prefill scan),
    realized portably; never the vendor's generated binary. (User, 2026-07-08:
    *"get from triton/cute-dsl the base to build ours ported 1:1"*; *"triton/cute-dsl
    would make it less portable?"*)
  - **SANCTIONED EXCEPTION (User, 2026-07-09) — a CUDA-only Triton AOT fast-path for
    PROVEN codegen-bound kernels.** The rule above is the DEFAULT and holds wherever
    portable C++ can match the vendor. But for a kernel where portable hand-C++ is
    *measured-exhausted* — i.e. we have BENCHMARKS showing the residual gap is the
    vendor's *compiler* codegen (register allocation / tensor-core instruction
    scheduling), not the algorithm/structure/pipeline (which we ported) — a
    **CUDA-only Triton kernel AOT-compiled to cubin at BUILD time** is allowed,
    **provided the portable contract is fully preserved**: (1) it lives behind the
    `vt::` op interface; (2) the CPU reference AND a correct portable hand-C++ CUDA
    fallback remain, so Metal/Vulkan/ROCm/XPU still port FROM `vt::`+CPU-ref and CUDA
    without Triton still works; (3) it is a gated build dependency (`VLLM_CPP_TRITON`,
    default OFF) and the RUNTIME stays Triton/Python-free (cubin via the CUDA driver
    API); (4) it is token-exact vs the portable path and greedy-16/16-gated. This is a
    bounded per-backend accelerator, NOT "AOT Triton everywhere." Justification, now
    EVIDENCE-BACKED: the GDN linear-attention chunk kernels are ~1.9× slower than
    vLLM's Triton/FLA, and we PROVED the portable path is exhausted — register-tiling
    (delta_h −22%), blocked tensor-core inverse, bf16 I/O, and BOTH async-pipeline
    tiers (Rung-1 cp.async + Rung-2 TMA+mbarrier) all landed, and the delta_h kernel
    still sits on a compute floor ~1.9× off vLLM. The "no compile-target" rule was
    premised on portable being able to match the codegen; that premise is disproven
    for these kernels. Mirror-vLLM (the PRIME POLICY — vLLM hits its numbers WITH
    Triton) + the ≥1.0× MVP gate both require it. Recorded as deviation in
    porting-inventory.md §9. Toolchain: branch `perf/triton-fastpath`
    (cmake/TritonAOT.cmake + triton.tools.compile/link), proven token-exact.
    NARROWED 2026-07-10: the generated artifacts are VENDORED per-arch in-repo
    (`src/vt/cuda/triton_aot_vendored/<arch>/` + MANIFEST), so condition (3)
    tightens — `VLLM_CPP_TRITON=ON` builds need ONLY a C compiler (no
    Python/Triton); Python+Triton is a MAINTAINER-only regen-time dependency
    (`VLLM_CPP_TRITON_REGEN=ON`, `scripts/regen-triton-aot.sh`), with
    configure-time staleness warnings when `triton_kernels/*.py` or the pins
    drift from the MANIFEST.
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
