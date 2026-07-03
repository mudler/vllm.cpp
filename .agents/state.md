# State log (append; newest last)

- **2026-07-02** — Project kicked off. Repo created (`mudler/vllm.cpp`),
  pushed to GitHub. vLLM reference mapped (engine core + full feature
  inventory via subagents). Gates defined (Qwen3.6 NVFP4 parity on GB10 +
  GGUF + library + tools/grammars + streaming/non-streaming + e2e suites).
  `.agents/porting-inventory.md` (then docs/) written with T0–T3 tiers. Core design doc
  rev 2 (CUDA-first on GB10, hybrid GDN+MoE+NVFP4 in T0, MRV2, library-first).
  Upstream synced to `e24d1b24`. AGENTS.md restructured as index over
  `.agents/*.md`. NOT STARTED: any implementation. NEXT: user reviews
  inventory + design doc → writing-plans for milestone M0.
- **2026-07-02 (later)** — Project made agent-maintainable: inventory moved to
  `.agents/porting-inventory.md` as a *living parity record* (inline status
  markers + upkeep rules), `parity-ledger.md` created (append-only: every
  change → what it does vs vLLM, upstream PR refs, verification),
  `roadmap.md` created (M0–M3 broken into PR-sized units with DoD),
  `workflow.md` rewritten as the agent session protocol. NEXT unchanged:
  user review → writing-plans for M0.1.
- **2026-07-02 (later)** — Upstream sync formalized as a protocol
  (`upstream-sync.md`): repo-wide PARITY PIN (`e24d1b24`), per-file pins in
  headers, and a repeatable 7-step sync cycle (enumerate → classify
  PORT-NOW/INVENTORY/IGNORE → report under `.agents/sync/` → port → re-verify
  → advance pin). Sync tooling + recurring cycle added to roadmap.
- **2026-07-02 (later)** — M0.1 done: repo builds as `libvllm` with tests
  (doctest/ctest), optional CUDA verified on dgx.casa at sm_121 (nvcc
  13.0.88, cubin confirmed, smoke kernel ran on GB10, 2/2 tests), CI green
  on GitHub Actions (run 28625582067 for `411c072`). Vendored: doctest
  v2.5.2, nlohmann/json v3.12.0, cpp-httplib v0.49.0. CI access was
  restored by adding localai-bot as collaborator. Final whole-branch review:
  READY TO CLOSE; deferred minors triaged (BUILD_INTERFACE → M3.5;
  vllm_test_main warnings + CMake nits → opportunistic in M0.2; smoke.cu
  placeholder dies in M0.2). NEXT: M0.2 vt runtime core (Tensor/dtype/
  device, arena, op dispatch, CPU scalar ops).
- **2026-07-02 (later)** — Backend portability scoped (user): Metal (MLX vs
  native MSL exploration, ANE for encoder-class only), Vulkan, Intel XPU —
  strategy in new `backends.md`; portability confined to vLLM's own seams
  (platforms/ + attention backends + vt op tables); binding vt:: interface
  requirements added to M0.2 roadmap unit; implementations post-MVP (D10 in
  design doc; deviation §9.6 for Metal/Vulkan extensions). NVIDIA gate
  unchanged.
- **2026-07-03** — M0.2 done: vt runtime core landed (dtype/tensor/backend/
  arena/dispatch + 5 CPU scalar ref ops, all unit-tested; backends.md
  interface requirements honored: open device enum, unified-memory flag,
  per-op Queue, graph-capture hook stubs). NEXT: M0.3 parity harness
  (upstream golden dumps on dgx.casa + C++ replay). Parity-harness note:
  upstream RoPE cos/sin cache is f32 and drifts at long context (~1e-2 at
  pos 131k) — M0.3 goldens need position-scaled tolerance; rmsnorm/silu
  standard paths keep f32 precision vs upstream's pre-multiply dtype
  rounding (documented in ops.h).
