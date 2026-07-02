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
  restored by adding localai-bot as collaborator. NEXT: M0.2 vt runtime
  core (Tensor/dtype/device, arena, op dispatch, CPU scalar ops).
