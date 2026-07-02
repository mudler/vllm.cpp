# State log (append; newest last)

- **2026-07-02** — Project kicked off. Repo created (`mudler/vllm.cpp`),
  pushed to GitHub. vLLM reference mapped (engine core + full feature
  inventory via subagents). Gates defined (Qwen3.6 NVFP4 parity on GB10 +
  GGUF + library + tools/grammars + streaming/non-streaming + e2e suites).
  `docs/porting-inventory.md` written with T0–T3 tiers. Core design doc
  rev 2 (CUDA-first on GB10, hybrid GDN+MoE+NVFP4 in T0, MRV2, library-first).
  Upstream synced to `e24d1b24`. AGENTS.md restructured as index over
  `.agents/*.md`. NOT STARTED: any implementation. NEXT: user reviews
  inventory + design doc → writing-plans for milestone M0.
