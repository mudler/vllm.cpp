# Parity ledger (append-only)

Static record of **every change we introduce**: what it does compared to vLLM,
and what vLLM has, with references. One row per PR/commit that affects engine
behavior, ports upstream code, or intentionally deviates. Newest last.

Columns:
- **Date / our commit** — when and what landed here.
- **What it does** — one line, engineer-readable.
- **Upstream equivalent** — vLLM path(s) + upstream commit/PR reference(s)
  (`vllm#NNNNN`) this corresponds to; "deviation §9" if none by design.
- **Parity status** — how equivalence was verified (op dump / behavioral test /
  logits match / benchmark), or what's still missing vs upstream.

| Date / commit | What it does | Upstream equivalent | Parity status |
|---|---|---|---|
| 2026-07-02 `462d673` | Project record + inventory established (docs only) | — (meta) | n/a |
| 2026-07-02 `411c072` | Build skeleton: libvllm static lib, doctest/ctest harness, optional CUDA (sm_121 verified on GB10), CPU CI | — (build infra; no engine behavior; upstream has setup.py/cmake in csrc — not mirrored, deviation §9 compute layer) | CPU ctest + GB10 CUDA smoke + green CI run |
| 2026-07-03 `8df527e` | vt runtime core: dtypes(f16/bf16), Tensor views, Backend/Queue abstraction (CPU; unified-memory + graph-capture hooks per backends.md), StepArena, op registry + CPU scalar matmul/rmsnorm(+gemma,+fused-residual)/silu_and_mul/embedding/partial-neox-rope | — (vt is deviation §9.1; op semantics mirror layers/{layernorm,activation}.py + rotary_embedding NeoX) | golden-value op unit tests (upstream dump parity lands in M0.3); reviewers verified op semantics against upstream layernorm.py/activation.py/rotary_embedding sources |
| 2026-07-03 `f063890` | Parity harness: oracle venv (pip vLLM 0.24.0 on dgx.casa), dump scripts, NPY reader, NaN-loud manifest-driven runner; 9 op golden cases green in CI | test infra (mirrors upstream forward_native semantics as oracle); drift-checked vs pin e24d1b24 | 9/9 cases pass at planned tolerances (rope 131k needs 2e-2 atol — upstream f32 cache drift) |
