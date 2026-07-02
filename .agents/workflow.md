# How we work

- Read `AGENTS.md` (the index) first, every session; follow its links for
  scope; append to [state.md](state.md) as work lands.
- Brainstorm/design before code (superpowers skills); specs in
  `docs/superpowers/specs/`; TDD + parity harness during implementation.
- Push directly to `main` (user-authorized, for now).
- Long CUDA builds/benchmarks run on `dgx.casa` over SSH.
- Benchmarks are honest: same box, same model files, request-rate sweeps,
  report TTFT/ITL/throughput; no cherry-picking.
- Upstream sync procedure: [upstream-sync.md](upstream-sync.md).
