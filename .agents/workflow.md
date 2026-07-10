# How we work — agent operating manual

This project is structured so **any agent (or engineer) can pick it up cold**
and continue. Follow this protocol every session.

## Session protocol

1. **Orient**: read `AGENTS.md` (index), then [state.md](state.md) (where we
   are, what's next), then the [roadmap_v1.md](roadmap_v1.md) unit you'll work on.
2. **Scope one unit**: work in roadmap-unit-sized chunks (one focused PR
   each). If a unit turns out too big, split it in roadmap_v1.md first.
3. **Read upstream first**: before implementing any subsystem, read the
   upstream Python at the paths listed in
   [porting-inventory.md](porting-inventory.md) — port, don't reinvent
   ([discipline.md](discipline.md)).
4. **Design → tests → code**: for non-trivial units, brainstorm/spec first
   (specs in `docs/superpowers/specs/`); TDD with the parity harness.
5. **Close the loop** (Definition of Done for every change):
   - tests green (op-parity / behavioral / e2e as applicable);
   - **committing parity goldens BEFORE their runner? In the SAME commit, add
     the op to `PendingRunnerOps()` in `tests/parity/test_op_parity.cpp`** —
     the harness scans golden dirs eagerly and hard-FAILs an op with no runner
     (anti-stale-golden gate). Skipping this reddens CI until the runner lands
     (burned us twice: M0.8 MoE, M0.9 qwen36). The runner task removes the
     entry; the milestone close-out asserts the set is empty of its ops. Always
     verify CI green (`gh run list`) after any commit that touches goldens.
   - ported files carry upstream path + commit headers;
   - [porting-inventory.md](porting-inventory.md) status markers flipped;
   - [parity-ledger.md](parity-ledger.md) row appended (what it does vs vLLM,
     upstream refs);
   - [roadmap_v1.md](roadmap_v1.md) unit status updated;
   - **`README.md` support tables** (architectures / acceleration /
     quantization) updated whenever a change alters what is supported, its
     status, or its tested formats — a family/backend becomes ✅ only when
     parity-tested per [discipline.md](discipline.md);
   - [state.md](state.md) entry appended (what landed, what's next);
   - commit + push to `main` (user-authorized, for now).

## Practicalities

- Long CUDA builds/benchmarks run on `dgx.casa` over SSH; artifacts in
  `~/work/vllm.cpp/` there. Specs/models: [environment.md](environment.md).
- Benchmarks are honest: same box, same model files, request-rate sweeps,
  report TTFT/ITL/throughput; no cherry-picking. Numbers go in the ledger.
- Upstream sync procedure: [upstream-sync.md](upstream-sync.md). When syncing,
  add newly-landed upstream features to the inventory with vLLM PR refs.
- Blocked or made a judgment call? Record it in state.md so the next session
  (or the user) sees it.
