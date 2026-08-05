# How we work — agent operating manual

This project is structured so **any agent (or engineer) can pick it up cold**
and continue. Follow this protocol every session.

## Session protocol

0. **Declare your role** before anything else, if this session has not already:
   `scripts/agent-role.py show` (exit 3 = undeclared), then
   `claim operator` or `claim helper --row <ROW-ID>`. Helpers then create their
   worktree, `row/<ROW-ID>` branch and DRAFT PR immediately — the draft PR is
   the claim. `scripts/ready-for-helper.py` lists what a helper may pick. Full
   protocol: [specs/operator-helper-protocol.md](specs/operator-helper-protocol.md).
1. **Orient**: read [NOW.md](NOW.md) FIRST — it is the one-Read resume surface
   (live claims, current gate, next actions) and is rewritten in place every
   checkpoint. The state tail is trustworthy only below the
   `<!-- state-order:enforced-below -->` marker, where every entry carries a
   sortable anchor and `scripts/check-state-order.py` proves the order; above it
   the record is frozen history that a union merge already interleaved. Then
   read `AGENTS.md` (index), the untracked
   `developer-preferences.md` when present, the top-level
   [roadmap_v1.md](roadmap_v1.md) row, its owning area matrix row,
   [coordination.md](coordination.md), and only the current carry-forward plus
   newest/relevant entries in [state.md](state.md). Search by stable row ID and
   inspect the tail; do not load the append-only state or parity ledger from
   beginning to end merely because it exists.
2. **Claim one stable row ID for coordinated roadmap work**: when parallel
   work is enabled or the task joins an existing claim block, add the
   sub-agent/worktree/branch and owned files to the coordination table before
   editing. One focused row or explicitly listed row block per PR; split
   oversized rows in the area matrix first. A single-agent ad hoc task does not
   create a claim merely to satisfy ceremony.
3. **Spike first**: create `.agents/specs/<slug>.md` and move the row through
   `INVENTORIED -> SPIKE -> READY`. The spike must cover upstream + dependency
   dispatch, tests to port, gates, hardware, dependencies, and row-sized work.
   Implementation must not start from a missing spike. The spike must be in the
   reviewable change before implementation and committed before integration.
4. **Read upstream first**: before implementing any subsystem, read the
   upstream Python at the paths listed in
   [porting-inventory.md](porting-inventory.md) — port, don't reinvent
   ([discipline.md](discipline.md)).
5. **Tests -> code**: port the upstream tests listed in the spike and use the
   parity harness before filling implementation anchors.
6. **Close the loop** (Definition of Done for every feature/iteration
   checkpoint). `scripts/agent-preflight.sh --staged` runs every record gate and
   its mutation suite in one command and prints [NOW.md](NOW.md); run it at
   session start and again before committing. Governance, review and Git-housekeeping work runs its relevant
   checks without claiming a feature-state change; touching a path classified
   as a checkpoint by `check-doc-checkpoint.py` still retains that checker's
   same-change public-document contract:
   - `python3 scripts/check-agent-record.py` and
     `python3 tests/scripts/test_agent_record.py` pass (canonical tables, stable
     IDs, semantic row fields, lifecycle/spec/claim/DONE integrity, pinned
     inventory size, and mutation proof that malformed rows fail);
   - `python3 scripts/check-doc-checkpoint.py --staged` and
     `python3 tests/scripts/test_doc_checkpoint.py` pass: every feature or
     iteration checkpoint updates the obligated public surfaces below in the
     same change, even when the honest result is pending, failed, or void;
   - tests green (op-parity / behavioral / e2e as applicable);
   - every feature/milestone that can plausibly affect speed, latency,
     scheduling, memory traffic, loading, or peak memory completes its own
     [benchmark-protocol](benchmark-protocol.md) checkpoint: same-binary
     pre/post A/B, fresh same-box applicable floor (vLLM, llama.cpp, or the
     backend-native oracle), correctness first, every applicable axis, and
     2–3 uncontended reproducing runs. Do not stack a second speed-sensitive
     milestone before the first checkpoint is recorded; unavailable hardware
     leaves the row `GATING` with the exact handoff recipe rather than `DONE`;
   - **committing parity goldens BEFORE their runner? In the SAME commit, add
     the op to `PendingRunnerOps()` in `tests/parity/test_op_parity.cpp`** —
     the harness scans golden dirs eagerly and hard-FAILs an op with no runner
     (anti-stale-golden gate). Skipping this reddens CI until the runner lands
     (burned us twice: M0.8 MoE, M0.9 qwen36). The runner task removes the
     entry; the milestone close-out asserts the set is empty of its ops. Always
     verify CI green after any commit that touches goldens when the developer
     preferences authorize a push and CI access; otherwise return the exact CI
     handoff.
   - ported files carry upstream path + commit headers;
   - [porting-inventory.md](porting-inventory.md) status markers flipped;
   - [parity-ledger.md](parity-ledger.md) row appended (what it does vs vLLM,
     upstream refs);
   - owning area-matrix row has exact implementation + test/evidence anchors;
   - [roadmap_v1.md](roadmap_v1.md) portfolio row and
     [coordination.md](coordination.md) claim updated;
   - **`docs/STATUS.md` per-capability status** updated at every
     feature/iteration checkpoint — not only at feature closure: the ONE binding
     current-state line for the capability this change moves, with its exact
     lifecycle stage, active gaps, and next gate. **NOT `README.md`** — the
     README is a user-facing landing page and changes only when a user-visible
     HEADLINE shifts (new architecture family, new backend or quantization
     format, changed headline numbers, changed pre-release caveat). A
     family/backend becomes ✅ only when parity-tested per
     [discipline.md](discipline.md);
   - **`docs/FEATURES.md` row** updated in the same change whenever a
     feature/model/backend/quantization surface moves;
   - **[NOW.md](NOW.md) refreshed** in the same change as the `state.md` append
     below (live claims, current gate, next actions, stamp) —
     `scripts/check-now-current.py` gates both its budget and its freshness;
   - **`docs/BENCHMARKS.md` benchmark disposition** updated at that same
     checkpoint with accepted comparable numbers or an explicit
     `PENDING`/`NOT APPLICABLE`/`FAILED`/`VOID` result and next reproduction
     recipe; partial evidence never becomes a public ratio;
   - all live status surfaces remain **current-state snapshots**: replace the
     prior checkpoint instead of appending chronology, and compact any
     accumulated superseded narratives to the binding result, exact present
     evidence/caveat, and next action. Detailed attempt history belongs only in
     append-only `state.md` / `parity-ledger.md`, Git, or an era-closed
     `completed/` document. Never rewrite entries inside an open era; at a
     roadmap/campaign boundary atomically freeze the raw files in `completed/`,
     seed concise live carry-forward files, and repair all live links;
   - [state.md](state.md) entry appended (what landed, what's next), below the
     `<!-- state-order:enforced-below -->` marker and carrying a
     `<!-- state: YYYY-MM-DD -->` anchor on the line after its heading, so the
     tail stays genuinely newest-last. After a union merge of two worktrees'
     appends, repair the order with
     `python3 scripts/sort-state-tail.py --apply`;
   - commit the completed in-scope change with the required trailers;
   - integrate, push, open a PR, or leave the commit local exactly as selected
     by `developer-preferences.md`. The project protocol itself grants no push,
     merge, force-update, or remote-host authority.

## Obligated public surfaces

These are the surfaces `scripts/check-doc-checkpoint.py` enforces, declared here
so the manual can never drift from the gate. `scripts/check-protocol-consistency.py`
asserts this block equals the checker's constants and matches AGENTS.md verbatim;
changing one without the other is a red build. `README.md` is deliberately absent
— pointing the per-checkpoint obligation at the README is what drifted it from a
landing page into a status log.

<!-- doc-obligation-contract:begin -->
| Public surface | Owed by |
|---|---|
| `docs/STATUS.md` | every feature/iteration checkpoint |
| `docs/BENCHMARKS.md` | every feature/iteration checkpoint |
| `docs/FEATURES.md` | any change to a feature/model/backend/quantization surface |
<!-- doc-obligation-contract:end -->

## Tabular lifecycle

| State | Meaning | Required before entering |
|---|---|---|
| `INVENTORIED` | Upstream item is listed; spike is not complete | Stable ID + upstream anchor |
| `SPIKE` | A named agent is investigating and writing the spec | Coordination claim + worktree |
| `READY` | Spike is merged; implementation can be claimed | Real spec link + tests/gates/dependencies |
| `ACTIVE` | Implementation is in flight | Implementation claim + owned files |
| `GATING` | Code exists; required parity/perf gates are running | Code anchors + ported tests |
| `PARTIAL` | Some modes work; missing modes are explicit | Code/test anchors for the working subset |
| `DONE` | Whole row scope is merged and gated | Code + tests/evidence + spec + ledger anchor + exact closing commit |
| `BLOCKED` | External dependency prevents progress | Concrete blocker + unblocking condition |
| `ANCHOR-BACKFILL` | Legacy code exists but the new evidence contract is incomplete | Honest working-scope note; cannot count as `DONE` |

Support inventories retain `DONE` rows permanently. When a roadmap execution
block has no open rows, archive its plan/report under `completed/` and point the
portfolio row at that archive; do not erase current support evidence.

All implementation and test/evidence anchors resolve to a permitted local path
class and an in-range line. `DONE` uses the closing commit in `Owner`, an exact
parity-ledger line, and a commit present in Git history. The CI mutation suite
must prove wrong-class links, out-of-range lines, missing claims/spec identity,
and false closure commits are rejected.

## Practicalities

- Run builds and benchmarks only on hosts selected by
  `developer-preferences.md`. Coordinated claims use distinct build/evidence
  directories and the selected `${GPU_LOCK}` policy whenever work may contend.
  A sole owner may run correctness work lock-free after verifying idleness if
  the preference file permits it; every benchmark/A-B/profile sequence remains
  uncontended and holds one exclusion mechanism across all arms. Compilation,
  read-only device inspection and file transfer do not require a GPU lock.
  Ettore-host examples and model locations are factual entries in
  [environment.md](environment.md), not defaults for other developers.
- Benchmarks are honest: same box, same model files, request-rate sweeps,
  report TTFT/ITL/throughput; no cherry-picking. Numbers go in the ledger.
- Upstream sync procedure: [upstream-sync.md](upstream-sync.md). When syncing,
  add newly-landed upstream features to the inventory with vLLM PR refs.
- Blocked or made a judgment call? Record it in state.md so the next session
  (or the user) sees it.
