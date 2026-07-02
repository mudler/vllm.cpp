# Upstream sync protocol

- **Reference checkout:** `/home/mudler/_git/vllm`, branch `main`
  (https://github.com/vllm-project/vllm).
- **PARITY PIN:** `e24d1b24` (2026-07-02) — the vLLM commit our current state
  is defined and verified against. All inventory statuses, golden dumps, file
  headers, and benchmark baselines are relative to the pin. This line is the
  single source of truth for it.

## Concepts

- **Parity pin** — one repo-wide vLLM commit. "We have feature X" always means
  "X as of the pin". Never compare against a moving target.
- **Per-file pins** — every ported file's header records the upstream path +
  the upstream commit it matches. Normally equal to the parity pin; a file may
  be temporarily ahead (hot-fix port) but never behind without a ledger note.
- **Sync cycle** — the repeatable task that advances the pin. Bounded,
  mechanical, agent-runnable on a cadence (weekly, or on demand, e.g. when a
  needed upstream fix lands).

## The sync cycle (repeatable task)

1. **Fetch & choose target.** `git fetch origin main` in the reference
   checkout. Target = `origin/main` HEAD (or a specific commit/tag if the user
   asked for one). Do NOT move the working tree yet if mid-cycle work exists.
2. **Enumerate.** `git log --oneline PIN..TARGET -- <subtree>` for each
   subtree we mirror. Because our structure mirrors upstream 1:1, the ported
   surface is derivable: map `src/vllm/**/*.{h,cpp}` back to `vllm/**/*.py`
   (plus `tools/parity/` dump scripts and ported test files).
3. **Classify** every commit touching those paths (upstream PR # is in the
   subject line):
   - **PORT-NOW** — changes behavior of code we've already ported.
   - **INVENTORY** — adds a feature we haven't ported; add/annotate it in
     [porting-inventory.md](porting-inventory.md) with its `vllm#` ref and a
     tier; do not port unless tiered T0/T1 and scheduled.
   - **IGNORE** — touches ported paths but is irrelevant to us (Python-only
     refactors, torch.compile plumbing, platforms we don't target); record
     the reason.
4. **Write the sync report** to `.agents/sync/YYYY-MM-DD-<target7>.md`:
   the three lists above, each entry `vllm#NNNNN <subject> — <disposition,
   one-line reason>`. The report is the reviewable artifact and the work
   queue for step 5.
5. **Port the PORT-NOW queue**, one upstream PR per commit where feasible:
   read the upstream diff (`git show` / `git diff <file-pin>..TARGET -- path`),
   translate it into the mirrored C++ file(s), bump those file headers to
   TARGET, add/adjust tests, append a [parity-ledger.md](parity-ledger.md)
   row per ported PR (upstream ref goes in the "Upstream equivalent" column).
6. **Re-verify.** Regenerate golden dumps at TARGET on dgx.casa (parity
   harness), run op/behavioral/model suites; if benchmarks are baselined, the
   vLLM baseline must be re-measured at TARGET before comparing.
7. **Advance the pin.** Fast-forward the reference checkout to TARGET, update
   the PARITY PIN line above, append a [state.md](state.md) entry linking the
   sync report. A cycle that stalls mid-way keeps the old pin and records
   what's left in the report ("carry-over" section) — the next cycle picks it
   up.

## Rules

- Ledger and inventory updates are part of the cycle, not optional follow-ups.
- Never mix a sync cycle with feature work in the same commit.
- If an upstream change conflicts with a recorded deviation (inventory §9),
  the deviation doc gets updated in the same cycle — deviations must always
  describe the current truth.
- Tooling: `tools/sync/` (roadmap unit P1) automates steps 2–4 (enumerate,
  map to ported files via headers, draft the report). Until it exists, do the
  steps by hand exactly as written.
