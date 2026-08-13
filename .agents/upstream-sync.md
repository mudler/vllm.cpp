# Upstream sync protocol

The reference checkout, fetch remote and eligible gate host come from the
untracked `developer-preferences.md`. The parity pin and classification rules
are repository-wide; machine paths are not.

**Active pin (advanced 2026-07-26):** current parity pin `555967922` (vLLM
0.26.0.dev0 + transformers 5.14.1, Torch 2.13.0, FlashInfer 0.6.15.post1, CUTLASS
DSL 4.6.0), FLIPPED from the prior `e24d1b24`/0.25.0 pin at W5 (`bc415a3e`), see
[specs/pin-advance.md](specs/pin-advance.md) §7. The advance re-validated with
zero real golden drift (27B-W4A4 + 32B-NVFP4A16 bit-identical, 35B/Coder
byte-stable) and unblocked DFlash (vllm#40898), Gemma-4 (`transformers.gemma4`),
and OLMo-3 (nested rope); a `vllm-oracle-v0.25.0-stage` rollback is preserved.

**The pin, as a running oracle reports itself.** The paragraph above names the
release; the block below carries the exact strings a runtime identity check can
compare, measured 2026-08-12 from `~/venvs/vllm-oracle-next` on dgx. They are not
derivable from the release number — the oracle reports
`0.23.1rc1.dev1511+g555967922`, not `0.26.0.dev0`, and its distribution metadata
adds a `.precompiled` suffix its runtime string lacks. `tools/bench/` reads this
block rather than duplicating it (#520); the duplicate drifted once, and the
harness spent 17 days *refusing* the oracle this record required. Advance it only
as part of a sync cycle, from a measured oracle, never by transcribing a version
number.

```parity-pin
vllm_commit = 5559679229bc961848b121ccdeaa8fa5d79bec98
vllm_runtime_version = 0.23.1rc1.dev1511+g555967922
vllm_distribution_version = 0.23.1rc1.dev1511+g555967922.precompiled
flashinfer_version = 0.6.15.post1
```

**`vllm_runtime_version` must carry a `+g<sha>` segment naming `vllm_commit`.**
That is a permanent constraint of this design, not a property of today's values:
`assert_oracle_commit` extracts the segment and requires it to prefix
`vllm_commit`. A released-wheel shape (`0.26.0`, no local version segment) is
therefore unusable — set the block to one and the harness refuses every oracle
including the pin itself; measured 2026-08-12, 34 of the 233 `tests/tools` cases
go red. Fail-closed and CI-caught rather than silent, but it means a pin advance
is taken from a source build's measured `vllm.__version__`, never transcribed
from a release number. If a future pin is genuinely a released wheel, give the
commit its own asserted field first; do not delete the assertion to make the
block parse.

**vLLM-Omni's pin does NOT live here.** It is a separate repository, and under
AGENTS.md §"When vLLM has no implementation" every oracle carries its own file:
[`.agents/oracles/vllm-omni.md`](oracles/vllm-omni.md), whose `oracle-pin` block
is the one place its revision is recorded. Do not add a second pin block to this
file; one file per oracle, read by glob, is what keeps a pin from becoming a
surface every change has to write. What belongs HERE is only the part that is
about the relationship between the two, which the oracle file cannot state on its
own: see the omni rules under §Rules, and
[specs/upstream-omni-pin.md](specs/upstream-omni-pin.md) (#633) for why they hold.

**Prior cycle (2026-07-12):** audited target v0.25.0 `702f4814fe54`; report
[`sync/2026-07-12-702f481.md`](sync/2026-07-12-702f481.md). The exact 145-commit
`e24d1b24..702f481` delta was classified (94 `INVENTORY`, 51 `IGNORE`, no
trace-independent `PORT-NOW` change in the implemented Qwen T0 slice). Immutable
`9cc7191` established the first new 27B denominator, while immutable `3f256ab`
superseded it at **55/124 axes pass, 69 fail**; the current binding is `9ecd9d0`
**114/124** (see roadmap_v1.md).

- **Reference checkout:** `${VLLM_SOURCE}`, branch selected by the developer
  preferences (normally `main`)
  (https://github.com/vllm-project/vllm).
- **STARTING PIN (MVP phase):** `e24d1b24` (2026-07-02) — the vLLM commit we
  port *from*. During the MVP build-out this is **not a parity claim**: we are
  not "at" this pin, we are building toward it, and the feature gaps vs this
  pin are documented in [porting-inventory.md](porting-inventory.md) (tier
  assignments + status markers = the gap record). Golden dumps, file headers,
  and benchmark baselines are all taken at this pin so the target stays fixed.
- Once the MVP gates pass, this line becomes the **PARITY PIN** — a statement
  of equivalence for the T0 surface — and moves only via sync cycles.

## Concepts

- **Starting pin (MVP)** — the fixed upstream snapshot the MVP is built
  against. Gaps vs it are normal and tracked in the inventory, not hidden.
- **Parity pin (post-MVP)** — one repo-wide vLLM commit. "We have feature X"
  always means "X as of the pin". Never compare against a moving target.
- **Omni parity pin** — the same idea for `vllm-project/vllm-omni`, recorded in
  [`.agents/oracles/vllm-omni.md`](oracles/vllm-omni.md), not here. It is a
  second pin rather than a second value of this one: it names the vLLM commit
  *it* ran against, which need not be ours.
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
6. **Re-verify.** Regenerate golden dumps at TARGET on an eligible host selected
   by the developer preferences, then run op/behavioral/model suites. If the
   required release-gate hardware is unavailable, retain the gate as `PENDING`
   with an exact handoff. If benchmarks are baselined, the vLLM baseline must
   be re-measured at TARGET before comparing.
7. **Advance the pin.** Fast-forward the reference checkout to TARGET, update
   the PARITY PIN line above, append an indexed immutable state event linking the
   sync report. A cycle that stalls mid-way keeps the old pin and records
   what's left in the report ("carry-over" section) — the next cycle picks it
   up.

## Rules

- Ledger and inventory updates are part of the cycle, not optional follow-ups.
- An omni-gated number is labeled with BOTH commits and is never cited in a
  vLLM-side parity claim, a binding grid, or a `docs/BENCHMARKS.md` row owned by
  a core-pinned row.
- Advancing the omni pin does not re-open the vLLM-side binding grids PROVIDED
  the omni oracle is installed in its own virtualenv and touches neither
  `${VLLM_SOURCE}` nor the environment the core pin measures itself from. If that
  isolation does not hold, the advance is a core sync cycle and is re-validated
  as one — the dependency tree under the denominator moved.
- Never mix a sync cycle with feature work in the same commit.
- If an upstream change conflicts with a recorded deviation (inventory §9),
  the deviation doc gets updated in the same cycle — deviations must always
  describe the current truth.
- Tooling: `tools/sync/` (roadmap unit P1) automates steps 2–4 (enumerate,
  map to ported files via headers, draft the report). Until it exists, do the
  steps by hand exactly as written.
