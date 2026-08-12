# BENCH-ORACLE-PIN-RECONCILE — the harness may not enforce an oracle the pin record forbids

Issue: [#520](https://github.com/mudler/vllm.cpp/issues/520) (the defect),
[#522](https://github.com/mudler/vllm.cpp/issues/522) (the pandas blocker that
gates the first binding run against the pin)
Row: `BENCH-ORACLE-PIN-RECONCILE`
Prior art: [#375](https://github.com/mudler/vllm.cpp/issues/375) (the symlink
points at the rollback), [#417](https://github.com/mudler/vllm.cpp/issues/417)
Finding 2 (first report), [#414](https://github.com/mudler/vllm.cpp/issues/414)
(the denominator's *configuration*, a separate defect on the same runs).

## Scope

**In scope.** The three constants that are the harness's *live* statement of
which oracle a comparison ran against, and the assertion that enforces them:

| file:line | constant | today |
|---|---|---|
| `tools/bench/online_gate.py:53` | `VLLM_ORACLE_VERSION` | `"0.25.0"` |
| `tools/bench/online_gate.py:54` | `FLASHINFER_VERSION` | `"0.6.13"` |
| `tools/bench/serve_low_common.py:28` | `VLLM_COMMIT` | `"702f4814…"` |

Every other consumer in the tree (`online_gate_summary.py`,
`gdn_packed_component.py`, `run_serve_low.py`, and four `tests/tools` modules)
**imports** these three names; grep confirms no fourth definition and no
hardcoded literal outside them. So the reconcile is three values, not N files.

**Out of scope, deliberately.**

- **Golden provenance.** `tools/bench/gdn_ba_projection_oracle.py:24` and
  `gdn_packed_decode_oracle.py:28` carry their own `TARGET_COMMIT = 702f4814…`,
  and `tests/parity/goldens/{gdn-packed-decode-oracle/manifest.json,
  gdn_ba_projection_bf16_sm121/oracle.json}` plus
  `tests/fixtures/nvfp4_flashinfer_v025_gb10/manifest.json` record
  `vllm_version: 0.25.0`. Those state **what produced an immutable artifact**,
  which is a true historical fact. Advancing them would make the tree lie about
  its own goldens. They move only when the goldens are recaptured.
- **Historical narrative** in `.agents/**` records, specs, and capture-script
  docstrings. History is git; a past run that used 0.25.0 keeps saying so.
  *Refined by fresh review (Finding 3):* a **present-tense claim about which
  oracle is pinned** is not narrative, and two such claims contradicted the
  record — `tools/bench/scheduler_wave_diff.py:21` ("the pinned vLLM oracle
  (~/venvs/vllm-oracle == vllm 0.25.0 == 702f481)") and
  `tools/bench/mxfp4_smoke_gate.py:11` ("captured from the pinned 0.25.0
  oracle"). Both are corrected to name the current pin while KEEPING the capture
  provenance, which is the historical fact. Neither is enforced by code. The two
  `--require-vllm-version 0.25.0` defaults in the GDN oracle scripts stay
  untouched under the golden-provenance bullet above.
- **Re-measuring anything.** No grid runs in this row — #522 blocks it, and the
  GPU is queued. This row makes the pin *reachable*; the first binding grid
  against it is a separate row with its own gate.
- **The `--language-model-only` denominator defect.** That is #414. This row
  resolves only whether the flag *exists* at the pin (§Task 2 finding), because
  a live claim said it did not.

## The defect

AGENTS.md: *"Comparisons run against the pinned oracle recorded in
`.agents/upstream-sync.md`."* The pin advanced 2026-07-26 to `555967922`
(`upstream-sync.md:7-9`). The harness did not follow, and it does not merely
default to the old oracle — `online_gate.py:3509-3533` **raises**
`HarnessError: vLLM oracle version drift` on anything that is not `0.25.0`.

**Nobody could have measured against the pin even deliberately.** That is the
part #375 does not cover. #375 is a symlink pointing at a rollback — an
operational slip. This is the gate *demanding* the rollback, which no amount of
operator care defeats. A record disagreeing with the tree, per AGENTS.md, is
reconciled, not argued.

## Measured identity of both venvs (2026-08-12, dgx, read-only)

| | `vllm-oracle-v0.25.0-stage` | `vllm-oracle-next` (**the PIN**) |
|---|---|---|
| `vllm.__version__` | `0.25.0` | `0.23.1rc1.dev1511+g555967922` |
| dist metadata | `0.25.0` | `0.23.1rc1.dev1511+g555967922.precompiled` |
| `flashinfer.__version__` / metadata | `0.6.13` / `0.6.13` | `0.6.15.post1` / `0.6.15.post1` |
| torch / transformers | 2.11.0 / 5.13.1 | 2.13.0 / 5.14.1 |
| pandas | 2.2.3 | **MISSING** (#522) |
| install | site-packages | **editable** → `~/work/vllm-src-5559679` @`5559679229bc9618` |

`~/venvs/vllm-oracle` → `vllm-oracle-v0.25.0-stage`, and
`scripts/dgx-online-serving.sh:35` resolves exactly that path, so the canonical
driver takes the rollback by construction.

**Two facts that must land in the constants, and that guessing would have got
wrong.**

1. The pin's runtime version is **`0.23.1rc1.dev1511+g555967922`** — *not* the
   `0.26.0.dev0` the pin record's prose names. Reading the constant out of
   `upstream-sync.md`'s prose would therefore have produced a string the oracle
   never reports.
2. Metadata carries a `.precompiled` suffix the runtime string lacks. The
   existing check asserts `metadata == runtime == CONST`, which **cannot hold**
   on the pin regardless of the value chosen. The assertion shape has to change,
   not just its constants.

## Design

**One source of truth, as a fenced machine-readable block, not a prose parser.**
`.agents/upstream-sync.md` gains a `parity-pin` fenced block carrying the five
exact strings a runtime check can compare. `serve_low_common.py` reads that block
with ~30 lines of stdlib (the module's existing constraint: stdlib only, so its
validators run in CPU CI without either engine) and exports
`VLLM_COMMIT`, `VLLM_ORACLE_VERSION`, `VLLM_DISTRIBUTION_VERSION`,
`FLASHINFER_VERSION`. `online_gate.py` imports the two it used to define.

This is admissible under Records because the block is **one keyed record read at
load time**, not a surface every PR writes: only a pin advance touches it.

**Rejected: parsing the prose pin line.** The prose says `0.26.0.dev0`; the
oracle says `0.23.1rc1.dev1511+g555967922`. A parser over prose would encode a
string that no assertion can ever match — the framework would be both more code
and wrong.

**Rejected: leaving three constants in two files.** They already drifted once,
which is this issue.

### The assertion is STRENGTHENED, never removed

Three changes, each closing a hole the old check left open:

1. **Commit SHA is asserted.** `VLLM_COMMIT` was previously never checked
   against a running oracle at all — only stamped into manifests. The new check
   requires the resolved SHA's `+g<short>` to appear in `vllm.__version__`.

   **CORRECTION (fresh review, Finding 2).** The first commit message of this
   row called that "THE CHECK #375 NEEDED", and that overstates it. The
   assertion is **defence in depth and is inert at this pin**: at all three call
   sites an exact equality against `VLLM_ORACLE_VERSION` runs first, and that
   constant already *contains* `+g555967922`, so every string that passes the
   equality also passes the assertion. It cannot fire in production today. What
   refuses the rollback today is **the updated constant** — the equality — which
   is exactly the reconcile this row is named for. The assertion earns its place
   where the two come apart: a manifest read off disk from another venv or
   another day, a hand-edited evidence file, or a pin whose recorded version is
   a plain release number. The commit message cannot be rewritten (`main` is
   never force-pushed), so the correction lives here and in the function's own
   docstring, which is where a reader meets it.
2. **Metadata is matched against its own recorded string**, so the
   `.precompiled` suffix is asserted rather than assumed away with a
   `startswith` that would also accept an arbitrary suffix.
3. **A missing or malformed pin block is fatal** at import, so the failure mode
   of the new indirection is a refusal to measure, never a silent default.

`PANDAS_VERSION` stays as-is and keeps failing closed on the pin. That failure
is the true state of the world (#522) and is not papered over here.

### The pin block permanently requires a `+g<sha>` runtime version

Recorded because it was nowhere written down (fresh review, Finding 4).
`assert_oracle_commit` extracts `+g<sha>` from a version string and requires it
to prefix `vllm_commit`, and it is applied to the **recorded** string as much as
to the oracle's. So `vllm_runtime_version` may never be a released-wheel shape:
set the block to `0.26.0` and the harness refuses every oracle including the pin
itself — **measured 2026-08-12: 34 of the 233 `tests/tools` cases go red**. That
is fail-closed and CI-caught, never silent, and it is the correct polarity. But
it constrains the next advance: take the value from a source build's measured
`vllm.__version__`, never transcribe a release number. A genuinely
released-wheel pin needs the commit carried in its own asserted field first —
never by deleting the assertion to make the block parse.

### Operational consequence on merge: `record-oracle` will ABORT on dgx

**This is the gate working, not a breakage.** `~/venvs/vllm-oracle` is host state
and still symlinks the preserved 0.25.0 rollback, so the next
`scripts/dgx-online-serving.sh record-oracle` through that symlink aborts with
`vLLM oracle version drift` naming the pinned strings. That is the whole point of
the row: before this change the same symlink silently produced a manifest
labelled with the pin it did not run. Repointing the symlink is **#375's**, not
this row's (see Stop conditions) — whoever hits the abort should fix the symlink
or pass the pinned venv, and must not "fix" it by editing the constants back.

## Tests

`tests/tools/test_oracle_pin.py`, stdlib `unittest`, no GPU, no vLLM:

1. **The constants equal the pin record.** Reads
   `.agents/upstream-sync.md`'s block independently of the loader and asserts
   the four exported names match. RED before: constants are `0.25.0` /
   `0.6.13` / `702f4814…` and the block does not exist.
2. **The rollback is REJECTED by SHA.** Feeds the check the rollback's exact
   measured strings (`0.25.0`, flashinfer `0.6.13`) and requires `HarnessError`.
   This is the #375 mutation: it fails if the SHA term is deleted.
3. **The pin's real strings are ACCEPTED**, including the `.precompiled`
   metadata suffix — the case the old `metadata == runtime` shape rejected.
4. **A malformed pin block is fatal**, not defaulted.

**Added after fresh review (Finding 1): the guard must be proven WIRED IN.**
Cases 1–4 exercise `read_parity_pin` and `assert_oracle_commit` as pure
functions, which says nothing about whether the manifest entry points consult
them — and they did not have to: the reviewer gutted `record_oracle_manifest`'s
identity check *and* deleted its `assert_oracle_commit`, and 226/226 stayed
green. Six mutations survived (each of the three `assert_oracle_commit` call
sites deleted individually; either equality dropped in `record_oracle_manifest`;
the runtime equality plus assert dropped in `record_execution_manifest`).
`OracleIdentityIsWiredIntoEveryEntryPointTests` adds negative cases at each
of the three entry points — `record_oracle_manifest`, `record_execution_manifest`, and
`gdn_packed_component._validate_execution` — feeding the rollback's measured
shape and requiring **that entry point's own refusal message**, not a bare
`HarnessError`: these functions raise it for a dozen unrelated reasons, so an
unanchored `assertRaises` stays green on a gutted check that merely fails later.

Two of the six mutations are unreachable by any manifest value, because the
version equality dominates the assertion (§Design correction, Finding 2), so
those cases patch the version constant to a release-numbered shape — the only
input that reaches the assertion while the equality holds. All six mutations,
plus a seventh (dropping the GDN runtime equality) that also survived, are now
RED; each restored byte-for-byte and verified by sha256.

Regression surface: `tests/tools` in full (208 tests on the base SHA
`a89b3c456`, 226 after the row's first two commits, **233** with the wiring
cases; a changed count is RED even when it prints `OK`).

## Gates

- `python3 -m unittest discover -s tests/tools -t .` — full, serial. `pytest`
  mis-collects this tree and reports false failures; do not use it.
- `scripts/agent-preflight.sh --staged`, then again on committed HEAD.
- **No benchmark gate.** Blocked on #522 and on GPU availability, and recorded
  `PENDING` with the exact handoff rather than waived.

## Stop conditions

- Stop if closing the constant gap would require editing a **golden provenance**
  record; that is a recapture, not a reconcile.
- Stop if the pin block cannot be expressed without a parser more complex than
  the constants it replaces — three duplicated constants beat a framework.
- Stop before installing anything into `~/venvs/vllm-oracle-next`; that is
  #522's authority question, not this row's.
- Stop before repointing the `~/venvs/vllm-oracle` symlink. It is host state, it
  is #375's, and this row's identity assertion makes the wrong target *loud*
  rather than silent — which is the durable fix.

## Evidence

- Venv identity, both venvs, and the `--language-model-only` resolution: read-only
  `ssh dgx.casa` transcripts, 2026-08-12, quoted verbatim in #520 and in the
  `.agents/benchmark-record.md` entry this row appends.
- Constants survey: `grep -rn` over `tools/ scripts/ tests/ .agents/ docs/`;
  1,399 raw hits, of which exactly **3** are live definitions and the rest are
  imports, golden provenance, or narrative. The table is in #520.
- Wiring mutations (fresh-implementer round, Finding 1): all seven applied one at
  a time with `count == 1` anchor assertions, each run over the full
  `tests/tools` suite before the new cases and over `test_oracle_pin` after.
  Before: 7/7 **survived** at 226 green. After: 7/7 **RED**, each attributed to
  the intended new case. Every mutation restored byte-for-byte, verified by
  sha256 rather than timestamp.
- Released-wheel pin shape (Finding 4): `vllm_runtime_version = 0.26.0` in the
  block, full suite → `FAILED (failures=29, errors=5)` of 233 collected; record
  restored from a byte copy.

## Now

`ACTIVE` — constants reconciled and the assertion strengthened; the first
binding grid against the pin is `PENDING` on #522 (pandas) and GPU availability.
