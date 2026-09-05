# Sync cycle `e126687a9a`, wave PINADVANCE — step 7

Row: `UPSTREAM-SYNC-HEADPIN`.
Issue: [#2817](https://github.com/mudler/vllm.cpp/issues/2817).
Row spec: [`upstream-sync-headpin.md`](upstream-sync-headpin.md).
Predecessors: [#2593](https://github.com/mudler/vllm.cpp/issues/2593) HEADPIN
(install half), [#2611](https://github.com/mudler/vllm.cpp/issues/2611) RUNHALF
(run half), the seven PORTQ waves, and
[#2771](https://github.com/mudler/vllm.cpp/issues/2771) STEP6, which found the
ordering defect this wave acts on.
Guide: [`../upstream-sync.md`](../upstream-sync.md) §"The sync cycle".

## Now

**The pin has advanced.** It is
`e126687a9a828d513c01a07cd69f025f27d63280` (2026-08-31, vllm#53896), 1465
commits past `5559679229bc961848b121ccdeaa8fa5d79bec98`.

**Nothing was executed on a device.** No build, no lease, no GPU, no golden
re-capture, no benchmark. This wave edits records and citations. What it costs is
in §5 and it is not small: every gate in this tree is now owed a re-validation it
has not had.

## 1. The developer ruling, and it is the authority for this wave

`.agents/upstream-sync.md` §"The sync cycle" orders step 6 (re-measure baselined
vLLM denominators at TARGET) before step 7 (advance the pin). **That order cannot
be executed in this repository, and the reason is measured rather than argued.**

Wave STEP6 ([#2771](https://github.com/mudler/vllm.cpp/issues/2771), landed as
[#2783](https://github.com/mudler/vllm.cpp/pull/2783)) ran a hermetic probe over
the committed `record_oracle_manifest` and established that
`tools/bench/online_gate.py` refuses any oracle that is not the pinned one. It
checks `VLLM_DISTRIBUTION_VERSION` and `VLLM_ORACLE_VERSION` at `:3529-3540`, the
commit at `:3542`, and FlashInfer at `:3552-3560`, and all four values come out of
the ` ```parity-pin ` block in `.agents/upstream-sync.md` by way of
`tools/bench/serve_low_common.py:100-109`. Editing that block **is** step 7. So
step 6 is a precondition of an edit that is a precondition of step 6. STEP6 filed
this `NEEDS_DECISION` and refused to resolve it by loosening an assertion, which
was correct.

**The developer has ruled: advance the pin first, then re-run step 6 against it.**

This spec records that as a **developer decision that inverts the documented step
order**. It is not a reading the protocol permits and it is not presented as one.
Its consequence is stated with it, because a ruling without its cost is not a
record:

> **Step 6 becomes post-hoc validation.** The five affected rows keep their
> published ratios while their denominator is no longer the pin. If the
> re-measurement at `e126687a9a` comes back red, the required response is to
> **revert the pin**, not to hold it and re-argue the rows. The advance was taken
> on the expectation that those numbers survive it, and that expectation is the
> thing being validated.

Owed by [#2818](https://github.com/mudler/vllm.cpp/issues/2818).

**No checker enforces this ruling and none is added.** A gate over step order
would have to read the same block it is deciding about, and the tree already
carries one circular assertion too many.

## 2. Scope

**In scope.**

1. The four fields of the ` ```parity-pin ` block, replaced with strings MEASURED
   at the target, never transcribed from a release number.
2. `.agents/oracles/vllm.md`'s ` ```oracle-pin ` block and the prose around it.
3. The two committed records the advance falsifies (§4).
4. The `include/vllm/config/speculative.h` comment that names the pin as the
   reason for a refusal (§4.3).
5. Public surfaces that state the pin as a current fact:
   `docs/benchmarks/how-we-measure.md`, `docs/FEATURES.md`, and a statement on the
   two benchmark pages whose binding rows were measured at the previous pin.
6. This spec, and the sync report
   [`../sync/2026-09-03-e126687-advance.md`](../sync/2026-09-03-e126687-advance.md).

**Out of scope, each for a stated reason.**

- **Step 5, the 290-entry PORT-NOW queue.** Reconciled by the seven PORTQ waves
  and ruled not a blocker; see §3 for the state of that ruling. Porting an entry
  is product work and `.agents/upstream-sync.md` §Rules forbids mixing it into a
  sync cycle commit.
- **Bumping the >= 177 per-file `Ported from: ... 55596792` headers** (a
  measured FLOOR; see §5.3, which also records the two wrong figures that
  preceded it). They are
  per-file pins and step 5 is what moves them. §5.3 records that they are now
  behind, which is the ledger note `.agents/upstream-sync.md` §Concepts requires.
- **The #2649 product change.** It becomes landable at this merge and is not
  bundled, for the same no-mixing rule. §4.3 makes the comment honest without
  changing behaviour.
- **Re-capturing goldens.** Needs a device and a lease. §5.1.
- **Running step 6.** That is the ruling's second half and its own change
  (#2818).

## 3. What the advance rests on, including one thing that has not landed

`.agents/oracles/vllm.md` named four owed items before this wave. Their state at
this change's merge base `23ac6f1a7`:

| Owed item | State | Where |
|---|---|---|
| 1. The 290-entry PORT-NOW queue | Reconciled, ruled **not a blocker** | [#2768](https://github.com/mudler/vllm.cpp/pull/2768) — **OPEN, NOT MERGED** |
| 2. A declared token-exact gate at the target | **Owed, unmeasured** | [#2794](https://github.com/mudler/vllm.cpp/issues/2794) |
| 3. Step-6 re-measurement | **Owed, now post-hoc** | [#2818](https://github.com/mudler/vllm.cpp/issues/2818) |
| 4. A reading on `dgx:gpu0` | **Owed**, only `thor:gpu0` was measured | [#2611](https://github.com/mudler/vllm.cpp/issues/2611) |

**Item 1's discharge is in an unmerged pull request, and this wave says so rather
than inheriting the claim.** #2768 is `OPEN` with `headRefName`
`row/PORTQ-RECONCILE-W1` and `mergedAt: null`, and `.agents/sync/` on
`origin/main` carries no reconciliation report — the only reference to it in the
tree is `.agents/sync/2026-09-03-e126687-step6.md:10`, which cites it as a pull
request. The seven PORTQ tranche reports themselves ARE on `main`
(`.agents/sync/2026-09-03-portq{1..7}.md`), so the classification work is landed;
what is not landed is the pass that ruled the queue discharged. **The advance
proceeds on the developer ruling, and this is the one input it takes on trust.**

Items 2, 3 and 4 stand and are unresolved by this wave. §5 states what that means
rather than letting the pin's movement imply otherwise.

## 4. The records the advance falsifies

Each was verified against the two revisions directly, with the return code read
into its own variable rather than after a pipe, and each zero paired with a
positive control through the same probe form.

### 4.1 `.agents/porting.md` — ModelOpt fp8's output dtype

The too-wide-dtype table teaches that a quant method's `out_dtype` is
`torch.get_default_dtype()`, naming ModelOpt fp8 as the example. vllm#48861
(`0b37d8389f`, "NVFP4 quantization out_dtype should match model dtype, not torch
default") replaces that with `get_current_vllm_config().model_config.dtype` in all
three `ModelOptFp8*LinearMethod.__init__`.

```console
$ git grep -n 'get_default_dtype' 5559679229 -- vllm/model_executor/layers/quantization/modelopt.py
5559679229:...:458:        self.out_dtype = torch.get_default_dtype()
5559679229:...:551:        self.out_dtype = torch.get_default_dtype()
5559679229:...:648:        self.out_dtype = torch.get_default_dtype()
$ git grep -n 'get_default_dtype' e126687a9a -- vllm/model_executor/layers/quantization/modelopt.py
TARGET_GETDEF_RC=1
$ git grep -n 'self.out_dtype' e126687a9a -- vllm/model_executor/layers/quantization/modelopt.py
TARGET_OUTDTYPE_RC=0
e126687a9a:...:452:        self.out_dtype = get_current_vllm_config().model_config.dtype
e126687a9a:...:547:        self.out_dtype = get_current_vllm_config().model_config.dtype
e126687a9a:...:646:        self.out_dtype = get_current_vllm_config().model_config.dtype
```

The positive control is what makes `RC=1` a finding: the same probe form over the
same file at the same revision finds `self.out_dtype` six times. The line is not
merely stale — it is the guide a reader consults for exactly the check that entry
is about, so a wrong example there propagates into the ports it advises.

### 4.2 The `calculate_kv_scales` anchors — SEVEN sites, not six

vllm#49389 (`dd11df04f3`, "Remove deprecated calculate_kv_scales runtime KV scale
calculation") deletes the field. Verified:

```console
$ git grep -n 'calculate_kv_scales' 5559679229 -- vllm/config/cache.py
5559679229:vllm/config/cache.py:111:    calculate_kv_scales: bool = False        (+ 4 more)
$ git grep -n 'calculate_kv_scales' e126687a9a -- vllm/config/cache.py
RC=1
$ git grep -n 'cache_dtype' e126687a9a -- vllm/config/cache.py      # positive control
RC=0, 5 hits
```

Tree-wide at the target the identifier survives only inside one unrelated comment
(`vllm/models/kimi_k3/nvidia/mla.py:473`), so the citation target is gone rather
than moved.

**The count has been wrong three times, in the same direction, and the mechanism
is the same each time: a `:111` grep cannot see a citation that wraps.**
`.agents/sync/2026-09-03-portq2.md:321` records an earlier draft saying "two"; a
fresh reviewer of #2783 found five; PORTQ-2 published six. This wave re-derived it
from the SYMBOL rather than the line number and finds **seven**:

| # | Site | Form | Found by a `cache.py:111` grep? |
|---|---|---|---|
| 1 | `include/vllm/config/cache.h:2` | `calculate_kv_scales:111` | **No** — the filename is on line 1 |
| 2 | `include/vllm/model_executor/layers/quantization/kv_cache.h:90` | `` `cache.py:111` `` | Yes |
| 3 | `include/vllm/model_executor/layers/quantization/kv_cache.h:108` | **inside the thrown `VT_CHECK` message a user reads** | Yes |
| 4 | `tests/vllm/entrypoints/test_kv_cache_fp8_wiring.cpp:8` | `` `:111` calculate_kv_scales `` | **No** — wraps from `:7` |
| 5 | `tests/vllm/entrypoints/test_kv_cache_fp8_wiring.cpp:680` | `(cache.py:111)` | Yes |
| 6 | `.agents/specs/fp8-kv-cache.md:1044` | `` (`config/cache.py:111`) `` | Yes |
| 7 | `.agents/specs/fp8-kv-cache.md:46` | `` `:99-104` `` | **No** — a DIFFERENT anchor for the same deleted field |

Site 7 is the one every prior count missed, and PORTQ-2 explicitly ruled it
"adjacent rather than a seventh instance". That ruling was about the `:111`
string. It is wrong about the obligation: the field is deleted, so an anchor
pointing at it is falsified whatever number it carries. **Site 7 was ALSO wrong at
the old pin** — `vllm/config/cache.py:99-104` at `5559679229` is
`prefix_caching_hash_algo`'s docstring, not `calculate_kv_scales`, which is at
`:111`. It is repaired here as part of the same family rather than left as a
second defect nobody owns.

**The historical records are not edited.** `.agents/sync/2026-09-01-cdefd9d.md`,
`.agents/sync/2026-09-03-portq2.md`, `.agents/specs/upstream-sync-portq2.md` and
`.agents/completed/**` describe what a past wave read at the pin it read it at.
Rewriting them would forge the record.

**The behaviour does not change.** `ResolveKvCacheScales` still refuses
`--calculate-kv-scales` by name. Upstream removing the flag is a reason for the
refusal to be MORE accurate, not less: the message now says the option was removed
upstream, and names the commit that removed it, so a user who reads it can find
out what happened to their flag.

### 4.3 `include/vllm/config/speculative.h` — a comment that names the pin

`async_scheduling_compatible()` returns `use_eagle()`, and the comment above it
says `"draft_model"` is refused **at the pin** and instructs a reader to
"reconcile when the pin advances past that commit". This wave is that advance:

```console
$ git merge-base --is-ancestor c0202c5603 e126687a9a828d513c01a07cd69f025f27d63280
ANCESTOR_OF_TARGET_RC=0
$ git merge-base --is-ancestor c0202c5603 5559679229bc961848b121ccdeaa8fa5d79bec98
ANCESTOR_OF_PIN_RC=1
```

The comment is corrected to say the pin HAS advanced past it, that upstream now
auto-enables async scheduling for a `draft_model` speculator, and that this tree
knowingly does not yet — owed by
[#2649](https://github.com/mudler/vllm.cpp/issues/2649). **The predicate is not
changed here.** #2649 is a product change with a behaviour test and its own fresh
review, and `.agents/upstream-sync.md` §Rules forbids mixing feature work into a
sync cycle. What the comment must not do is keep asserting a condition the tree no
longer satisfies.

## 5. What the advance costs, said before anything else can imply otherwise

### 5.1 No gate has been run at this pin, and every golden predates it

The 2026-07-26 advance re-captured goldens on the new oracle on GB10 and recorded
zero real drift (`pin-advance.md` §7 W3b). **This advance has no such pass.** Every
committed golden in `tests/parity/goldens/` was captured against `555967922` or
earlier. The declared token-exact gate at the target is
[#2794](https://github.com/mudler/vllm.cpp/issues/2794), which additionally records
that the OPT golden was never re-validated at the OLD pin either.

The run half's six-prompt, sixteen-token agreement with `opt_greedy` is
informative and is not a gate; the wave that measured it said so
(`../sync/2026-09-03-e126687-runhalf.md` §7 C2).

### 5.2 Gateability: what "runs a model" does and does not claim

`gateable = yes` is set, `evidence` points at the run half, and the claim is
bounded in the oracle file itself. **Established**: at `e126687a9a`, on
`thor:gpu0` (aarch64, `sm_110`, driver 595.78), vLLM builds from source in 94
minutes, ships seven working compiled extensions with a kernel EXECUTED on device,
imports as `0.28.1rc1.dev132+ge126687a9` read from `cd /`, and serves greedy
tokens on FLASH_ATTN/FA2 in both eager and compiled configurations.

**Not established, and the advance does not make any of it true**:

- `qwen4_exp` — the model the target was wanted for — **does not run on this
  fleet** at this revision. Its QSA indexer's `cooperative_topk` refuses to launch
  with a cluster misconfiguration ([#2626](https://github.com/mudler/vllm.cpp/issues/2626),
  cause unestablished), and its published safetensors arms exceed the largest fleet
  box.
- **No reading on `dgx:gpu0`**, which is `sm_121a` and the device every binding
  benchmark number in this tree was taken on. One device, one day.
- **No gate model.** `facebook/opt-125m` is not one.
- Nothing about any other board. `.agents/oracles/vllm.md`'s device-scoped table
  keeps its rule: absence means unmeasured, never unsupported.

"The oracle builds and runs" and "this row can be gated against it" are different
statements and the record must not let the first read as the second.

### 5.3 Every ported file header now names a revision below the pin

`.agents/upstream-sync.md` §Concepts allows a per-file pin to be temporarily
ahead, and says it may never be behind "without a ledger note". Step 5 is what
moves those headers and step 5 did not run. Measured over `include/`, `src/` and
`tests/`: 556 files carry a `Ported from:` header, **at least 177** name
`55596792` and go behind with this advance, and **at least 330** name `e24d1b24`
and were ALREADY behind against the prior pin. Both are FLOORS, because the
headers wrap and no single probe sees all of them; the sync report §5.3 carries
the three probes, their blind spots, and the two wrong figures that preceded this
one. **This section is that ledger note.** They are
owed by the PORT-NOW queue's own issue,
[#2611](https://github.com/mudler/vllm.cpp/issues/2611), and by the per-entry
issues the PORTQ waves filed.

### 5.4 The published binding ratios were measured against the previous pin

Five rows across two public pages carry a vLLM denominator measured at
`555967922` with FlashInfer `0.6.15.post1`. They keep their numbers and gain a
statement of that fact and of the owed re-measurement (#2818). Marking them is
the whole difference between a published ratio and a published claim about the
current pin.

## 6. Design of the block edit

Four fields, each a measured string:

| Field | Before | After | Where it was measured |
|---|---|---|---|
| `vllm_commit` | `5559679229...bec98` | `e126687a9a...3280` | the target itself |
| `vllm_runtime_version` | `0.23.1rc1.dev1511+g555967922` | `0.28.1rc1.dev132+ge126687a9` | `IMPORT VLLM_VERSION`, `../sync/2026-09-02-e126687.md` §5.4 and `../sync/2026-09-03-e126687-runhalf.md` §5, read from `cd /` |
| `vllm_distribution_version` | `...+g555967922.precompiled` | `0.28.1rc1.dev132+ge126687a9.precompiled` | `DIST DIST_VERSION`, `../sync/2026-09-02-e126687.md` §5.4 |
| `flashinfer_version` | `0.6.15.post1` | `0.6.18` | `PIPLIST flashinfer-python`, `../sync/2026-09-02-e126687.md` §5.2 |

The `+g<sha>` constraint holds: `+ge126687a9` is a 9-character prefix of the
target commit, so `assert_oracle_commit` is satisfiable, and it is longer than the
7-character floor `tests/tools/test_oracle_pin.py` pins.

**The `.precompiled` suffix is a build-mode property and this wave records the
risk rather than hiding it.** It was measured under `VLLM_USE_PRECOMPILED=1`. The
run half's SOURCE build at the same revision produced
`vllm-0.28.1rc1.dev132+ge126687a9-cp312-cp312-linux_aarch64.whl`, with no suffix.
`online_gate.py:3529-3540` compares the distribution string for EQUALITY, so an
oracle built from source at this target will be refused on that field. The same
was true at the old pin, whose block also carries `.precompiled`, and the value
recorded here is the one that was measured — the alternative is transcribing a
string nobody read, which is the #520 failure. **Whoever runs #2818 must record
which build mode their oracle used and, if it is a source build, fix the field
from THAT measurement rather than editing it to make a run pass.**

**One OPEN discrepancy from the old pin does not carry forward, and the reason is
measured.** [#1185](https://github.com/mudler/vllm.cpp/issues/1185) recorded a
lease build reporting `0.1.dev1+g555967922` against the block's
`0.23.1rc1.dev1511+g555967922`, caused by a shallow fetch that leaves
`setuptools_scm` unable to count commits since the last tag. The target's own
build recorded `SHALLOW=false REVCOUNT=20591
GIT_DESCRIBE=v0.28.1rc0-132-ge126687a9` (`../sync/2026-09-03-e126687-runhalf.md`
§2), which is why its full string matches. The mechanism is general and still
applies to any shallow clone.

## 7. Risks

| Risk | Mitigation, or why it is accepted |
|---|---|
| Step 6 comes back red and the published ratios do not survive | Accepted by the developer ruling; §1 states that the response is to revert the pin. #2818 carries that sentence too |
| A source-built oracle is refused on `vllm_distribution_version` | §6. Recorded, not papered over; #2818 owes the corrected measurement |
| Goldens drift at the new pin and nobody knows yet | #2794. §5.1 says no gate has run rather than implying one has |
| The queue's discharge lives in an unmerged pull request | §3 names it and its state at the merge base |
| Per-file headers now lag the pin | §5.3 is the ledger note the protocol asks for |
| Somebody reads `gateable = yes` as "every row can be gated" | §5.2, and the same limit is written into `.agents/oracles/vllm.md` beside the block. The residual is real and is NOT closed: every checker reads the field, not the prose, and no field names the device or model the `yes` was measured on. The device-scoped table now says outright that it holds ZERO rows at this pin |
| The two pin surfaces silently disagree at the next advance | Not mitigated. [#2829](https://github.com/mudler/vllm.cpp/issues/2829), filed by this wave with the mutation that proves three checkers stay green over a disagreement. They agree here, verified by hand |

## 8. Gates

This wave changes no product behaviour, so its gate is the record's own
consistency. Each is a command, not a description.

1. `python3 scripts/check-oracle-pins.py` — the advanced `oracle-pin` block still
   parses, `gateable = yes` still resolves its evidence to an existing path, and
   the registry still names exactly one primary.
2. `python3 -m pytest tests/tools/test_oracle_pin.py` (or the committed runner) —
   `read_parity_pin` accepts the new block, `assert_oracle_commit` accepts the new
   runtime and distribution strings and still refuses the rollback and a
   commit-less version. These are the assertions that would catch a malformed
   advance, and they are value-agnostic by construction, which is why they are the
   right ones to cite.
3. `python3 scripts/check-symbol-anchors.py --upstream-root <pinned vllm checkout>`
   — run BEFORE and AFTER the block edit, and both counts reported. This is not a
   CI gate and it is red on this tree at both pins; the RESULT is the delta, which
   is the population of symbol citations the advance falsified. **Measured**:
   15 stale before, 28 after; 15 newly stale, 2 repaired by the advance, and one
   apparent move that is this wave's own edit relocating a citation. Thirteen of
   the fifteen are `.agents/model-matrix.md` rows, and they are THREE different
   things, not one: 8 genuine deprecations (vllm#53608), 4 migrations to the
   Transformers fallback (`48d7132962`), and 1 relocation (`aeeb36b1f1`
   vllm#50000). **Two of them reach models this project SHIPS** —
   `KimiLinearForCausalLM`, relocated and still registered, and
   `Olmo3ForCausalLM`, now routed to the generic fallback, which the anchor
   checker cannot even see. The remaining two of the fifteen are
   `moe-semantics.md:29` and `gdn-state-kv-budget.md:101`. All of it is
   [#2819](https://github.com/mudler/vllm.cpp/issues/2819).
   `../sync/2026-09-03-e126687-advance.md` §4.4 and §4.4a.
4. `scripts/agent-preflight.sh --staged`, run ONCE on the frozen tree, with the
   `gate(s) failed` and `NOT a green` lines read rather than the exit status.

## 9. Stop conditions

- **Do not run anything on a device.** No lease is held and none is requested.
- **Do not port a PORT-NOW entry**, and do not land the #2649 predicate change.
- **Do not edit an evidence file** under `docs/bench-evidence/` or a sync report
  of a past wave. They record what was true when they were written.
- **Do not weaken an assertion to make the new block pass.** If
  `assert_oracle_commit` or a harness check refuses the advanced values, the
  values are wrong or the measurement is missing; report it and stop.
- **Do not merge.**

## Owed

- [#2818](https://github.com/mudler/vllm.cpp/issues/2818) — step 6, post-hoc at
  the new pin: five rows, `C1c`, and a red means revert.
- [#2794](https://github.com/mudler/vllm.cpp/issues/2794) — the declared
  token-exact gate at the target, on `dgx:gpu0`, which also answers the OPT
  golden that was never re-validated at the old pin.
- [#2611](https://github.com/mudler/vllm.cpp/issues/2611) — a reading on
  `dgx:gpu0`, the 290-entry queue, and with it the >= 177 per-file headers §5.3
  leaves behind the pin.
- [#2626](https://github.com/mudler/vllm.cpp/issues/2626) — `qwen4_exp` does not
  run on this fleet at the pinned revision.
- [#2649](https://github.com/mudler/vllm.cpp/issues/2649) — unblocked by this
  merge, and until it lands the tree disagrees with its own pin at
  `include/vllm/config/speculative.h`.
- [#1185](https://github.com/mudler/vllm.cpp/issues/1185) — the shallow-fetch
  version discrepancy, still open as a mechanism.
- [#2829](https://github.com/mudler/vllm.cpp/issues/2829) — the pin is
  transcribed in `.agents/upstream-sync.md` and `.agents/oracles/vllm.md` and
  nothing checks the two agree; three checkers stay green over a deliberate
  disagreement. Pre-existing, made live by this advance, filed by this wave,
  owned by no row yet.
- [#2819](https://github.com/mudler/vllm.cpp/issues/2819) — fifteen citations go
  stale at the new pin. Thirteen are `.agents/model-matrix.md` rows, split 8
  deprecations / 4 Transformers-fallback migrations / 1 relocation, and **two of
  them reach models this project ships** (`KimiLinearForCausalLM`,
  `Olmo3ForCausalLM`), which makes those mirror obligations rather than matrix
  cleanup. Two more are `moe-semantics.md:29` and `gdn-state-kv-budget.md:101`.
  The pinned registry is unreconciled in both directions. Filed by this wave,
  owned by no row yet, and listed here as AGENTS.md §"Every change starts from an
  issue" requires.
