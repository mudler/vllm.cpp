# Re-derive the Nemotron oracle golden under a NAMED configuration

**Issue:** [#1694](https://github.com/mudler/vllm.cpp/issues/1694)
**Parent defect:** [#926](https://github.com/mudler/vllm.cpp/issues/926), which
this row does NOT discharge -- see `## Owed`
**Row:** `GATE-NEMOTRON-GOLDEN-REDERIVE`
**Blocked by:** [#1431](https://github.com/mudler/vllm.cpp/issues/1431), the
oracle's engine start-up collapse on `dgx:gpu0`
**Predecessor:** [`nemotron-oracle-golden-provenance.md`](nemotron-oracle-golden-provenance.md),
which committed the generator and the contract and left this under its `## Owed`
**Index row:** `#1694` in [`issue-index.md`](../issue-index.md). #926 is NOT
appended again there: [PR #1432](https://github.com/mudler/vllm.cpp/pull/1432)
is already adding a #926 row, and two branches appending one issue is the
duplicate `check-agent-record.py` reports rather than a merge.
**Authority:** the developer explicitly ratified re-deriving this golden under a
named engine configuration (2026-08-22). A gate-reference change is a product
decision and it has been made. This row executes it; it does not re-argue it.

## 0. Headline, and the thing this row does NOT do

The committed golden
`tests/parity/goldens/nemotron_35_lightning_greedy/oracle.json` records the
model, the revision, the sampling parameters and the library versions and **not
one engine knob**, and its generator was never committed. That is #926. This row
produces a **second** golden, captured by the committed generator under the
named configuration `nhspeed-a`, which records the configuration it ran under
and can therefore be regenerated and attributed.

**Re-deriving does NOT tell us what [#1289](https://github.com/mudler/vllm.cpp/pull/1289)
scores.** #1289 is worth 7.28x on the warm basis per output token and is the
campaign's only measured speed win, and it is held DRAFT on a 95/96 against the
unattributable golden. What this row produces is a reference that can be
regenerated and attributed. **What #1289 scores against it is unknown and could
go either way**, and nothing in this row, its artifacts or its pull request may
be worded to imply otherwise.

Nor does this row **repoint** any gate. §5.3 says why, and names what repointing
would take.

## 1. The decisive measurement this row is built on

Three runs of this checkpoint exist under a recorded configuration, and they
disagree on exactly one prompt:

| Run | Configuration | Prompt 0 | Prompt 1 | Prompt 2 |
|---|---|---|---|---|
| 2026-08-18 `oracle_only.sh` attempt `a` (`nhspeed-a`), twice in one process | `mml=512, mns=8, gmu=0.30, mnbt=512, enforce_eager=False`, one `TokensPrompt` per `generate()` | 32/32 | 32/32 | **26/32** |
| the #926 rebuild | `enforce_eager=True, gmu=0.25, mml=4096` | 32/32 | 32/32 | **29/32**, diverging at index 29 |
| #1289's device Mamba arm (ours, not the oracle) | — | 32/32 | 32/32 | **31/32**, diverging at generation position 32 |

**Prompt 2 has never been reproduced by any nameable configuration.** Prompts 0
and 1 have been, twice, under two different recorded configurations. #1289's
single divergence sits at **generation position 32** -- the last token of the 32,
0-based index 31 -- on **the one prompt both other oracle-side runs also fail**.
WHERE inside the 32 tokens those two runs miss is not something `main` records,
so the overlap is established at prompt granularity and no finer. That is still
the fact that makes the 95/96 unreadable: nobody can say
whether the moved token is a defect in our arm or a property of a configuration
nobody recorded.

**Where that position comes from, and how firm it is.**
[#1388](https://github.com/mudler/vllm.cpp/issues/1388), which filed the 95/96,
does **not** name a position -- its own next step is "read the `got:`/`exp:` ids
for the mismatching row". The position is named in
[PR #1432](https://github.com/mudler/vllm.cpp/pull/1432) §10.5, which is **open
and unmerged**, so it is cited here as a claim in flight rather than as something
`main` records. The #926 rebuild's index 29 comes from the same section. Neither
number is re-derived here, and this row does not depend on either: what it depends
on is that prompt 2 is the prompt all three disagree on, which every recorded run
shows on its own. It deliberately does not depend on WHERE in prompt 2 they
disagree, because only the in-flight #1432 numbers say that.

Two configurations, each internally repeatable, disagreeing on the same prompt is
**configuration sensitivity, not non-determinism**. AGENTS.md admits a ratified
distributional gate only where the oracle's greedy decode is non-deterministic,
so a distributional gate stays inadmissible here. What is licensed, and what the
developer ratified, is re-deriving under a name.

## 2. The configuration, by name, and why it is this one

**`nhspeed-a`** — `max_model_len=512`, `max_num_seqs=8`,
`gpu_memory_utilization=0.30`, `max_num_batched_tokens=512`,
`enforce_eager=False`, `num_gpu_blocks_override` unset. It is the sole entry in
the generator's `PROFILES` table (`scripts/nemotron-h-oracle-capture.py`,
symbol `PROFILES`).

Four properties, and each one is a measurement rather than a preference:

1. **It has a full resolved engine configuration on record.** The 2026-08-18 log
   is readable at `/mnt/nas_share/rc/nhspeed/oracle.a.out` (the worker's
   `/workspace/nhspeed`), and the driver that produced it is at
   `/mnt/nas_share/rc/nhspeed/oracle_only.sh`. "Whatever the script did" is not
   a name; this is a name because someone else can read what it resolved to.
2. **It is the only configuration on this checkpoint with determinism
   evidence** — two legs in one process, byte-identical, `ORACLE TOKEN MATCH:
   180/192`.
3. **It is the only configuration that has ever completed engine start-up on
   this box.** #1431's five failures include one that copied it, so this is a
   necessary condition and not a sufficient one.
4. **CUDA graphs stay ON.** `--enforce-eager` is never the denominator, and the
   #926 rebuild's configuration is eager.

It is a **token**-golden configuration and **not** a speed denominator.
`max_num_batched_tokens=512` against the denominator's 8192 is a regime you
cannot tune down and keep a ratio through.

**It is a name for a run that happened. It is not a reconstruction of the lost
one**, and the fact that it produces 26/32 on prompt 2 is what makes that
visible rather than what makes it suspect.

### 2a. The one deviation, named

`--capture` submits **one text prompt per `generate()` call`**; the 2026-08-18
run submitted **one pre-tokenized `TokensPrompt` per `generate()` call**. The
generator refuses `--capture --tokens-prompt` on purpose — `prompt_token_ids`
only come from a golden, and a capture that reads its prompts out of the artifact
it is replacing is circular.

The difference is confined to the tokenizer step and it is **checkable**: the
captured golden records `prompt_token_ids`, so if those are identical to the
committed golden's the engine received the identical token sequence and the
submission shape differed only in who tokenized. §7 records the comparison.
Either way the shape is written into `capture.batch.shape`, which is the field
that exists so this cannot be silent.

## 3. #1431, and why this is worth attempting now

Five previous attempts died in engine **start-up**, killed by a host-memory
watchdog before generating a single token. Minima against a 15000 MB floor:
12597 / 19433 / 19797 / 13941 / 14846 MB. Runs 2 and 3 used a **20000 MB** floor,
so their outcome at 15000 is inferred and not measured.

Already excluded by those runs, and **not re-derived here**:
`torch.compile` (run 5 hit the AOT cache, 0.30 s), CUDA graph capture (run 2 was
eager), and KV sizing (run 3 overrode the block count to 8, run 5 set an absolute
`kv_cache_memory_bytes`). What remained was the first forward.

What has changed: every one of those runs was on a box at load 100-150 with disk
near 100%. **The fleet is idle** — `dgx:gpu0`, `thor:gpu0` and `orin:gpu0` all
`ready`, no holder — and the box reports 2.4 TB free. That is a different
condition, and this row **re-establishes it by measurement** rather than assuming
it either way: §7 records `uptime`, `free -m`, `df`, the boot id and
`nvidia-smi --query-compute-apps` from inside the job.

**If it still cannot start, that is a legitimate result** and this row reports
#1431 confirmed under idle conditions, which is a stronger statement than the
existing evidence. **The watchdog floor is not lowered**, and no artifact of this
row may lower it: this box OOM-**reboots** rather than OOM-kills, and a reboot
takes down every other job on the fleet.

## 4. Scope

**In scope**

- One `rc` lease on `dgx:gpu0`, running the committed generator's `--capture`
  mode under `--profile nhspeed-a --legs 2`.
- The re-derived golden, committed **beside** the existing one.
- Extending the Python contract suite to hold **every** golden in the directory
  to the contract, read with a glob, so the new artifact is reached by a gate.
- The evidence, the comparison against the committed golden, and the verdict on
  whether #926 is discharged.

**Out of scope**

- Repointing `tests/vllm/models/test_nemotron_h_loader.cpp`, the A3 driver or
  #1289's score at any golden. §5.3.
- Root-causing #1431. This row measures whether it still fires; it does not fix
  it.
- The index-29 top-2 margin ([#1388](https://github.com/mudler/vllm.cpp/issues/1388)).
- Deciding whether #1289's moved token is a defect.

## 5. Design

### 5.1 The generator, not a hand-rolled driver

`scripts/nemotron-h-oracle-capture.py --capture`, already committed by the
predecessor row. It is used rather than replaced because it does three things a
hand-rolled driver would have to be trusted to do:

- **It asserts the pin's identity before anything else** and aborts otherwise
  (`assert_oracle_identity`). `$HOME/venvs/vllm-oracle` on this box has resolved
  to a 0.25.0 rollback that predates `NemotronHMoEDecoderLayer`, and a run
  through it fails in a way that reads as "the model is unsupported".
- **It reads the configuration back OUT of the built engine**
  (`read_resolved_config`), so what is written down is what vLLM ran and not what
  a driver passed. `kv_cache_dtype`, the block size, the block count and the
  backends are all chosen by vLLM.
- **It refuses to write** a golden that fails its own contract, or whose legs
  disagree. A golden written from disagreeing legs records a coin flip.

Two values it cannot read are supplied by the job and then **checked against the
run's own log**: `attention_backend` and `moe_backend`. At this pin
`VLLM_ATTENTION_BACKEND` and `VLLM_FUSED_MOE_BACKEND` do not exist — setting
either logs `Unknown vLLM environment variable` and changes nothing — and vLLM
does not expose the selection on the config, so the generator takes what the
caller says the startup log said. The job passes `FLASHINFER` and `MARLIN`, which
is what the 2026-08-18 run logged, and then greps **this** run's log for
`Using FLASHINFER attention backend` and `Using 'MARLIN' NvFp4 MoE backend`. A
zero count means the golden's backend fields are false and the artifact is not
committed.

### 5.2 The old golden is PRESERVED

`oracle.json` stays **byte-for-byte unchanged**, sha256
`659c26bd2301317d4a6999df0b7afc3243dcff129de89abcb66b46817dd6f9e9`. The new one
lands beside it as `oracle.nhspeed-a.json`.

Three reasons, and the first is the binding one:

1. **Deleting evidence to make a record tidy is never the repair.** If the two
   disagree on prompt 2, **that difference is the finding**, and it is only
   legible while both files exist. Overwriting in place would leave a reader with
   one file, a diff in history, and no statement of which configuration produced
   which.
2. The committed golden is what every current consumer reads and what #1289's
   95/96 was scored against. Moving that reference silently, in the same change
   that produces the replacement, would make the score change and the reference
   change indistinguishable.
3. The contract already admits exactly two states and both files can state
   theirs: `oracle.json` says `engine_config_recorded=false` and argues why;
   `oracle.nhspeed-a.json` says `true` and carries all twenty resolved keys.

### 5.3 Nothing lands dead: what reaches the new golden

A golden nothing reads is dead. `tests/scripts/test_nemotron_h_oracle_capture.py`
pins `SHIPPED_GOLDEN` to `oracle.json` by name, so a second file in that
directory would be gated by nothing.

The repair is a **glob**, not a second hard-coded constant: the suite holds
**every** `*.json` under `tests/parity/goldens/nemotron_35_lightning_greedy/` to
`check_golden`, so this artifact and every future capture are reached without any
change writing a shared list. That is the record-shape rule AGENTS.md states —
one file per row, read with a glob — applied to goldens.

Proven by mutation rather than asserted: gut the new golden's `capture` block and
the suite must red **naming that file**. §7 records it.

**A glob over the contract is not enough, and the fresh review of PR #1703
measured why.** `check_golden` asks whether a golden records the configuration
it was captured under. Nothing in that contract is tied to a model, a checkpoint
or a battery, so a contract-VALID file naming a completely different checkpoint,
carrying one entry instead of three and the prompt `"Write a haiku about ducks"`,
passed the glob and `--check` printed `0 problem(s)` over it. The suite now runs
`identity_problems(doc, reference)` beside the contract on every golden it finds:
`model`, `revision`, the prompt battery in ORDER, and `prompt_token_ids` must all
match `oracle.json`'s. The last of those is §2a's own check — it is the field
that says the engine received the identical token sequence despite `--capture`
submitting text where the 2026-08-18 run submitted `TokensPrompt`s — so a
difference there reds and gets read, rather than landing under a green. The guard
is reference-relative rather than a second table of literals, and the reference
is anchored by its own cases: the battery to `capture.PROMPTS`, the revision to
`capture.CHECKPOINT_REVISION`.

**What is NOT reached, and is owed.** `test_nemotron_h_loader.cpp` and the A3
driver still read `oracle.json` only. Repointing them is a change to what the
token gate compares against, which needs its own fresh review and its own
operator gate run, and it should be taken **after** somebody has measured what
#1289 scores against the new reference. It is listed under `## Owed`.

## 6. Gates

Contract, no GPU, no vLLM, no checkpoint:

```sh
python3 scripts/nemotron-h-oracle-capture.py --check \
  tests/parity/goldens/nemotron_35_lightning_greedy/oracle.json
python3 tests/scripts/test_nemotron_h_oracle_capture.py
scripts/agent-preflight.sh --fail-on-skip
```

This list no longer names `oracle.nhspeed-a.json`. It did, and the file is not
committed (§7.10), so the command would have failed on an absent path and read
as a gate failure rather than as the decision it is waiting on. The suite reads
the goldens directory with a GLOB, so it picks the file up on the commit that
lands it without this list changing again.

The capture itself is gated by the generator: `--capture` runs `check_golden` on
the document it built and raises rather than writing when it fails. It did so on
`20260823T021635Z` and printed `0 problem(s)` with `engine_config_recorded=True`
(§7.9).

## 7. Evidence

**Measured on** `mudler-ubuntu-box` (x86_64, 20 cores) in the worktree
`/home/mudler/_git/vllm.cpp-golden-rederive`, branch
`row/GATE-NEMOTRON-GOLDEN-REDERIVE`, base **pinned** at `2f2a709253c60599f`
after the second merge (`8540a27558813faef` before it), **nothing overlaid**.
The GPU side ran on `dgx:gpu0` (NVIDIA GB10, sm_121,
driver 580.173.02, aarch64) inside an `rc` lease, never over `ssh`, submitter
`claude/mudler-ubuntu-box/golden-rederive`. Job artifacts on the share under
`/mnt/nas_share/rc/golden-rederive/` (the worker sees `/workspace/golden-rederive`).

### 7.1 The contract gates, and the count beside every green

Measured at the review-repair commit, whose suite file is
sha256 `9a4407f77e73627bd5adfbc09b6002e2ab18f9a7e190567802a89e14b15f3af8` --
named by content rather than by commit id because the branch was rebuilt onto a
newly pinned base after the measurements and the commit id moved while the file
did not. The record commit after it touches neither the suite nor the golden, so
every figure below still reads at the branch head. They all MOVED with the
repair, so the pre-repair numbers this section used to carry -- 43 cases, and
`RANGE_COUNT=3` against `8540a27558813faef` -- are superseded rather than
re-stated. An evidence table has to name the tree it was measured on.

```
python3 scripts/nemotron-h-oracle-capture.py --check .../oracle.json   -> 0 problems, rc=0
python3 tests/scripts/test_nemotron_h_oracle_capture.py               -> 54 cases, OK
```

`scripts/agent-preflight.sh --fail-on-skip`: **zero gates skipped**, and
`commit-trailers` and `commit-style` both report `ok` against the pinned base
`2f2a709253c60599f` at **`RANGE_COUNT=7`**. `RANGE_COUNT=3` was 4 when it was
written -- `git rev-list --count 8540a2755..HEAD` read 4, not 3 -- and the
figure is now recounted at the head it is recorded on, which is the only place
a count means anything.

The count is not decoration. Both checkers were run against an EMPTY range
(`HEAD..HEAD`) and **returned rc=0** -- a gate that examined nothing prints the
same green as a gate that examined everything -- so the range width is reported
beside the result rather than inferred from it. The first preflight of this
branch is the other half of that lesson: it SKIPPED both gates because
`origin/main` had moved and was no longer an ancestor, and a skipped gate
reported nothing about this tree. The merge commit exists to make them run --
and it had to be made TWICE, because `origin/main` moved again while the branch
was in review and put the branch behind for the second time. `agent-preflight.sh`
takes its `TRAILER_BEHIND` arm on exactly that condition, so the merge is not
housekeeping: without it both trailer gates report nothing and the run still
looks green. `git merge-base --is-ancestor origin/main HEAD` is the check, and
it returns 0 at the head this section records. The base is PINNED by SHA
rather than by ref name, because `origin/main` moved a THIRD time mid-repair --
another worktree in this shared checkout fetched and the ref advanced from
`08c81a892` to `2f2a70925` under a merge already resolved against it.

**Three reds, and none of them is this row's.**

`test_cpu_x86_llamacpp_floor` fails on
`test_a_contended_leg_is_discarded_and_never_summarised`, reporting
`NO_QUIET_WINDOW`. That is
[#618](https://github.com/mudler/vllm.cpp/issues/618) by its exact case name --
the harness is load-dependent and this box was carrying other sessions' builds
at `load average: 60.87`, which is also why the earlier run of this branch saw a
second case of the same suite fail and this one did not.
`git diff --stat origin/main -- tests/scripts/test_cpu_x86_llamacpp_floor.py
scripts/cpu-x86-llamacpp-floor.sh` is **empty**, against a positive control on
this row's own files that is not, so the suite is byte-identical to `origin/main`.

`check-agent-record` and `test_agent_record` fail on the same one error:
`.agents/issue-index.md: issue #1649 listed twice`. Both `#1649` rows are on
`2f2a70925` already -- `git show origin/main:.agents/issue-index.md | grep -c
'^| \[#1649\]'` is `2`, added by `a7bb3130b` and `2f2a70925` -- and the three
rows this branch adds are `#1694`, `#1729` and `#1730`, which
`git diff origin/main -- .agents/issue-index.md` shows as the only additions.
`main` is red on this gate, and it is filed as
[#1731](https://github.com/mudler/vllm.cpp/issues/1731). The repair is not
available to this branch anyway: the file is append-only, so deleting a row is
the one edit its own rule forbids.

### 7.2 The re-derived golden is REACHED, proven by mutation

`SHIPPED_GOLDEN` named `oracle.json` alone, so a golden captured beside it would
have been gated by nothing. Every mutation below was applied alone, on the head
named in §7.1, and the whole table was **re-measured there**: the review repair
edited the very case M-B deletes, which silently disarms a mutation proof taken
before it.

| # | Mutation | Proof it applied | Result |
|---|---|---|---|
| — | baseline | — | **54 cases, `OK`** |
| M-A | a second golden in the directory carrying the `af8170154` shape (no `capture` block) | `git status --short` shows the untracked file | **1 failure**, and the subTest NAMES the file: `(golden='oracle.MUTANT.json')`, `missing 'capture'` |
| M-B | the same mutant golden, and the glob CASE deleted instead | `git diff --stat` **27 deletions**, `parse_rc=0` | **53 cases, `OK`** — nothing else in this tree holds it |
| M-G | a CONTRACT-VALID second golden of a DIFFERENT model: bogus `revision`, ONE entry, prompt `"Write a haiku about ducks"` | `git status --porcelain` shows the untracked file; `--check` reads it and prints `1 golden entry` | **before the repair: 54→43 cases `OK`, `--check` `0 problem(s)`.** After: **1 failure** naming all four differences |
| M-1 | M-G present, and the `identity_problems` CALL SITE deleted from the glob loop | `git diff --stat` 1 deletion, compiles | **54 cases, `OK`** — the call site is what makes the glob reach the guard |
| M-2 | M-G present, and `identity_problems` neutered to `return []` | `git diff --stat` 1 insertion, compiles | **7 failures**, all in `CaptureIdentityTests` |

M-B is the reachability question rather than the contract question. M-G is the
review finding that motivated the repair: it is what a glob gating provenance
SHAPE cannot see, and M-1 and M-2 are the two halves of its fix — that the guard
is REACHED from the glob, and that the guard itself is gated by cases rather than
by a mutation somebody has to remember to repeat.

**M-A also exhibits the `--check` labelling defect**, in its own evidence line:
the subTest says `oracle.MUTANT.json` and the contract message says
`oracle.json: missing 'capture'`, because the top-level `where` label is a string
literal. That is [#1729](https://github.com/mudler/vllm.cpp/issues/1729), listed
under `## Owed`; it is pre-existing and not repaired here.

Restored byte-for-byte after every mutation: suite sha256
`9a4407f77e73627bd5adfbc09b6002e2ab18f9a7e190567802a89e14b15f3af8` at the head
named in §7.1, `git status --porcelain` and `git diff` both clean over
`tests/`, and the committed golden's sha256 unchanged at
`659c26bd2301317d4a6999df0b7afc3243dcff129de89abcb66b46817dd6f9e9`.

### 7.3 The capture shape was checked BEFORE spending a lease

A contract failure after a twelve-minute engine start costs a lease, so the
document `--capture` will build was constructed offline from the 2026-08-18
`nhspeed-a` log's own resolved values and run through `check_golden`:
**0 problems**. Dropping each of the twenty engine keys in turn, the contract
named the one dropped **every time** -- no key is decorative. In particular,
removing `attention_backend` or `moe_backend` is refused, which is why the job
passes both and then re-reads them out of the run's own log.

### 7.4 The BOX condition, measured twice, and it is not what the record assumed

`rc devices` reported `dgx:gpu0` `ready` and free before both submissions. It
says who holds the GPU. **It says nothing about what the host is doing**, and on
this box the two came apart:

| Lease | `uptime` load average | `MemAvailable` at start | Swap used |
|---|---|---|---|
| `20260822T150416Z` | **92.58, 135.52, 93.72** | **10464 MB** | 14163 MB |
| `20260822T152834Z` | **92.85, 85.02, 64.53** | **5031 MB** | 17354 MB |

`MemTotal` is 122502 MB and this job's own watchdog floor is 15000 MB, so **both
leases began BELOW the floor**. `/proc/pressure/memory` on the second read
`full avg60=83.64`. `nvidia-smi --query-compute-apps` was empty both times, so
the GPU was genuinely free and the pressure is host-side; `ps` inside the job
container sees only its own five PIDs, so whatever holds the other ~117 GB is
outside this job.

**This falsifies the premise the row was dispatched under.** The fleet was not
idle. The box sat in the same load band (100-150) that
[#1431](https://github.com/mudler/vllm.cpp/issues/1431) records for all five of
its failures, and other campaigns (`fp8-gate`, `ltx25-pixel-ab`) were queued on
it throughout.

The first lease was **killed rather than run**. A capture started at
`MemAvailable=10464` would have been shot by its own watchdog within a second
and the result would have read as `#1431 confirmed` -- a contended host wearing
the shape of a verdict about the oracle.

### 7.5 The quiet gate, and both of its arms

The second submission carries a precondition that measures the condition and
**exits 93 on timeout rather than proceeding**, because a wait-for-quiet loop
that times out and runs anyway while printing its success label is a failure this
repository has already had.

Thresholds are derived, not chosen. `MIN_AVAIL_MB=60000`: the one control
(2026-08-18 `nhspeed-a`) consumed 38 GB and the floor is 15 GB, so 53 GB is the
arithmetic minimum for that run to fit, and 60 GB is that with margin -- and it
is deliberately BELOW the control's own 90274 MB start, because a gate set at the
control's exact start might never open. `MAX_LOAD1=20`: `rc describe dgx:gpu0`
reports `cpus=20`, so `load1 <= cpus` is "not oversubscribed".

Falsified on five arms before it was trusted, by feeding the loop synthetic
`/proc` readings:

| Input | Expected | Result |
|---|---|---|
| avail 90000 MB, load1 1.50 | pass | `QUIET GATE PASSED`, rc=0 |
| avail 10464 MB, load1 92.58 (the real 15:04 condition) | refuse | `PRECONDITION_NOT_MET`, **rc=93** |
| avail 117417 MB, load1 135.52 (#1431's own condition) | refuse | `PRECONDITION_NOT_MET`, **rc=93** |
| avail 51528 MB, load1 2.00 | refuse | `PRECONDITION_NOT_MET`, **rc=93** |
| avail 60000 MB, load1 20.00 (both boundaries exactly) | pass | `QUIET GATE PASSED`, rc=0 |

The last arm is what stops it passing by refusing everything. The harness needed
one repair of its own first: its initial version extracted the loop from the
job script by string index and picked up the COMMENT rather than the code, and
all four arms then "failed" with a shell syntax error -- an instrument failure
that would have read as the gate working.


### 7.6 Five leases, and the wall moved every time

| Run | Driver | Wall | What it is evidence of |
|---|---|---|---|
| `20260822T150416Z` | `job.sh` | killed before starting; `MemAvailable_at_start_MB=10464`, already below this job's own floor | nothing about the oracle |
| `20260822T152834Z` | `job.sh` | quiet gate refused, six flat samples at ~5340 MB over 100 s, rc=93 | the host held ~117 GB with an idle CPU ([#1710](https://github.com/mudler/vllm.cpp/issues/1710)) |
| `20260823T014118Z` | `job.sh` | watchdog kill at 14813 MB, `CAPTURE_RC=137`, `peakUsed_MB=102948` | the first run to reach the engine |
| `20260823T020902Z` | `job.v2.sh` | watchdog kill at 14644 MB, `CAPTURE_RC=137`, `peakUsed_MB=102719` | the run that NAMED the cause (§7.7) |
| `20260823T021635Z` | `job.v3.sh` | **`CAPTURE_RC=0`**, `peakUsed_MB=53053`, both legs agree, golden written | the capture (§7.9) |

The first two are recorded in §7.4 and §7.5 and are unchanged. The rest are new.

**`20260823T014118Z` -- the box was idle and it still died.**

```
MemAvailable_at_start_MB=117761   MemTotal_MB=122502   GPU free, quiet gate passed on sample 1
IDENTITY ASSERTED: 0.1.dev1+g555967922
Using 'MARLIN' NvFp4 MoE backend      (vLLM's own line, count 1)
Using FLASHINFER attention backend    (vLLM's own line, count 1)
01:49:17  Loading weights took 235.11 seconds        52/52 shards
01:49:21  Model loading took 17.85 GiB memory and 241.081402 seconds
01:49:39  torch.compile took 16.63 s in total        <- last line that reached the log
01:50:41  WATCHDOG: MemAvailable 14813MB < 15000MB -- killing process GROUP 49140
          CAPTURE_RC=137   peakUsed_MB=102948   samples=291
```

This **removes the confound §7.4 and §7.5 were about**. Those two leases began
below this job's own watchdog floor on a host carrying ~117 GB; this one began
with 117761 MB of 122502 MB free and the GPU genuinely idle. It is the first run
of [#1431](https://github.com/mudler/vllm.cpp/issues/1431)'s population to reach
the engine at all, and both backends are vLLM's own log lines rather than the
job's claim about them. No golden was written, verified with `ls` on the run
directory rather than inferred from the exit path.

It also **narrows the phase to a single call**. In the AOT path
`vllm/compilation/decorators.py:660-667` logs `torch.compile took %.2f s in
total` when compilation finishes, and `:669-670` then runs
`self.aot_compiled_fn(...)` under `monitor_profiling_run()`, which logs `Initial
profiling/warmup run took %.2f s` on exit (`vllm/compilation/monitor.py:81-84`).
Between those two lines there is exactly one call: the profiling forward inside
`profile_run()` (`gpu_worker.py:491-494`). The run printed the first and never
the second. Its 1 Hz sampler puts the whole descent there -- 93673 MB at
01:49:40, 14813 MB at 01:50:36, monotone at ~1.4 GB/s -- while the 235 s of
weight loading before it was flat.

**What it was NOT evidence of: a cause.** `mem.samples` measures the BOX. Three
runs had now reported that the box lost memory and none of them could say to
whom, and the row was one inference away from writing that down as an engine
defect. §7.7 is what asking the question instead produced.

**`20260823T020902Z` -- `job.v2.sh`, same configuration, instrumented.** Four
additions, all evidence and none of them able to move a token: `PYTHONUNBUFFERED=1`
(the previous run was SIGKILLed with its last lines still in a pipe buffer),
`pip freeze` into the run directory, a SEPARATE 2 s attribution sampler, and the
phase greps. The watchdog loop was left byte-for-byte alone -- the sampler is a
separate process precisely so the 1 s poll keeps its period. It died at the same
place, 102719 MB against 102948 MB, and it caught the cause doing it.

The stack is now on record for the first time, which no previous run of this row
or of the 2026-08-18 control ever wrote down: `torch==2.13.0`,
`triton==3.7.1`, `flashinfer-python==0.6.15.post1`, `transformers==5.15.1`,
`numpy==2.3.5`, 195 lines in `pip-freeze.txt`.

### 7.7 The cause, measured: it was never an engine allocation

`mem.samples` says the BOX lost 79 GB during the profiling forward. It cannot
say who took it, and for three runs nobody asked. job.v2.sh added a second
sampler at 2 s -- every process over 256 MB RSS, and
`nvidia-smi --query-compute-apps` beside it -- and `20260823T020902Z` caught the
answer in the act:

```
1787451296 RSS 53800 1327276 cudafe++
1787451296 RSS 53816 1323608 cudafe++
1787451296 RSS 53879 1302148 cicc
1787451296 RSS 53881 1288500 cicc
1787451296 RSS 53902 1115744 cicc
1787451296 RSS 53890  882120 cicc
1787451296 RSS 53919  635916 cicc
1787451296 GPU 51726, 18610 MiB
```

`cicc` is NVVM, `cudafe++` is nvcc's front end. Over the whole run
(`proc.samples`, 612 compiler samples: 432 `cicc`, 180 `cudafe++`):

| | |
|---|---|
| peak CONCURRENT nvcc pipelines | **22** |
| peak TOTAL compiler RSS | **80789 MB** |
| peak RSS of the vLLM `EngineCore` process itself | **2197 MB** |
| peak device memory `nvidia-smi` attributed to that pid | **21436 MiB** |
| `peakUsed_MB` the watchdog measured | **102719** |

**Those four numbers close the account.** 21436 MiB of device-resident weights
(the 17.85 GiB the loader reports, plus context and allocator overhead) plus
2197 MB of engine host RSS plus 80789 MB of compilers is ~104 GB against a
102719 MB peak. The vLLM process was never big. **The 79 GB was nvcc**, running
one job per CPU on a 20-CPU box, and the device side never moved during the
descent -- `nvidia-smi` sat flat and then FELL as the kill landed.

This is FlashInfer's JIT build. The pin selects `FLASHINFER` attention
(`cuda.py:482` in the run's own log) and FlashInfer compiles its kernels on
first use, which is the first forward -- the profiling forward. job.sh's own
step 1c already knew this ("FlashInfer JIT-BUILDS sampling and needs
`curand.h`"); what nobody had measured is that the build's PARALLELISM, not the
model, is what takes the box down.

**Two things follow immediately.**

The 2026-08-18 control's 180 s `Warming up Mamba2 SSD Triton kernels...` window
consumed no memory because Triton compiles in-process; the nvcc fan-out is a
different mechanism, and why the control did not pay it is a question for #1431
and not for this row.

And a watchdog kill lands in the MIDDLE of the build, so it leaves no usable
cache: `20260823T014118Z` and `20260823T020902Z` each paid the full compile from
scratch, which is why the second one recompiled 28 minutes after the first. A
run that COMPLETES the build should leave the next one nothing to do.

### 7.8 Why no engine knob could have contained it, and why there is no `nhspeed-b`

The obvious question about `20260823T014118Z` is why `gpu_memory_utilization=0.30`
did not contain it: 0.30 of 122502 MB is ~36.7 GB and the process took 102948 MB.
§7.7 answers it in one line -- the memory was nvcc's, and no engine setting
governs a compiler subprocess. The source says the same thing from the other
side, and it is worth recording because the next reader will reach for these
knobs too. Read at the pin `5559679229bc961848b121ccdeaa8fa5d79bec98`:

**`gpu_memory_utilization` is a KV-cache BUDGET, not a cap.** It is used in
exactly two places. `vllm/v1/worker/utils.py:393-413` computes
`requested_memory = ceil(total_memory * gpu_memory_utilization)` and only
VALIDATES it -- it raises if `free_memory < requested_memory`, and otherwise
returns the number. `vllm/v1/worker/gpu_worker.py:387` calls that once at worker
init, and `:535-539` subtracts the profiled non-KV memory from it AFTER the
profiling forward has run. There is no arena and no allocator limit between
those two points. Setting it too low yields a small KV cache, never a smaller
forward pass.

**`kv_cache_memory_bytes` does not skip the forward.**
[#1647](https://github.com/mudler/vllm.cpp/issues/1647)'s `## Owed` records that
`--kv-cache-memory` bounds only the KV pool and that `--gpu-memory-utilization`
is "accepted and ignored" ([#83](https://github.com/mudler/vllm.cpp/issues/83)).
At the pin it is stronger: `gpu_worker.py:465-468` takes the
`kv_cache_memory_bytes` branch and STILL calls
`self.model_runner.profile_run()`, and says why in its own comment -- "still need
a profile run which compiles the model for max_num_batched_tokens". What the
branch skips is the `memory_profiling` accounting at `:491-494`, not the forward
inside it. `VLLM_ENABLE_STARTUP_PLAN` (`vllm/v1/worker/startup_plan.py:134-158`)
is the same lever wearing a cache: it sets `kv_cache_memory_bytes` and lands in
that branch.

**The only knob that SIZES the forward is `max_num_batched_tokens`** --
`profile_run` calls `self._dummy_run(self.max_num_tokens, is_profile=True)`
(`gpu_model_runner.py:6470-6471`) and `self.max_num_tokens` is
`scheduler_config.max_num_batched_tokens` (`:505`) -- **and it is the wrong size
to cut.** The MARLIN MoE path declares its own M-dependence in
`vllm/model_executor/layers/fused_moe/experts/marlin_moe.py` (`workspace_shapes`):
`workspace1 = (M*topk, max(N,K))`, `workspace2 = (M*topk*max(2N,K),)`. At this
model's shapes -- `topk=6`, `N=1856`, `K=2688`, bf16 -- M=512 puts the pair at
39 MB. Cutting `max_num_batched_tokens` 8x buys ~35 MB against a 79 GB overrun,
and it would have bought nothing at all, because the 79 GB was in a different
process.

**So no `nhspeed-b` was defined, and `nhspeed-a` is still the name of what was
measured.** The lever that works is `MAX_JOBS` -- how many nvcc pipelines run at
once. It is not an engine setting, it appears in none of the twenty keys the
golden records, and it cannot move a token: the same sources go through the same
nvcc with the same flags and produce the same cubins, four at a time instead of
twenty-two. AGENTS.md's own DGX profile already requires `-j 4` on this box
because unconstrained parallelism has OOM-rebooted it; this row simply found the
build that was not obeying it.

**And the configuration was never the variable, independently of all of the
above.** vLLM prints its complete resolved configuration on one line. That line
from the 2026-08-18 run that COMPLETED start-up
(`/mnt/nas_share/rc/nhspeed/oracle.a.out:14`) and from `20260823T014118Z`, which
was killed, are the same 3561 characters -- compared field by field after
splitting on top-level commas, zero differing keys, same pin, same `FLASHINFER`,
same `MARLIN`, same `FULL_AND_PIECEWISE`, same `fp8_e4m3`. One consumed 38746 MB
and the other 102948 MB. Weight loading was identical in both (17.85 GiB,
~22-23 GB of host `MemAvailable`, flat throughout); they diverge in one phase
only. A golden captured under a moved configuration would therefore have paid a
real cost -- a reference nobody could compare to that control -- to fix
something the configuration never caused.

### 7.9 The capture ran, and the lever was `MAX_JOBS`

`job.v3.sh` changes **one thing** against `job.v2.sh`, and it is not an engine
setting:

```sh
export MAX_JOBS=4          # was: one nvcc job per CPU, and this box has 20
export NVCC_THREADS=1
export FLASHINFER_NVCC_THREADS=1
```

It cannot move a token. The same sources go through the same nvcc with the same
flags and emit the same cubins; only the number of compilations running at once
changes. None of the twenty engine keys moves, so **`nhspeed-a` is still the
name of what was measured and no `nhspeed-b` exists**. `4` is not taste:
AGENTS.md's DGX profile already requires `-j 4` on this box because
unconstrained parallelism has OOM-rebooted it.

**`20260823T021635Z`**, `rc` job `84056945-3185-4745-834f-1d02635d3d64`,
`dgx:gpu0`, run directory `/mnt/nas_share/rc/golden-rederive/20260823T021635Z`,
driver sha256 `6b406f4b7d61fceea01989c174359374f56ebc6885eecf47849ad88680f6f420`.

| | `20260823T020902Z` (uncapped) | `20260823T021635Z` (`MAX_JOBS=4`) |
|---|---|---|
| peak concurrent nvcc pipelines | 22 | **4** |
| peak compiler RSS | 80789 MB | **31126 MB** |
| minimum `MemAvailable` | 14644 MB (**killed**) | **64720 MB** |
| `peakUsed_MB` | 102719 | **53053** |
| profiling forward | never returned | **`Initial profiling/warmup run took 606.42 s`** |
| `CAPTURE_RC` | 137 | **0** |

The memory trace becomes a sawtooth instead of a cliff: batches of four
compilers take ~15-30 GB and give it back, floor 64720 MB against a 15000 MB
watchdog, **and the watchdog never fired** (`watchdog.log` is empty). The JIT
build took 531 s at four-way, between `torch.compile took 16.86 s` at 02:22:47
and `Warming up Mamba2 SSD Triton kernels...` at 02:31:38. What it costs at
twenty-two-way is **unknown and cannot be stated**: no uncapped run ever
finished one. `20260823T014118Z` was killed 62 s in and `20260823T020902Z` 56 s
in, which are lower bounds on an unfinished build and not a duration. The trade
is therefore "a build that completes, against builds that did not", and the 531 s
is the only measured number in it.

Everything downstream then ran for the first time in this row:

```
02:31:38  Warming up Mamba2 SSD Triton kernels...
02:32:53  Initial profiling/warmup run took 606.42 s
02:34:33  Using triton Mamba SSU backend.
02:34:33  FlashInfer resolved query dtypes: ... decode_backend=flashinfer-native, arch=sm121
02:35:54  Available KV cache memory: 14.92 GiB
02:35:54  GPU KV cache size: 127,897 tokens
02:36:44  Graph capturing finished in 6 secs, took 0.61 GiB
02:36:45  init engine (profile, create kv cache, warmup model) took 857.43 s
          ORACLE_LEG 1 / ORACLE_LEG 2 / ORACLE_LEGS_AGREE=True
          WROTE .../oracle.nhspeed-a.json    CAPTURE_RC=0
```

Both backend claims are vLLM's own lines, counted by the job:
`attention_FLASHINFER_lines=1 moe_MARLIN_lines=1`. The artifact is sha256
`d2a59a24674470d01178f8da9c5c1d180492ad55e10afab88fccc541c44e0d40`, it passes
its own contract (`3 golden entries, engine_config_recorded=True, 0 problem(s)`),
and it carries all twenty resolved keys, read back out of the built engine:
`block_size=512`, `num_gpu_blocks=1249`, `kv_cache_dtype=fp8_e4m3`,
`dtype=torch.bfloat16`, `quantization=modelopt_mixed`, `seed=0`,
`enforce_eager=false`, `enable_chunked_prefill=true`,
`cudagraph_mode=FULL_AND_PIECEWISE`, `cudagraph_capture_sizes=[1,2,4,8,16]`,
`attention_backend=FLASHINFER`, `moe_backend=MARLIN`, `tensor_parallel_size=1`.

**This is the first time #1431's wall has been passed.** It is not a fix for
#1431 and it does not close it: what this row measured is that the wall was
nvcc's fan-out, on this box, in this job shape. Whether every #1431 failure has
that cause is not established here.

#### The tokens

`oracle.json` is the reference on the left. Both were captured on the same
checkpoint (`config.json` sha256 `f1d98b530846087dc08b574a219713a94f945bf6583dc7230a19ebf1e8c50933`,
52 shards) and `prompt_token_ids` are identical for all three prompts, so §2a's
own check passes: `--capture` submitting text where the 2026-08-18 run submitted
`TokensPrompt`s did not change what the engine received.

| Prompt | compared | matched | first divergence |
|---|---|---|---|
| 0, "The capital of France is" | 32 | **32** | none |
| 1, "Write the first five Fibonacci numbers:" | 32 | **32** | none |
| 2, "Explain what a state space model is, in one sentence:" | 32 | **31** | **index 31** |
| | **96** | **95** | |

The one differing token is the **last** of prompt 2's 32: committed `3468`,
re-derived `11286`, on `"...typically using a combination of transition"`.

**Two readings this row does NOT make.**

It does not say what [#1289](https://github.com/mudler/vllm.cpp/pull/1289)
scores. Nobody has run #1289 against `oracle.nhspeed-a.json`, and §0 and §9 bind
here: a token comparison is not a verdict on #1289. That measurement is owed.

And it does not say `nhspeed-a` reproduces. **It did not.** The 2026-08-18 run
of this same named profile read `180/192` over two legs -- prompt 2 at `26/32`,
twice -- and this one reads `95/96`, prompt 2 at `31/32`, over two agreeing
legs. Same profile name, same 3561-character engine-config line, two agreeing
legs each time, two different answers. The profile is repeatable **within a
process** and is not, on this evidence, repeatable **across runs**. The
resolved configuration is not identical either, and the generator records it
because of exactly this: 2026-08-18 resolved `128,819` KV tokens at
`15.03 GiB`, this run `127,897` at `14.92 GiB` and `num_gpu_blocks=1249`, because
that number is an OUTPUT of a memory measurement rather than an input somebody
set. Whether that is the term that moved the token is **not established here**
and must not be asserted; what is established is that a golden captured under
this profile is a record of one run, and §1's "configuration sensitivity, not
non-determinism" now has a third data point it has to account for.

### 7.10 The artifact is NOT committed, and the guard is why

`identity_problems` -- the reference-relative guard §5.3 added after the fresh
review -- **reds on this golden**, and the red is correct:

```
FAIL: test_every_committed_golden_satisfies_the_contract (golden='oracle.nhspeed-a.json')
AssertionError: ["model: '/workspace/a3/ckpt-stage' is not the committed golden's
                  '/mnt/nas_share/checkpoints/nemotron-3.5-lightning-30b-nvfp4'"]
```

Only `model` differs. `revision`, the prompt battery in order, and
`prompt_token_ids` all match, and the two paths hold the **same checkpoint**:
`config.json` sha256 is `f1d98b530846087dc08b574a219713a94f945bf6583dc7230a19ebf1e8c50933`
at both, with 52 shards at both. So this is a **path** difference, not an
identity difference -- and it is nevertheless a genuine difference in the
artifact, because `model` records the string the engine was given and the two
runs were given different strings.

**It was not made green.** The golden is not committed, the guard is not
widened, and the committed `oracle.json` is untouched at sha256
`659c26bd2301317d4a6999df0b7afc3243dcff129de89abcb66b46817dd6f9e9`. Making a red
gate green by loosening its assertion is the one repair this repository forbids,
and `model` is not decoration: dropping it from the guard is a checker-semantics
change that owes its own spec, its own red-before case and its own fresh review.

**It also cannot be fixed by re-running.** The generator records `args.model`
verbatim (`scripts/nemotron-h-oracle-capture.py:696`) and its own usage block
says `--model $CHECKPOINT_ROOT/nemotron-3.5-lightning-30b-nvfp4` (`:63`), but
that path does not exist inside an `rc` lease: the worker sees the share's `rc/`
subfolder as `/workspace`, and the checkpoints live in a sibling subfolder that
is outside it. Manufacturing the path would mean recording a location the engine
did not read, which is worse than the mismatch.

So the decision is a real one and it is **escalated rather than taken here**:
what should `model` mean in a golden -- the provenance path the engine was
given, or the checkpoint identity? Owner is listed under `## Owed`. The artifact
and its complete evidence are preserved on the share at
`/mnt/nas_share/rc/golden-rederive/20260823T021635Z/oracle.nhspeed-a.json`,
sha256 `d2a59a24674470d01178f8da9c5c1d180492ad55e10afab88fccc541c44e0d40`, so
nothing is lost while it is decided.

### 7.11 Measurement A: the two arms bracket the two values the oracle itself produced

§7.9 left the row with one differing token and no way to read it. Measurement A,
run `20260823T071352Z`, reads it. It runs the same three prompts through **four
legs**, and **each leg is scored against exactly ONE reference**: L1, L3 and L4
against the committed `oracle.json`, L2 against the newly captured
`oracle.nhspeed-a.json`. Four legs, four `TOKEN MATCH` lines, **four MEASURED
scores**. The four remaining cells of the table below are **DERIVED**, and they
are marked as such, because no run produced them.

**Evidence.** `/mnt/nas_share/rc/goldenab/20260823T071352Z/` -- `job.log` plus
one `L*.log` per leg (`L1_a2q1_device_vs_committed.log`,
`L2_a2q1_device_vs_nhspeeda.log`, `L3_a2q1_host_vs_committed.log`,
`L4_main_vs_committed.log`), and the two references the job read, staged one
level up at `/mnt/nas_share/rc/goldenab/oracle.committed.json` sha256
`659c26bd2301317d4a6999df0b7afc3243dcff129de89abcb66b46817dd6f9e9` and
`/mnt/nas_share/rc/goldenab/oracle.nhspeed-a.json` sha256
`d2a59a24674470d01178f8da9c5c1d180492ad55e10afab88fccc541c44e0d40`. The job
hashed both itself at `job.log:29-30` and both match: the first is byte-identical
to the committed `tests/parity/goldens/nemotron_35_lightning_greedy/oracle.json`,
the second to the artifact §7.10 preserves.

**The derivation is exact, and this is the fact that makes it exact: the two
goldens differ in ONE token.** Their three `golden` rows carry identical
`prompt` and identical `prompt_token_ids`, and their `token_ids` differ **only
at row 2, index 31** -- committed `3468`, re-derived `11286`. All 95 other
positions are equal. So a leg's score against the second golden is fixed by its
score against the first together with its own p2[31] value, with nothing left to
measure: a leg emitting `11286` is 96/96 against `oracle.nhspeed-a.json` and
95/96 against `oracle.json`, and a leg emitting `3468` is the mirror. That is
arithmetic over two files in the evidence directory, not an inference about the
engine -- but it is still a derivation, and reporting it as a measurement is the
"a stub's shape written up as a measurement" failure this repository has paid
for before.

| Leg | tree | `VT_NEMOTRON_H_DEVICE_MAMBA` | vs committed `oracle.json` | vs `oracle.nhspeed-a.json` | p2[31] | wall/prompt |
|---|---|---|---|---|---|---|
| L1 | a2q1 `a34e153f9` | 1 | **[M]** 95/96, diff at 31 | *[D]* 96/96 | `11286` | 44-60 s |
| L2 | a2q1 `a34e153f9` | 1 | *[D]* 95/96, diff at 31 | **[M] 96/96 PASS** | `11286` | 45-59 s |
| L3 | a2q1 `a34e153f9` | 0 | **[M] 96/96 PASS** | *[D]* 95/96, diff at 31 | `3468` | 323-337 s |
| L4 | main `8eecc05a9` | flag does not exist | **[M] 96/96 PASS** | *[D]* 95/96, diff at 31 | `3468` | 284-296 s |

**[M] = MEASURED** -- that leg ran against that golden and its log carries the
`TOKEN MATCH` line. There are exactly four. **[D] = DERIVED** -- computed from
the leg's p2[31] and the single-token difference between the goldens. There are
exactly four, and no run produced any of them. The `p2[31]` column is measured
for every leg: it is read out of the leg's own `got:` line, or out of its 96/96
against the golden that carries that value.

Binaries: a2q1 `917b40aa...`, main `c5e66d0a...`. The two builds resolve
**identical kernel feature sets** -- `features-a2q1.txt` and `features-main.txt`
are byte-identical -- so **no leg is separated from any other leg** by a
build-time capability difference. That is the within-run half of the claim and
it is fully supported.

The **cross-date** half is narrower than this section first stated it, and it is
narrowed here rather than repeated. The eight `CUDA feature` lines are
byte-identical to the 2026-08-19 reference run's
(`/mnt/nas_share/rc/a2d1/20260819T165727Z/features.txt`), so the **declared
feature set** matches across the two dates. Build capability does not follow
from that: the 08-23 configure log reads `CUTLASS found at
/root/cutlass-v4.5.0` where the 08-19 log reads `CUTLASS found at /root/cutlass`
(`configure-a2q1.log:24` against `configure.log:24`), and **the 08-19 CUTLASS
revision is recorded nowhere** -- no version string appears anywhere in that
run's directory. A degraded CUTLASS build has drifted a near-tie gate in this
repository before, so what this row claims across dates is only that the feature
LINES match, with the CUTLASS revision behind the 08-19 lines unknown.

**Every one of the four MEASURED scores is 95 or 96 out of 96, every derived
cell is its mirror, and every miss is the same token.** p2[31] is the **final**
token of prompt 2 -- 0-based index 31 of 32. On the other 95 tokens all four
legs agree with both goldens, which is the same statement as the enabling fact
above and is measured for each leg against the one golden that leg ran on.

#### The reading: it is a near-tie

The two arms are **exact mirror images**. The device arm emits `11286` and
matches `oracle.nhspeed-a.json` 96/96 while missing `oracle.json` by that one
token; the host arm emits `3468` and matches `oracle.json` 96/96 while missing
`oracle.nhspeed-a.json` by the same one token. They agree everywhere else.

And **vLLM itself has produced both values**. `3468` is what the oracle produced
in August and what the committed golden records; `11286` is what the oracle
produced on 2026-08-23 under `nhspeed-a` (§7.9). So **our two arms bracket
exactly the two values the oracle has produced** on this position, and there is
no third value anywhere in the evidence.

Three statements follow, and the row makes no fourth:

- **Neither arm is defective.** Each reproduces one of the two answers the
  reference itself has given, exactly, and nothing distinguishes those two
  answers in authority.
- **Neither capture is authoritative.** `oracle.json` cannot be regenerated or
  attributed (that is #926); `oracle.nhspeed-a.json` can, and did not reproduce
  across runs (§7.9). One of them being reproducible does not make its value at
  p2[31] the right one.
- **`origin/main` is 96/96 against the committed golden today.** L4 establishes
  it on `8eecc05a9` with the flag absent from the tree. The
  [#1388](https://github.com/mudler/vllm.cpp/issues/1388) 95/96 belongs **solely
  to [#1289](https://github.com/mudler/vllm.cpp/pull/1289)'s device arm**, which
  does not exist on `main` -- verified with a positive control, L3, which is the
  same a2q1 binary as L1 and L2 with the flag set to `0` and which scores 96/96.

§0 still binds. A near-tie is not a verdict on #1289: it says the moved token
does not separate the arms, not that #1289 passes.

#### The recommendation: do NOT re-pin `oracle.json`

Re-pinning the committed golden to `oracle.nhspeed-a.json` would move the gate
from agreeing with the host arm to agreeing with the device arm. It would not
resolve anything, because the position is undetermined -- it would **encode an
undetermined position as determined**, and it would do so in the one file every
Nemotron token claim is scored against. The recommendation of this row is
therefore that `oracle.json` stays exactly as it is, at sha256
`659c26bd2301317d4a6999df0b7afc3243dcff129de89abcb66b46817dd6f9e9`, until
something actually determines p2[31]. §9 already forbids overwriting it; this is
the measured reason.

#### A LEAD, not a cause: KV sizing is not named by the profile

Recorded as a lead and explicitly **not** asserted as the mechanism. The two
oracle runs resolved different block counts -- `num_gpu_blocks` **1258** and
**1249** -- from different `Free memory on device` readings, **86.1 GiB** and
**112.62 GiB**. On this unified-memory box vLLM reads free *device* memory out
of the *host's* available memory (`vllm/utils/mem_utils.py:148-155`, still read
rather than measured -- see `## Owed`), so **host state at start-up feeds KV
sizing**. The consequence for this row is a naming one and it is real:
**`nhspeed-a` names the engine arguments but NOT the derived block count.** Two
runs can carry the same 3561-character engine-config line, the same profile
name, and different KV geometry.

That is as far as the evidence goes. Nobody has shown that block count moves
p2[31], no run has varied block count with everything else held, and the pairing
does not decompose into any arithmetic this row can check -- `gpu_memory_utilization`
is 0.30 in both runs and the resolved KV budget landed within 0.11 GiB of itself
either way. Treating this as the cause would be the §8 failure this row already
made once.

#### A SECOND LEAD, at the same level as the first: `ignore_eos`

Recorded as a lead and explicitly **not** asserted as the mechanism, exactly as
the KV one is. The two captures do not record the same `sampling` block. The
committed `oracle.json` records `{"temperature": 0.0, "max_tokens": 32}`;
`oracle.nhspeed-a.json` records
`{"temperature": 0.0, "max_tokens": 32, "ignore_eos": true}`. `max_tokens` is
**32 in both**, so the entire delta is the `ignore_eos` key.

It is a lead because of **where** the divergence sits. p2[31] is the FINAL token
of its row, and the final token is precisely the position at which a difference
in the stop condition lands. It is also the only position on which the two
goldens disagree at all.

It is **only** a lead because the record cannot say which of two things the
difference is. It may be a **non-SETTING** -- the August capture ran without
`ignore_eos`, and `3468` is what a run able to stop produced. Or it may be a
**non-RECORDING** -- the August capture set it and never wrote it down, which is
[#926](https://github.com/mudler/vllm.cpp/issues/926)'s own subject: that
capture ran from a venv on a reimaged host, committed no generator, and cannot
be regenerated or attributed. Nothing in the evidence separates those two, and
no run has varied `ignore_eos` with everything else held. Asserting it as the
cause would be the same §8 failure the KV lead is labelled to avoid.

One further recorded difference is named here so that it is not mistaken for a
third lead: `vllm` differs as a version **STRING** --
`0.23.1rc1.dev1511+g555967922` against `0.1.dev1+g555967922` -- while the git
hash `g555967922` is **the same in both**, so the two captures ran the same vLLM
commit and the strings differ only in how each environment was installed.
(`transformers` differs as well, `5.14.1` against `5.15.1`; this row does not
read that in either direction and does not offer it as a lead.)

#### What Measurement A did NOT establish

**The direct arm proof is missing.** The intent was to confirm from the model's
own counters which arm each leg executed. All four legs printed
`NO NH-DIAG LINE`: the `[NH-DIAG] ARM step` counters exist only in an
uncommitted instrumented tree, and `grep -rn 'ARM step' src/ tests/` on this
branch exits 1. What stands in for them is **indirect**: the wall-time split is
up to 7.3x between the flag-on and flag-off legs (44 s against 323 s at the
fastest and slowest prompts), against a recorded 6.6x device-versus-host regime
for this model. That is consistent with the flag having selected the arm it
names, and it is not a proof that it did. Listed under `## Owed`.

## 8. Risks

- **The box does not start the engine.** Realised three times and then
  **resolved**: the wall was nvcc's JIT fan-out, not the model (§7.7), and
  `MAX_JOBS=4` cleared it (§7.9). The floor was never lowered.
- **A contended host gets reported as an oracle defect.** The quiet gate stopped
  this twice (§7.4, §7.5).
- **A BOX-level measurement gets reported as a CAUSE.** This is the one that
  nearly landed. Three runs reported that the host lost 79 GB and none could say
  to whom; the row was one inference away from recording an engine defect that
  did not exist. A second sampler, costing nothing, named the processes instead
  (§7.7). `mem.samples` measures the box. It is not attribution.
- **A named profile is mistaken for the recovered one.** Mitigated in §2, in the
  generator's comments and in the golden's `capture.engine.profile`.
- **The re-derived golden disagrees with the committed one on prompt 2.**
  Realised: 31/32, diverging at the last token (§7.9), and then READ by
  Measurement A: it is a near-tie, and our two arms bracket the two values the
  oracle itself has produced there (§7.11).
- **A reader treats this row as scoring #1289.** Mitigated in §0, §7.9 and the
  pull request body. The 95/96 in §7.9 is `oracle.nhspeed-a.json` against
  `oracle.json`. It is not #1289's score and #1289 has not been run against it.
- **The named profile does not reproduce across runs.** Realised, and it is new:
  the same profile read 26/32 on prompt 2 on 2026-08-18 and 31/32 here, each
  with two agreeing legs (§7.9). A golden captured under it is a record of one
  run. §1's reading of the evidence has to account for this.
- **A near-tie gets reported as a verdict, in either direction.** The live one.
  §7.11 has the device arm 96/96 against one golden and the host arm 96/96
  against the other, on the same 96 tokens, with vLLM having produced both
  values. That is readable as "#1289 passes" and equally as "#1289 fails", and
  it is neither. Mitigated in §0, §7.11 and the pull request body: what is
  established is that the moved token does not separate the arms.
- **The KV-sizing observation gets promoted from lead to cause.** `num_gpu_blocks`
  1258 against 1249, from `Free memory on device` readings of 86.1 GiB against
  112.62 GiB, is recorded in §7.11 as a **LEAD**. No run has varied block count
  with everything else held, and no arithmetic here explains the direction. It is
  the §8 "a BOX-level measurement gets reported as a CAUSE" failure wearing a
  different hat, and the reason it is labelled rather than concluded.

## 9. Stop conditions

- Do **not** lower the watchdog floor below 15000 MB, for any reason.
- Do **not** `ssh` to a fleet device. `rc` is the only path, and the file mutex
  is retired for fleet devices.
- Do **not** overwrite `oracle.json`.
- Do **not** ratify a distributional gate.
- Do **not** write a reconstructed configuration into any golden. A capture that
  cannot read a key leaves it absent and the writer refuses.
- Do **not** report a token comparison as a verdict on #1289.

## 10. Now

**The capture ran, the golden exists, and Measurement A has read the one token
they disagree on. It is a near-tie. The golden is still NOT committed.**

`20260823T021635Z` completed engine start-up, generated both legs, agreed, and
wrote `oracle.nhspeed-a.json` under `nhspeed-a` with all twenty resolved engine
keys read back out of the built engine, at `revision`
`29f2d1746d8f41e316523194b19018707749b1b1` -- the SAME revision as the committed
golden (§7.9). The blocker this row was dispatched against is **gone and
understood**: it was not the model and not the engine configuration but
FlashInfer's nvcc JIT running one job per CPU on a 20-CPU box -- 22 concurrent
`cicc`/`cudafe++` processes holding 80789 MB while EngineCore held 2197 MB and
its device figure sat flat at 21436 MiB (§7.7). `MAX_JOBS=4` fixed it, changes
no engine key, and cannot move a token. Minimum `MemAvailable` was 64720 MB
against a 15000 MB watchdog that never fired, and `peakUsed_MB` was 53053
against 102948 for the killed attempt.

**Measurement A (§7.11) reads the 95/96.** Four legs and four MEASURED scores,
one leg against each golden it ran on; the four remaining table cells are
DERIVED, exactly, because the two goldens differ in one token and agree on the
other 95. The device arm emits `11286` at p2[31] and is 96/96 against
`oracle.nhspeed-a.json`; the host arm emits `3468` and is 96/96 against
`oracle.json`; they agree on all 95 other tokens. vLLM itself produced `3468` in
August and `11286` on 2026-08-23 under the same named profile, so **our two arms
bracket exactly the two values the oracle has produced**. Neither arm is
defective and neither capture is authoritative. `origin/main` ships the host arm
and is **96/96 against the committed golden today** (L4), and #1388's 95/96
belongs solely to #1289's device arm, which does not exist on `main` -- proven
with a positive control rather than assumed.

**The recommendation is that `oracle.json` is NOT re-pinned.** Re-pinning would
only move the gate from agreeing with the host arm to agreeing with the device
arm, which encodes an undetermined position as determined (§7.11).

What still blocks the artifact is smaller and it is a decision, not a defect.
The golden's `model` field records `/workspace/a3/ckpt-stage`, the path the
engine was actually given inside the lease; the committed golden records
`/mnt/nas_share/checkpoints/nemotron-3.5-lightning-30b-nvfp4`. Same checkpoint,
proven by sha256, and the guard reds anyway because the strings differ (§7.10).
The guard was NOT widened and the golden was NOT committed, so this branch is
green and `oracle.json` is untouched. **What `model` should mean in a golden --
provenance path or checkpoint identity -- is escalated, not decided here.**

Three things are measured-and-owed rather than done: the DIRECT arm proof
(every leg printed `NO NH-DIAG LINE`, so the 7.3x wall-time split is indirect
evidence only), the oracle's own across-run variance over six independent
processes, and #1710's pre-lease host-memory gap. #926 stays OPEN and is NOT
discharged.

## 11. Owed

- **The decision that blocks the artifact: what `model` means in a golden.**
  `identity_problems` compares it as identity; the capture records it as the
  provenance path the engine was given, and the canonical checkpoint path is not
  reachable inside an `rc` lease (§7.10). Both readings are defensible and the
  choice changes checker semantics, so it needs its own row, its own red-before
  case and its own fresh review. Until it is taken, the artifact and its evidence
  live on the share at
  `/mnt/nas_share/rc/golden-rederive/20260823T021635Z/oracle.nhspeed-a.json`,
  sha256 `d2a59a24674470d01178f8da9c5c1d180492ad55e10afab88fccc541c44e0d40`.
- **`nhspeed-a` did not reproduce across runs, and nobody knows why.** 26/32 on
  prompt 2 on 2026-08-18, 31/32 on 2026-08-23, each with two agreeing legs and
  the same 3561-character engine-config line. The resolved KV sizing differed
  (`128,819` tokens at `15.03 GiB` then, `127,897` at `14.92 GiB` and
  `num_gpu_blocks=1249` now) because it is an output of a memory measurement, but
  **that is a candidate and not a cause** and this row does not assert it.
- **#1431 is passed on this box, not fixed.** §7.7 names the mechanism for the
  runs this row measured -- nvcc fan-out in this job shape. Whether every #1431
  failure has that cause is unestablished, and the FIX belongs upstream of the
  job: any oracle job on this box that JITs should cap `MAX_JOBS`, and nothing
  enforces that today.
- **Whether the UMA branch fired is read, not measured.**
  `vllm/utils/mem_utils.py:148-155` sets `free_memory = psutil.virtual_memory().available`
  on an integrated GPU, and the 2026-08-18 arithmetic corroborates it, but the
  snapshot is logged at `logger.debug` (`gpu_worker.py:388`) and these runs ran
  at INFO. One `VLLM_LOGGING_LEVEL=DEBUG` line settles it.

- **[#926](https://github.com/mudler/vllm.cpp/issues/926) is NOT discharged by
  this row, and stays open.** #926 is that the reference every Nemotron token
  claim is scored against cannot be regenerated or attributed. This row produces
  an attributable golden, but it does not move any consumer onto it: the C++
  token gate, the A3 driver and #1289's score still read `oracle.json`, which is
  still the unattributable file. #926 closes when the reference in USE is
  attributable, not when an attributable file exists somewhere in the tree.
- **Repointing the token gate.** `test_nemotron_h_loader.cpp` and the A3 driver
  still read `oracle.json`. Moving them to the attributed golden is its own row,
  and should follow a measurement of what #1289 scores against it.
- **A VERDICT on #1289 -- still owed, and a near-tie is not one.** Measurement A
  scored the device arm as carried by tree a2q1 `a34e153f9` against both
  references in two separate legs -- MEASURED 96/96 against
  `oracle.nhspeed-a.json` (L2) and MEASURED 95/96 against `oracle.json` (L1) --
  mirrored exactly by the host arm, whose `oracle.json` score is MEASURED (L3,
  L4) and whose `oracle.nhspeed-a.json` score is DERIVED (§7.11). That establishes that
  the moved token does not separate the arms. It does not establish that #1289
  is correct, because the position it turns on is undetermined and both values
  came out of vLLM. What would settle it is p2[31] itself, which is the variance
  measurement below.
- **The capture itself** -- #1694 stays open, but for a different reason than
  when this line was written. The capture RAN (§7.9): `CAPTURE_RC=0`, both legs
  agree, the golden exists and passes its own contract. What it now needs is not
  a lease but the `model` decision above, because the artifact cannot be
  committed while the identity guard reds on it.
- **Who holds ~117 GB on `dgx.casa` while the CPU is idle.** Filed on #1431. It
  needs the host process table, which this row has no authority to read.
- **#1431's root cause.** Untouched by this row: its wall was never reached.
- **[#1710](https://github.com/mudler/vllm.cpp/issues/1710) -- `dgx:gpu0` begins
  a lease with 5 GB of 122 GB available and `rc` exposes no available-memory
  label, so `ready` cannot mean usable.** This is the blocker §10 describes, in
  its filed form: the capture is one `rc` job away, and the job cannot be
  usefully queued while a lease can be granted on a box with no memory. Owner:
  #1710. This row is no longer blocked on it -- the 2026-08-23 leases each began
  above 117000 MB and the quiet gate passed on the first sample -- but the defect
  is unrepaired and the next job to hit a starved box will pay it again. The fix
  is a controller-side label, which this row has no authority over.
- **The DIRECT arm proof is not taken.** Measurement A intended to confirm from
  the model's own counters which arm each leg executed, and all four legs printed
  `NO NH-DIAG LINE`. The `[NH-DIAG] ARM step` counters exist only in an
  uncommitted instrumented tree -- `grep -rn 'ARM step' src/ tests/` exits 1 on
  this branch -- so `VT_NEMOTRON_H_DIAG` prints nothing that names an arm. What
  stands in for the proof is the up-to-7.3x wall-time split between the flag-on
  and flag-off legs against a recorded 6.6x device-versus-host regime, and that
  is **indirect evidence only** (§7.11). Committing the counters, or any other
  arm-naming print, would settle it in one run. **A run aimed at exactly that was
  LAUNCHED at `/mnt/nas_share/rc/goldenab/20260823T095204Z-arm/`**, at 09:52 UTC
  on 2026-08-23 -- AFTER this branch's head `8d5c74922` was committed at 09:49
  UTC. It is therefore not part of this head, nothing from it is folded into
  this row, and neither its completion nor its outcome is claimed here. Whatever
  it reads belongs to the row that lands it, and this bullet stays owed until
  one does.
- **The oracle's own across-run variance is unmeasured.** §7.9 has the same named
  profile reading 26/32 on prompt 2 in August and 31/32 on 2026-08-23, each with
  two agreeing legs; §7.11 has two different values at p2[31] from vLLM itself.
  Two runs are not a distribution. The measurement owed is **six independent
  oracle processes** under `nhspeed-a` on this checkpoint, scored at p2[31], so
  that "the oracle produces both values" becomes a frequency rather than an
  anecdote. Nothing about p2[31] can be settled below that.
- **[#1729](https://github.com/mudler/vllm.cpp/issues/1729) -- `--check <path>`
  reports every top-level violation against the hardcoded name `oracle.json`.**
  Found by the fresh review of PR #1703 and visible in this spec's own §7.2
  M-A line. PRE-EXISTING: it predates this branch, and it becomes reader-visible
  the moment a second golden exists beside `oracle.json`, which is what this row
  produces. NOT fixed in flow: the repair threads the checked path into
  `check_golden` and owes its own regression case, which is a checker-semantics
  change rather than a one-line accuracy fix. Owner: #1729.
- **[#1730](https://github.com/mudler/vllm.cpp/issues/1730) -- this suite's only
  registration is one line of `ci.yml`, and `check-test-registration.py` stays
  at rc=0 when it is deleted.** The suite is also absent from
  `agent-preflight.sh`'s `SUITES` array. Found by the same review; PRE-EXISTING,
  since the suite was registered this way when it landed under #926. NOT fixed
  in flow: the repair edits a checker's required set and a shared runner array,
  which needs its own red-before evidence. Owner: #1730. Related: #408, #1509.
- **`sampling` is NOT part of the identity guard, and the reason first recorded
  here was the wrong one.** `identity_problems` holds a second golden to the
  same model, revision, prompt battery and tokenization as `oracle.json`, and
  deliberately not to the same `sampling` block. The justification originally
  written was that `check_golden` already ties every row's `token_ids` length to
  that file's own `sampling.max_tokens`, so a capture at a different DEPTH stays
  internally consistent. **That reasoning does not describe these two files.**
  `max_tokens` is 32 in both, so depth is not the delta at all: the delta is
  `ignore_eos`, present and `true` in `oracle.nhspeed-a.json` and absent from
  `oracle.json`, and it is §7.11's second lead because it sits on the stop
  condition and the goldens disagree at exactly the final token. So whether the
  guard should compare `sampling` is a **live question**, not a settled
  omission, and it wants its own row alongside the `model` decision above --
  with the caveat that comparing a key that may be absent because it was never
  RECORDED, rather than never SET, would gate an unattributable capture on
  something #926 says cannot be recovered. Named here so the omission stays a
  decision rather than an oversight, and so the wrong reason does not stand.
