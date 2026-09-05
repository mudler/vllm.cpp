# Sync cycle `e126687a9a`, wave TOKENGATE

Row: `UPSTREAM-SYNC-HEADPIN`.
Issue: [#2794](https://github.com/mudler/vllm.cpp/issues/2794).
Predecessors: [#2593](https://github.com/mudler/vllm.cpp/issues/2593) wave
HEADPIN, [#2611](https://github.com/mudler/vllm.cpp/issues/2611) wave RUNHALF,
[#2764](https://github.com/mudler/vllm.cpp/issues/2764) wave PORTQ-RECONCILE,
which named three blockers, and [#2771](https://github.com/mudler/vllm.cpp/issues/2771)
wave STEP6, which worked the second. This wave is blocker 1.
Row spec: [`upstream-sync-headpin.md`](upstream-sync-headpin.md).

## Now

**The gate does not yet exist, and what prevents it is a queue, not a
refusal.** This wave establishes what the gate must compare, makes that
comparison executable, proves the instrument on four mutations, stages every
input, and queues the one job that can produce the number. At the time of
writing that job is `efc30c74-005e-4e80-bc28-bd34f5b76b77`, position 12 in the
`dgx:gpu0` queue.

**Two findings stand on their own, and neither needed the job.**

1. **The token path is not pin-gated.** STEP6 established that the online-serving
   harness structurally refuses to measure at any revision but the pinned one.
   That refusal does not reach the token path: `scripts/opt-oracle-capture.py`
   and `scripts/opt-dgx-gate.sh` read no pin constant, and neither does
   `tests/vllm/models/test_opt_paged_engine.cpp`. §2.1 records how that was
   checked, because a zero from a grep is not an absence.
2. **The OPT strict golden was never re-validated at the ACTIVE pin.** It was
   captured once, at `b8358a5b9`, against vLLM 0.25.0 / `e24d1b24`. The
   `5559679229` advance's W3b table put OPT in the "already RATIFIED
   near-tie-robust" row and skipped its re-capture on the grounds that a
   distributional gate absorbs drift. OPT's gate is not distributional. §2.4.

The pin does **not** advance and nothing here is a reason to move it. The active
parity pin remains `5559679229bc961848b121ccdeaa8fa5d79bec98`.

**SUPERSEDED 2026-09-03: the pin HAS advanced**, to
`e126687a9a828d513c01a07cd69f025f27d63280`, by the developer ruling recorded in
[`upstream-pin-advance-e126687.md`](upstream-pin-advance-e126687.md) §1
([#2817](https://github.com/mudler/vllm.cpp/issues/2817)). The sentence above was
true of THIS wave and is kept because it was that wave's own stop condition. It
is no longer true of the tree.

## 1. Scope

**The question.** `.agents/oracles/vllm.md` lists four obligations a pin advance
to `e126687a9a828d513c01a07cd69f025f27d63280` owes. The second is "a declared
token-exact gate." Does one exist at the target, and if not, what exactly
prevents it?

**In scope.** The OPT-125m greedy gate and its committed goldens
(`tests/parity/goldens/opt_greedy/`), its capture script
(`scripts/opt-oracle-capture.py`), its gate source
(`tests/vllm/models/test_opt_paged_engine.cpp`), the comparison between a
candidate-revision capture and the committed golden, and the record of what the
`5559679229` advance did and did not re-validate.

**Out of scope.** Advancing the pin. The PORT-NOW queue. The FlashInfer step-6
re-measurement. Porting anything. The other model goldens the pin advance's W3b
step covers (27B, 32B, 35B, Coder); those were re-captured at `5559679229` and
their re-capture at the target is a separate, larger job that this wave does not
attempt and does not discharge.

**Not a substitute.** Nothing measured on `thor:gpu0` closes this. §2.3.

## 2. Design

### 2.1 The token path is not pin-gated, and how that was checked

STEP6's refusal is real and it is narrow. `tools/bench/online_gate.py` reads
`VLLM_ORACLE_VERSION`, `VLLM_DISTRIBUTION_VERSION` and `FLASHINFER_VERSION` from
`serve_low_common.py`, which reads `.agents/upstream-sync.md`'s ` ```parity-pin `
block, so that harness cannot measure at a revision the block does not name.
Generalising that across paths is exactly the error the predecessor wave's
discharge withdrawal punished, so it was checked rather than assumed.

The check is a **full read** of both harness files, not a grep, because a grep's
zero has eight recorded ways of being wrong in this tree. `opt-oracle-capture.py`
is 142 lines and `opt-dgx-gate.sh` is 57; both were read end to end. Neither
imports anything from `tools/bench/`, neither opens `.agents/upstream-sync.md`,
and neither takes a revision as input at all: the capture script's only
oracle-side input is whichever `vllm` is importable in the venv on `PATH`.

The grep is recorded as corroboration with its positive control in the same
probe form, so that a zero is legible:

```console
PAT[serve_low_common]     -> opt-dgx-gate.sh:0  opt-oracle-capture.py:0
PAT[read_parity_pin]      -> opt-dgx-gate.sh:0  opt-oracle-capture.py:0
PAT[assert_oracle_commit] -> opt-dgx-gate.sh:0  opt-oracle-capture.py:0
PAT[FLASHINFER_VERSION]   -> opt-dgx-gate.sh:0  opt-oracle-capture.py:0
PAT[VLLM_ORACLE_VERSION]  -> opt-dgx-gate.sh:0  opt-oracle-capture.py:0
PAT[parity-pin]           -> opt-dgx-gate.sh:0  opt-oracle-capture.py:0
PAT[def main]             -> opt-dgx-gate.sh:0  opt-oracle-capture.py:1   <- control
PAT[PROMPTS]              -> opt-dgx-gate.sh:0  opt-oracle-capture.py:5   <- control
PAT[greedy_ids]           -> opt-dgx-gate.sh:0  opt-oracle-capture.py:4   <- control
```

**Scope of that claim.** It is about these two files and the gate source they
feed. It is not a claim about `tools/bench/`, which does read the block, and it
is not a claim that any other gate in the tree is revision-portable.

### 2.2 What the gate compares, and why #2628's run was not one

The declared token-exact gate for OPT is `test_opt_paged_engine.cpp`: **our**
paged engine's greedy tokens against **vLLM's**, the latter committed as
`greedy_ids.npy`. To hold "at `e126687a9a`" the golden must be the answer vLLM
gives at `e126687a9a`. So the gate at the target is the pin advance's own W3
step, which `.agents/specs/pin-advance.md` states as: regenerate the golden at
the target, diff it against the committed one, keep it if unchanged, commit the
new one and re-pass the gate against it if changed.

#2628 laid a target-revision run beside the committed golden and got 96/96. It
is not that step, for three reasons, and the third was not named there.

| | #2628's run | What the gate needs |
|---|---|---|
| Device | Thor, `sm_110` | GB10, `sm_121a` — the device the golden was captured on |
| Checkpoint | raw `facebook/opt-125m` fp16, rounded by vLLM at load | the bf16-materialized artifact `scripts/opt-materialize-checkpoint.py` writes, decisions D1/D2 of [`sweep-opt-125m.md`](sweep-opt-125m.md) |
| K | one run per leg | `--runs 5`, because K **selects the gate** |

**The device is the load-bearing confound; the checkpoint is a weaker point and
is stated as one.** Revision moved together with silicon, and an agreement
across two moved variables is not attributable — neither would a disagreement
have been. The checkpoint difference is real but the repository's own record
says it should be inert: decision D1, written into
`scripts/opt-oracle-capture.py`'s header, is that the materializer applies the
**same** single fp16-to-bf16 rounding vLLM applies at load, and
`.agents/scripts/runhalf-e126687-gen.py` passes `dtype="bfloat16"` on the raw
snapshot. So the two artifacts should agree, and an earlier draft of this spec
calling the checkpoint "a second uncontrolled variable" overstated it. What is
true and sufficient is narrower: the capture at the target must use the same
artifact the golden was captured against, because an undeclared difference in a
gate's inputs is not something a gate gets to assume away — not because that
difference has been shown to matter.

The third is not a confound but a missing measurement, and it is the one that
makes this a gate rather than a diff. `test_opt_paged_engine.cpp` uses a STRICT
token-exact bar **with no near-tie band at all**, and the thing that licenses
that bar is `greedy_dist.npy`: K=5 oracle runs with zero multi-valued
(prompt,pos) cells, re-asserted at the top of the test. #2628 ran
`.agents/scripts/runhalf-e126687-gen.py`, a copy of the capture script's prompt
battery with no K loop and no dist output, so **the candidate oracle's own
self-determinism at the target is unmeasured**. The committed `greedy_dist.npy`
cannot stand in for it: it is the previous oracle's measurement of itself. If
the candidate has become non-deterministic on this battery, the strict bar must
be re-derived per [[near-tie-distributional-gate]] and not silently kept — and
no diff of `greedy_ids.npy` alone can see that.

**Every capture argument is the capture script's own default**, passed
explicitly so the log records it rather than implying it: `--runs 5`,
`--max-tokens 16`, `--gpu-mem-util 0.20`, `--max-model-len 2048`. `--runs` is
the only argument the committed invocation in that script's docstring supplies.
An earlier draft of the job passed `--gpu-mem-util 0.10`, which is RUNHALF's
**thor** value chosen for a unified-memory box, and that would have been an
undeclared config delta of exactly the kind this wave's central argument is
about. `0.20` is what the committed golden was captured under. It is corrected
in the job, and recorded here rather than left to a reader to notice.

**So the construction is:** on GB10, against the same bf16-materialized
checkpoint, with `scripts/opt-oracle-capture.py --runs 5` verbatim, capture
`greedy_ids.npy`, `greedy_dist.npy` and the six `p{i}_prompt.i32`; then diff all
three against the committed set. The revision is then the only thing that moved.

### 2.3 Why `thor:gpu0` cannot substitute

Thor is `sm_110` and already carries a built candidate wheel, so it is the cheap
option and it is the wrong one. A capture there moves the device again, and the
result would be a different measurement wearing the label of the one that is
owed — the reason STEP6 declined a free Thor for its own re-measurement.

The asymmetry is worth stating because it is the only thing Thor could have
contributed. A **non**-deterministic K=5 result on Thor would be a positive
finding: it would show the candidate can diverge on this battery at all. A
deterministic result there transfers nothing to GB10. Since only one of the two
outcomes is usable and the useful one is the unlikely one, no Thor lease was
spent, and no number from Thor appears in this wave.

There is a second, independent reason. The `sm_110` wheel is built with
`TORCH_CUDA_ARCH_LIST=11.0`, which emits `sm_110` cubins and no PTX, so it
cannot run on GB10 at all. The GB10 arm is a build, not a copy.

### 2.4 The OPT golden was never re-validated at the active pin

`git log --follow` over `tests/parity/goldens/opt_greedy/greedy_ids.npy` and
`greedy_dist.npy` returns exactly one commit, `b8358a5b9`, the OPT W0-W4 landing,
whose subject records the oracle as vLLM 0.25.0. `b8358a5b9` is an ancestor of
`bc415a3e4`, the `e24d1b24` -> `5559679229` pin flip, so the golden predates the
advance and no commit has touched it since.

The advance's W3 step is explicit that every SACRED gate's golden is regenerated
at the new oracle and diffed. Its W3b table discharges OPT in a row covering
"~30 model-matrix `*_greedy` rows (llama/opt/phi/mistral/internlm/minicpm/yi/
olmo2/deepseek-v2/glm4-moe-lite/…)", with the method column reading "already
RATIFIED near-tie-robust gates (`greedy_dist.npy`, `kNearTieMnats=500`)" and the
result column "no re-capture: the distributional gates absorb near-tie drift by
construction."

**OPT is not in that class.** `test_opt_paged_engine.cpp` says so in its own
header — "Where vLLM is self-consistent the honest bar is exact agreement — so
no near-tie band is used here at all" — and the body carries no `kNearTieMnats`.
The row's premise is false for this member, so the discharge does not cover it.

**How wide the mis-bucketing is, measured rather than guessed.** Of the 26
`tests/**/test_*paged_engine*.cpp` gates in the tree, exactly five carry no
`kNearTieMnats` at all: `test_qwen27_paged_engine.cpp`,
`test_qwen27n_fp8_tower_paged_engine.cpp`, `test_qwen36_paged_engine.cpp`,
`test_gemma4_paged_engine.cpp` and `test_opt_paged_engine.cpp`. Every other one
has 4 or more, `test_qwen3_5_vl_video_e2e.cpp` has 5 through the same probe, and
that is the positive control which makes the zeros legible. Cross that against
the ten models the W3b row names — llama, opt, phi, mistral, internlm, minicpm,
yi, olmo2, deepseek-v2, glm4-moe-lite — and **OPT is the only one of the ten
whose gate has no near-tie band.** The row is right about its other nine named
members and wrong about this one, which is why the error survived.

The two remaining no-band gates, `qwen27n_fp8_tower` and `gemma4`, are not in the
W3b table and are **not examined here**. Gemma-4 was unblocked by this very
advance, so it was plausibly born at the new pin; `qwen27n_fp8_tower` is
unchecked either way. Neither is claimed as drift and neither is claimed as
clean.

The consequence is bounded and should not be overstated. The active pin does have
declared token-exact gates: 27B W4A4, 32B-NVFP4A16, 35B and Coder were each
re-captured or re-measured at `5559679229` and recorded bit-identical or
byte-stable. What is false is the narrower claim that *every* strict golden was
carried across that advance. OPT's was not, and the same capture this wave
queues answers both questions at once, since a candidate capture that reproduces
the committed bytes reproduces them across both advances.

### 2.5 The instrument

`scripts/opt-oracle-capture.py` writes a golden and prints a determinism report.
Nothing in the tree ever laid two captures beside each other; the pin advance's
W3 diff was done by hand. `.agents/scripts/tokengate-e126687-diff.py` makes that
step executable. It checks three things that are not the same check:

1. `greedy_ids.npy`, the bar, by sha256 and by mismatched position.
2. the six `p{i}_prompt.i32`, the **input**. A matching output on a moved input
   would be a coincidence, not agreement.
3. `greedy_dist.npy`, the **selector**, recomputed from the candidate's own K
   runs rather than read from the committed file.

Exit 0 is all three holding; exit 1 is a real difference, which is a finding;
exit 2 is a missing or malformed input, which is the instrument failing rather
than the comparison failing. That separation is deliberate: an instrument whose
failure looks like a result is this row's recurring trap, and three of RUNHALF's
four reds were its own instruments.

**Two properties of the job carry the verdict, and a fresh review of #2801
found both missing from the first draft.**

**The staged inputs are sha256-asserted, not only the checkpoint.** `/workspace`
is a shared CIFS surface other sessions write. The first draft asserted the
checkpoint and nothing else, while reading the capture script, the differ and
`goldens-committed` — **which is the bar** — off that same surface unchecked. A
stale staged golden would have produced a confident verdict against the wrong
reference, and the `VERBATIM` comment beside the capture invocation was a claim
no reader of the job's output could check. The job now asserts all ten files
against the sha256 they have in the tree, prints each one, reports every
mismatch rather than stopping at the first, and exits 9. This is the property
#2628 named as its own strength.

**A drift has its own exit code.** The first draft put `DIFF_RC` into a log line
and fell through to exit 0, which made a real golden drift — the one outcome the
wave exists to detect — indistinguishable from success in `rc`'s eyes. The map
is now: 0 reproduces, **7 drift**, 8 the differ could not compare, 2-6 and 9 the
environment or the instrument. Only 0 and 7 are statements about the target.

`.agents/scripts/tokengate-e126687-job.sh` is the lease job. It **refuses to run
on anything but compute capability 12.1** rather than adapting to the device it
finds, because a capture on the wrong silicon is the failure this wave exists to
avoid; the refusal is the gate's device assertion, expressed where it cannot be
skipped. It is resumable in two stages (`STAGE=build`, `STAGE=capture`) and
persists the wheel to `/workspace` the moment it exists, because `dgx:gpu0` has
crashed under sequences shorter than the ~1.3 h source build this needs.

### 2.6 Nothing records which oracle revision a golden was captured at

The reason OPT's golden crossed a pin advance unexamined is structural, and the
tree already contains the fix it did not apply.

`test_opt_paged_engine.cpp` names no revision at all — a scan of all 298 lines
for `serve_low_common`, `parity-pin`, `upstream-sync`, `assert_oracle_commit`,
`FLASHINFER`, `5559679`, `e126687` and `e24d1b24` returns zero for every one,
against `PARITY_GOLDENS_DIR` at 1 and `greedy_ids` at 3 as controls. That is
correct design: the gate compares bytes, and the revision those bytes came from
is a property of the file, not of the test. But the file does not carry it
either. `tests/parity/goldens/opt_greedy/` holds eight binaries and nothing else,
so the only record that its oracle was vLLM 0.25.0 is the subject line of
`b8358a5b9`. A pin advance sweeping ~30 golden directories has nothing to read.

**One greedy golden already does this right.**
`tests/parity/goldens/qwen35_greedy_0_8b/manifest.json` carries an `oracle` block
with `vllm_commit`, `vllm_runtime_version`, the wheel filename, the container
image, the hardware, the exact invocation and the capture date, plus a `model`
block with the HuggingFace `revision`. That is precisely the record that would
have made this wave's §2.4 finding a one-line check instead of a `git log
--follow`. It is one file per golden, read by glob, so it is the record shape
AGENTS.md prescribes rather than another surface every change must write.

Of the 21 `tests/parity/goldens/*_greedy*` directories, exactly **one** has a
manifest. This wave does not add the other twenty, and deliberately does not add
`opt_greedy`'s: the honest value of its `oracle.vllm_commit` field is not
`e24d1b24`. The pin at the capture date was `e24d1b24` and the commit subject
says "vLLM 0.25.0", but `.agents/environment.md` records the 0.25.0 oracle as a
**pip** stack, and `702f4814fe54` is the revision this repository audited as
v0.25.0 — so the release the wheel carried and the commit the pin named are not
known to be the same object, and writing either into a provenance field would
manufacture a fact. The capture this wave queues resolves it by measurement: a
manifest written from that job records a commit it actually ran.

### 2.7 The job could not build, and why the repair reads the requirement from the tree

Job `efc30c74-005e-4e80-bc28-bd34f5b76b77` ran on `dgx:gpu0` and exited **4**
three seconds into the build stage
([#2895](https://github.com/mudler/vllm.cpp/issues/2895)):

```
File "<string>", line 21, in <module>
ModuleNotFoundError: No module named 'setuptools_rust'
```

Everything the job asserts before that passed: `COMPUTE_CAP=12.1`,
`SUM DEVICE_RC=0`, `HEAD_SHA` confirmed at `e126687a9a8`, `SUM PIPREQ_RC=0`. The
goldens, the differ and the staged-input block are not implicated, and neither
is the device guard.

**The mechanism.** The build runs `pip wheel --no-deps --no-build-isolation`, and
`--no-build-isolation` makes pip skip `pyproject.toml`'s `[build-system]
requires` **entirely**. Nothing else in the job put a build dependency in the
venv except one hand-kept list, `pip wheel setuptools setuptools_scm ninja cmake
numpy`. At the target the `requires` are `cmake>=3.26.1`, `ninja`,
`packaging>=24.2`, `setuptools>=77.0.3,<81.0.0`, `setuptools-scm>=8.0`,
`setuptools-rust>=1.9.0`, `torch==2.13.0`, `wheel` and `jinja2`; the list was
three names behind. `setup.py:21` at the pin is
`from setuptools_rust.build import build_rust`, which is the frame in the
traceback.

**The repair does not add three names.** It installs
`requirements/build/cuda.txt` out of the clone the job already made at
`$TARGET_SHA`. `pyproject.toml` names that file itself — "Should be mirrored in
`requirements/build/cuda.txt`" — and at `e126687a9a8` the mirror covers all nine
`requires` and adds `build`, `protobuf` and `regex`. A list read from the tree
cannot drift from the pin. A list retyped into the harness silently can, and
that is the defect rather than the three missing names: the next pin advance
would reproduce it.

**Upstream's mirror comment is an intention, not a guarantee**, so the job now
also asserts the outcome. It parses `[build-system] requires` out of the cloned
`pyproject.toml` and requires every distribution named there to resolve in the
build venv, refusing with `BUILDREQ_RC=1` and **exit 3** — the existing
"build prerequisites missing" code — and naming what is absent. It checks
presence, not version specifiers: pip resolved the versions when it installed
the mirror, and the class of failure that stops the build is a name nothing
installed.

**The assertion is falsifiable without a lease.**
`tests/scripts/test_tokengate_buildreq.py` extracts that program from the
committed script and runs it against scratch trees it writes: a distribution
that resolves, one that does not, PEP 503 spellings of one name, an extras
marker, and a `pyproject.toml` whose `requires` cannot be read. Run against the
pin's real `pyproject.toml` on a host with none of the build tools it prints
`BUILDREQ MISSING cmake setuptools-scm setuptools-rust` and exits 1. The same
suite pins the three things the repair must not have moved: the ten staged-input
`assert_sha` calls, the compute-capability guard, and the exit map in which
DRIFT is 7 and no instrument failure is.

**The checker and the shell that obeys it fail independently, so both are
pinned.** Running the extracted program says what it RETURNS and reaches none of
the lines that act on that return value. A first review found every one of them
unmeasured: `BUILDREQ_RC=$?` replaced by a literal, the `-ne 0` branch replaced
by `if false`, the five-line refusal deleted outright, and `exit 3` turned into
`exit 0` all left the suite green, and so did repointing the checker at a
snapshot instead of the clone — which is the fix's whole thesis. The
compute-capability guard was in the same state: deleting the whole
`if [ "$CAP" != "12.1" ]` block, or the `CAP=` line it compares, changed
nothing the suite could see. `TheShellRefusesWhenTheEmbeddedAssertionReds` and
`test_the_compute_capability_guard_survives` close both, and §4 records each
mutation going red.

## 3. Risks

- **The queue.** `dgx:gpu0` had eleven jobs ahead of this one at submission. The
  wave's deliverable degrades gracefully: the design, the two findings and the
  proven instrument stand whether or not the job runs.
- **The build.** No aarch64 wheel exists for this revision, so GB10 needs a
  from-source build (~1.3 h at the pin, 94 min at the candidate on Thor). The
  worker has no `github.com` egress, so all nine `FetchContent` repositories and
  the vLLM tree itself are staged from `/workspace`; RUNHALF's staging is reused
  unchanged.
- **FlashAttention on GB10.** `flash-attention/CMakeLists.txt:140` computes
  `FA2_ARCHS` as a loose intersection of `"8.0+PTX"` with the target archs, so a
  `12.1` build reaches `sm_121` by a driver JIT of `compute_80` PTX — the mode
  recorded as failing on GB10 with `cudaErrorUnsupportedPtxVersion`. If the
  engine cannot select its default backend the job records which backend
  produced the tokens; a capture on a non-default backend is reported as such
  and is not silently labelled the gate.
- **A false pass by coincidence.** Ruled out by checking the input tokenization
  alongside the output, §2.5 item 2.

## 4. Gates

| Gate | Command | Result |
|---|---|---|
| The differ detects a drifted token id | mutate `greedy_ids[2,7]`, rerun | **rc 1**, `IDS_DIFF prompt[2] pos 7: committed 4 candidate 5` |
| The differ detects a moved input | mutate `p3_prompt.i32[1]`, rerun | **rc 1**, `PROMPT[3] … EQUAL False` |
| The differ detects a lost selector | mutate `greedy_dist[0,3,4]`, rerun | **rc 1**, `SELECTOR K=5 multi_valued_cells 1` |
| A missing CANDIDATE input is NOT a finding | delete the **candidate's** `greedy_dist.npy` | **rc 2**, `FATAL missing input` |
| A missing CANDIDATE bar is NOT a finding | delete the **committed** `greedy_ids.npy` | **rc 2**, `FATAL missing input` |
| A missing COMMITTED dist is NOT an error, and says so | delete the **committed** `greedy_dist.npy` | **rc 0**, `DIST committed ABSENT (not required: the selector is the CANDIDATE's own K runs)` |
| Identical inputs pass | committed vs a copy of itself | **rc 0**, `TOKENGATE_VERDICT PASS` |
| `assert_sha` accepts a matching file | the real staged `greedy_ids.npy` | **rc 0**, `STAGED OK 078d1593…` |
| `assert_sha` rejects a wrong hash | same file, zeroed expectation | **rc 1**, `STAGED MISMATCH` + want/got |
| `assert_sha` rejects an absent file | a path that does not exist | **rc 1**, `STAGED MISSING` |
| The integrity block passes on the REAL staged tree | run the block extracted from the staged job against `/workspace` | **rc 0**, ten `STAGED OK` lines |
| The integrity block catches a corrupted BAR | flip one byte of the staged `greedy_ids.npy`, rerun | **rc 9**, mismatch named, other nine still checked |
| A drift leaves the job with its own status | `DIFF_RC` 0 / 1 / 2 / unset through the job's exit map | **0 / 7 / 8 / 0** |
| The token harness reads no pin constant | full read + the §2.1 probe with controls | zero hits, controls 1/5/4 |
| The device guard is pinned: its refusal | delete the whole `if [ "$CAP" != "12.1" ]` block | **suite rc 1**, `test_the_compute_capability_guard_survives` |
| The device guard is pinned: its input | delete the `CAP="$(nvidia-smi …)"` line | **suite rc 1**, same case |
| The device guard is pinned: its exit | wrong-device branch `exit 2` → `exit 0` | **suite rc 1**, same case |
| The shell reads the CHECKER's status | `BUILDREQ_RC=$?` → `BUILDREQ_RC=0` | **suite rc 1**, `test_the_status_the_refusal_tests_is_the_checkers_own` |
| The shell BRANCHES on it | `if [ "$BUILDREQ_RC" -ne 0 ]` → `if false` | **suite rc 1**, `test_a_red_checker_stops_the_job_with_the_prerequisites_code` |
| The branch REFUSES | that branch's `exit 3` → `exit 0` | **suite rc 1**, same case |
| The refusal exists at all | delete all five lines of it | **suite rc 1**, both cases |
| The ABSENT-file branch refuses too | its `exit 3` → `exit 0` | **suite rc 1**, `test_a_missing_requirements_file_refuses_instead_of_building` |
| The checker reads the CLONE | `$WORK/vllm/pyproject.toml` → `$WORK/snapshot-pyproject.toml` | **suite rc 1**, `test_the_checker_is_handed_the_cloned_tree_not_a_snapshot` |
| Red before the repair | the 16-case suite against the parent's script | **13 red**; the 3 green in both directions are the must-not-move guards |
| The capture at the target on GB10 | `efc30c74-005e-4e80-bc28-bd34f5b76b77` | **PENDING**, queued |

Every rc above was read directly, never after a pipe.

## 5. Stop conditions

- Stop on a `DIFF_RC=1`: a drifted golden is a finding to record and re-gate
  against, not a defect to repair in this wave.
- Stop rather than capture on any device whose compute capability is not 12.1.
- Stop rather than advance the pin. That is not this wave's authority.

## Owed

- **The capture itself**, until job `efc30c74-005e-4e80-bc28-bd34f5b76b77`
  returns (#2794).
- **Our arm re-run against whatever golden that job produces.** If the bytes are
  identical the existing green `test_opt_paged_engine` measurement carries over
  to those bytes unchanged; if they drift, the gate must be re-passed on GB10
  with our code byte-unchanged, which is a second lease (#2794).
- **A re-capture of the OPT golden at the ACTIVE pin `5559679229`**, which §2.4
  shows was skipped. A candidate capture that reproduces the committed bytes
  answers this too; one that does not leaves it open (#2794).
- **[#2805](https://github.com/mudler/vllm.cpp/issues/2805): the STRICT bar's
  licence is inside an existence guard.**
  `tests/vllm/models/test_opt_paged_engine.cpp:155` wraps the self-determinism
  re-assertion in `if (fs::exists(gdir / "greedy_dist.npy"))`, so deleting that
  file leaves `multi_cells` at `-1`, fires no `CHECK`, and runs the strict bar
  unlicensed. It is owned by `MODEL-TEXT-opt-optfor-causal-lm`, not by this
  wave, because it changes what a SACRED gate asserts.

  **The reachable form is a degenerate K, and the first draft of that issue got
  the mechanism wrong.** It claimed a re-capture could write `greedy_ids.npy`
  without `greedy_dist.npy`; `scripts/opt-oracle-capture.py:113-114` saves both
  unconditionally on every path and `:86` is `K = max(1, args.runs)`, so the
  script cannot do that. The claim came from a review of #2801 and was relayed
  without being run down; it is corrected on the issue. What is true and worse:
  **there is no floor on `DK` anywhere in the gate.** `grep -n DK` over
  `test_opt_paged_engine.cpp` returns exactly three lines — the read at `:158`,
  the print at `:172`, the loop at `:164` — against 14 `REQUIRE`/`CHECK`
  occurrences as a control. A capture at `--runs 1` therefore writes a
  well-formed `[N,T,1]` dist whose every cell has one member by construction,
  `multi_cells` computes to 0, the `CHECK` passes, and the strict bar is
  licensed by a measurement that cannot detect non-determinism at all. The
  capture script's help says `K>=5`; nothing enforces it and the gate never
  looks. That bears directly on this wave, whose whole argument is that K
  selects the gate.
- **[#2798](https://github.com/mudler/vllm.cpp/issues/2798): `opt-dgx-gate.sh`
  claims a `flock` it never takes.** Deliberately not repaired: a bare `flock`
  would recreate the two-mutexes-that-do-not-exclude-each-other failure that
  file's own history caused, so the repair needs the lease relationship decided
  first.
- **A provenance manifest for `opt_greedy`, and for the other nineteen greedy
  goldens that lack one**, on the shape
  `tests/parity/goldens/qwen35_greedy_0_8b/manifest.json` already uses. §2.6. The
  OPT one should be written from the capture job's own output rather than
  reconstructed, which is why it is owed rather than done here (#2794).
  A reviewer of #2801 noted that writing nothing is not the only honest option:
  a manifest recording date, device, invocation and checkpoint with
  `vllm_commit` explicitly **null** would manufacture no fact and would still
  make the debt legible. That option is recorded here rather than taken, because
  the queued capture produces the authoritative value and a second manifest
  would then have to be reconciled with it. The survey behind this is stronger
  than an earlier draft claimed: `find tests/parity/goldens -name manifest.json`
  returns **67**, and `grep -rl vllm_commit` over that tree returns exactly
  **one**, `qwen35_greedy_0_8b/manifest.json`. The review of #2801 reported 68
  for the first of those; re-counted here it is 67, and the count that belongs
  in the record is the one this wave measured rather than the one it was
  handed.
- **[#2794](https://github.com/mudler/vllm.cpp/issues/2794): whether the leased
  container can satisfy the pin's Rust workspace.** The target carries
  `rust/Cargo.toml` and a `rust-toolchain.toml` on channel 1.95, and the
  container's toolchain is unknown. `setup.py:1495` passes
  `optional=not should_require_rust_frontend()` and `VLLM_REQUIRE_RUST_FRONTEND`
  is unset, so an absent `cargo` **should** be tolerated and the frontend
  skipped. Read from the source, never observed: the build has not yet reached
  that line on the fleet. The next run answers it, and it is recorded rather
  than probed because probing it costs the same lease the run needs.
- **[#2794](https://github.com/mudler/vllm.cpp/issues/2794): whether
  `--max-runtime 5h` covers a COLD build.** The job's ccache remote store at
  `$WS/ccache-remote` is empty, because the run that would have filled it never
  reached compilation. Every published timing for this tree's CUDA builds is a
  warm one, so the budget for a cold `MAX_JOBS=4` build at `TORCH_CUDA_ARCH_LIST=12.1`
  is an estimate. The build stage is a no-op once the wheel is persisted and
  `STAGE=build` is separable from `STAGE=capture`, so a timeout costs a requeue
  and not the measurement.

  Both are anchored on #2794 and not on #2895. #2895 is the build failure, and
  the change that records these questions closes it; an `## Owed` bullet whose
  issue closes in the same commit points at a closed issue from the moment it
  lands. #2794 is the obligation that outlives it — the declared token-exact
  gate at `e126687a9a` — and the leased run that discharges it is the same run
  that answers both questions.
- **The other strict goldens at the target** — 27B W4A4, 32B-NVFP4A16, 35B,
  Coder. The pin advance re-captured all four at `5559679229`; none has been
  re-captured at `e126687a9a`, and this wave does not attempt it (#2794).
