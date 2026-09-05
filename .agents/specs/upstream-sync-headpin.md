# Sync cycle `e126687a9a`, wave HEADPIN

Issue: [#2593](https://github.com/mudler/vllm.cpp/issues/2593).
Predecessors: [#2589](https://github.com/mudler/vllm.cpp/issues/2589) (PINWINDOW,
`db92053e97`) and [#2524](https://github.com/mudler/vllm.cpp/issues/2524)
(PINPORT, `cdefd9d499`).
Guide: [`../upstream-sync.md`](../upstream-sync.md) §"The sync cycle".

## Now

**The pin has advanced**, to `e126687a9a828d513c01a07cd69f025f27d63280`, on
2026-09-03, by wave PINADVANCE
([#2817](https://github.com/mudler/vllm.cpp/issues/2817),
[`upstream-pin-advance-e126687.md`](upstream-pin-advance-e126687.md),
[`../sync/2026-09-03-e126687-advance.md`](../sync/2026-09-03-e126687-advance.md)).
The order was inverted by a developer ruling: step 6 now runs against the new pin
and is owed by [#2818](https://github.com/mudler/vllm.cpp/issues/2818), where a
red requires REVERTING the pin. Also still owed at the new pin: the declared
token-exact gate ([#2794](https://github.com/mudler/vllm.cpp/issues/2794)), a
reading on `dgx:gpu0` and the 290-entry PORT-NOW queue
([#2611](https://github.com/mudler/vllm.cpp/issues/2611)), and `qwen4_exp` still
does not run on this fleet
([#2626](https://github.com/mudler/vllm.cpp/issues/2626)).

This wave (HEADPIN) did **not** advance it; §1 below is its own scope and stop
condition, unchanged.

## 1. Scope

One measurement decides this wave, and everything else in it is the record of
that measurement.

**The question.** #2524 concluded that no revision carrying `qwen4_exp` can be
pinned on this fleet. `instanttensor` entered `requirements/cuda.txt` unguarded
at `2d7f42b4f3` vllm#52801 on 2026-08-19, which is an ancestor of
`e126687a9a` vllm#53896, the 2026-08-31 commit that registers the model, and
every fleet device is aarch64. #2589 then measured that `instanttensor 0.1.9`
builds from source on aarch64 once `python3-dev` is installed, and said in the
same breath that this makes the earlier conclusion a **hypothesis** rather than a
result, because nobody had re-run the runtime-requirements install at a
`qwen4_exp`-carrying revision with the header package present. This wave runs it.

**The target.** `e126687a9a828d513c01a07cd69f025f27d63280`, 2026-08-31,
`[Model] Support Qwen3.8-Flash-Next (#53896)`. It is the earliest revision that
serves `MODEL-MM-QWEN4-EXP`: it registers `Qwen4ExpForCausalLM`,
`Qwen4ExpForConditionalGeneration` and `Qwen4ExpMTP`, and adds the
`vllm/models/qwen4_exp/` tree with its `nvidia/` and `amd/` backends. Every
earlier revision either carries the model and the blocker, or neither.

**Why not PINWINDOW's target.** `db92053e97` is the last revision below the
`instanttensor` promotion, and it installs. It also predates `qwen4_exp` by
twelve days, so an advance to it leaves `MODEL-MM-QWEN4-EXP`'s vLLM citations
ahead of the pin and gives the row no oracle for its own model. That is the
difference between a pin advance that is bookkeeping and one that serves a row.

In scope:

1. Measure `APTDEV_RC`, `BUILDDEPS_RC`, `RUNDEPS_RC`, `BUILD_RC`, `IMPORT_RC` and
   `EXT_RC` at the target on an aarch64 fleet device, in one `rc` lease, each
   reported separately and literally.
2. Measure whether `qwen4_exp` is registered in the installed package and
   whether `vllm.models.qwen4_exp` imports.
3. Record the result and say what a pin advance to this target would cost.

Out of scope:

- **Advancing the pin.** #2589 left 206 unworked PORT-NOW entries for a strictly
  narrower range; this range is larger. An advance needs that queue worked.
- Working that queue, or re-deriving its dispositions. They are cited, not
  re-read.
- Any product code. This wave touches records and documents only, so it mixes no
  feature work into a sync cycle.

## 2. Design

One job, one lease, one script whose sha256 the report names. The script asserts
that each step ran rather than inferring it from the next step's result:

> **The outcome falsified the first three words, and they stand as written
> because they were the plan.** It took **two** lease jobs, both on `thor:gpu0`,
> never two devices and never two at once. The first asked `pip` for
> `requirements/build.txt`, which does not exist because `requirements/build` is
> a directory, so it lost its whole build-and-import half and a second job was
> needed to repair the instrument. Neither script is committed; see
> [`../sync/2026-09-02-e126687.md`](../sync/2026-09-02-e126687.md) §5.3, C6 and
> C10, the last of which records why the attempt to commit them was reverted.
> The stop condition in §5 was about not spending a lease on upstream `main`,
> and that held: no lease measured `main`.

- Every leg captures `$?` into its own named variable on the line after the
  command, never after a pipe, and the report prints the six variables verbatim.
- `python3-dev` is installed **first**, because it is the hypothesis under test,
  and `Python.h` is checked for presence before and after so the install is
  observed rather than assumed.
- The source is a **real git clone**, not a release tarball. #2589's job A
  measured `BUILD_RC=1` purely because `setuptools_scm.get_version()` cannot run
  outside a git repository, and a future job reaching for a tarball would read
  that as a target defect. The clone uses `git -c http.version=HTTP/1.1`, which
  is the form that clones anonymously from this container; the default HTTP/2
  path answers 401.
- The checkout asserts `HEAD_SHA` against the target and aborts on a mismatch,
  before anything is installed.
- Versions are read from the built environment (`vllm.__version__`,
  `importlib.metadata`), never transcribed from a tag or a release number, and
  `git describe` is recorded from the fresh clone rather than from the reference
  checkout, which #2589 §7.3 measured as carrying a tag upstream has deleted.

## 3. Risks

- **`EXT_RC=1` is expected and is not a target defect.** `VLLM_USE_PRECOMPILED=1`
  has no wheel for aarch64, so `import vllm._C` raises `ModuleNotFoundError` in
  every job on this fleet. Reading it as a failure of the target would be the
  wrong verdict.
- **A green install and import is not `gateable = yes`.** AGENTS.md is explicit
  that an oracle is gateable only once it demonstrably builds and **runs** the
  model. This wave can answer the build-and-import half on a box whose container
  exposes no GPU, and it says so rather than letting a green import promote the
  target.
- **A registered architecture is not a running model.** Resolving the class
  proves the module imports against the installed torch. It proves nothing about
  weights, kernels or output.
- **An instrument whose failure looks like a result** is this row's recurring
  trap. A leg that did not run must read as absent, not as a pass, which is why
  the script prints presence probes beside the return codes.

## 4. Gates

This wave changes documents and records rather than product code, so its gate is
the record and style gate. It claims no numeric parity result, regenerates no
golden dump and re-baselines no benchmark.

```sh
scripts/agent-preflight.sh --staged
python3 scripts/check-commit-style.py
python3 scripts/check-agent-record.py
```

`agent-ready.py` runs a compile step and #2589 measured `--staged` producing no
output for thirty minutes on this box under concurrent preflights, so the
commit-scoped checkers are run individually rather than a timeout being reported
as a pass.

## 5. Stop conditions

- **Do not advance the pin**, whatever the measurement says. The queue decides
  that, not the install.
- **Do not advance the pin on an install-and-import result.** Carried unchanged
  from #2589 §5.
- Report and stop if the target does not install. That the repair does not reach
  a `qwen4_exp`-carrying revision is then the finding, and it belongs to the
  developer.
- Take one lease, not two. Report `main` as unmeasured rather than spending a
  second lease on it; the earlier revision is the one that serves this row and
  carries less drift.

## Owed

- ~~The run half of gateability: a model served from the built target inside a
  lease, on a device whose container exposes the GPU (#2593, carried from
  #2589 C7).~~ **Discharged 2026-09-03** by wave RUNHALF
  ([#2611](https://github.com/mudler/vllm.cpp/issues/2611),
  [`upstream-sync-headpin-runhalf.md`](upstream-sync-headpin-runhalf.md),
  [`../sync/2026-09-03-e126687-runhalf.md`](../sync/2026-09-03-e126687-runhalf.md)):
  `SRCBUILD_RC=0`, `EXT_PRESENT=True`, `RUN_RC=0`, `COMPILED_RC=0` on
  `thor:gpu0`. The pin still does not move, and `qwen4_exp` itself does not run
  there ([#2626](https://github.com/mudler/vllm.cpp/issues/2626)).
- The PORT-NOW queue for `5559679229..e126687a9a`, which is #2589's 206 entries
  plus everything between `db92053e97` and this target (#2593).
- Whether upstream `main` installs under the same repair (#2593). Deliberately
  unmeasured; see §5.
- `scripts/check-pr-size.py` cannot classify `.agents/scripts/**`, so the two
  job scripts this wave ran could not be committed beside the three already
  tracked there ([#2607](https://github.com/mudler/vllm.cpp/issues/2607)). Until
  that lands, this wave's A/B control is recorded as a verbatim `diff` rather
  than as two readable files, which the sync report's C10 states as a
  limitation.
