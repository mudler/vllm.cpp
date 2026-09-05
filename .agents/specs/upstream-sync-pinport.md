# Sync cycle `cdefd9d499`, steps 5 to 7 (wave PINPORT)

Issue: [#2524](https://github.com/mudler/vllm.cpp/issues/2524).
Predecessor: [#2490](https://github.com/mudler/vllm.cpp/issues/2490), steps 1 to
4, whose report is [`../sync/2026-09-01-cdefd9d.md`](../sync/2026-09-01-cdefd9d.md).
Guide: [`../upstream-sync.md`](../upstream-sync.md) §"The sync cycle".

## Now

The pin has **not** advanced and does not advance in this wave. The active parity
pin remains `5559679229bc961848b121ccdeaa8fa5d79bec98`.

## 1. Scope

The cycle's steps 5, 6 and 7 against target
`cdefd9d4997f00da72dc6245cc60678b50761b7e`. The report classified 1077
mirrored-surface commits out of a 1566-commit range: 315 PORT-NOW, 356
INVENTORY, 406 IGNORE.

**This wave does three things and deliberately does not attempt the fourth.**

1. Reconcile the deviations in [`../porting-inventory.md`](../porting-inventory.md)
   §9 that this advance conflicts with. The guide requires it in the same cycle,
   because a deviation must describe current truth.
2. Re-derive the PORT-NOW queue's dispositions against **this tree**, which the
   report never did and says so (its carry-over C7). Measure the rate at which a
   PORT-NOW entry is a real code gap here.
3. Answer the gateability question for the target, or name the issue that owes
   it.

It does **not** work the 315-item queue to completion. That is more than one
wave, and the guide's documented behaviour for a cycle that stalls is to keep
the old pin and record the remainder as carry-over.

### Out of scope

- Advancing the pin. See §6.
- `qwen4_exp_*`, `src/vt/cuda/cuda_ops.cu` and `qsa_block.cpp`: three sibling
  waves are live in those files (#2509, #2488, and the correctness-debt cluster).
- Tiering the 356 INVENTORY entries. Owed by #2524.
- Any feature work. The guide forbids mixing a sync cycle with feature work in
  the same commit.

## 2. Upstream anchors

Read in the reference checkout `${VLLM_SOURCE}` at the revision each names, not
in a fork and not from a cited line in another document. Anchors read in a fork
are wrong at the pin, and that error has been paid for here.

| Anchor | Revision | What it settles |
|---|---|---|
| `6adad08767` | merged squash of vllm#51655 | Muse Glimmer, deviation 16's discharge |
| `ad5d29db70` | merged vllm#50210 | Qwen3.5 text-only arms, deviation 17's anchor |
| `9035151d6c` | merged vllm#51255 | `dots3_note`, deviation 18's origin |
| `da0b2d8b17` | vllm#53517 | the one `dots3_note` commit after W3's re-read |
| `cdefd9d499` | target | the revision every claim here is measured against |

All three discharge commits were verified to be ancestors of the target with
`git merge-base --is-ancestor`, and every registry key was counted at both
revisions rather than assumed.

## 3. Design

### 3.1 The unit of a header bump is the FILE, not the PR

A ported file's header records the upstream revision it matches. Bumping it to
TARGET asserts reconciliation against TARGET. Porting one of the several
PORT-NOW commits that touch an upstream path and then bumping that path's header
makes the header lie about the others.

**Rule adopted here:** bump a file header to TARGET only when *every* PORT-NOW
commit touching its upstream path is reconciled. Otherwise port the behaviour,
leave the header, and record the partial reconciliation.

Measured over the queue: 315 commits touch 212 distinct ported upstream files.
Only 75 of those files are touched by exactly one PORT-NOW commit. The hot files
are the shared seams and cannot be bumped by a single-item port:
`vllm/config/vllm.py` is touched by 22, `mla_attention.py` by 19,
`v1/core/sched/scheduler.py` by 18, `v1/worker/gpu_model_runner.py` by 17 and
`v1/core/kv_cache_utils.py` by 13.

### 3.2 A PORT-NOW disposition is not evidence of a code gap here

The report derived its dispositions by reading upstream diffs. It did not read
this tree, and it records that as carry-over C7 with the exact failure mode: a
behaviour-neutral API widening upstream whose behaviour lands in an unported
file.

Spot-checking found the converse shape as well, which C7 does not name: an
upstream **defect fix** whose defect this tree's structure makes impossible, so
there is nothing to port. §5 records the measurement.

### 3.3 Ordering constraints honoured

This wave ports nothing from the KV cache layout series, nothing from either
revert pair, and nothing from the DSpark validation triple. Since it ports no
member of any ordered group, no ordering constraint can be violated by it. The
constraints are restated in #2524 for the wave that does take them.

## 4. Risks

- **Sampling a queue and generalising.** A rate measured on 32 of 315 entries is
  an estimate with a real interval, not a count. It is reported as an estimate,
  and it justifies re-deriving the queue rather than deleting entries from it.
  No PORT-NOW entry is reclassified in the report on the strength of the sample.
- **A grep asserts which change is compiled; only a compiler says it compiles.**
  Any code change here carries a build, and the rc is read literally.
- **Merging `main` has falsified this tree's own prose through conflict-free
  hunks.** Every count and claim in this spec is re-read after each merge, not
  only the conflicted hunks.
- **Keyed records.** For a concurrent edit to `porting-inventory.md`, take the
  complete target-branch version and re-apply the scoped edit. Never accept an
  automatic three-way merge of a keyed record.

## 5. Tests and gates

- `scripts/agent-preflight.sh` before edits, `--staged` before each commit.
- Commit-scoped record and style gates run individually.
  `scripts/agent-ready.py` runs a compile step and has timed out at ten minutes
  here; a timeout is never reported as a pass.
- Known-red and not owned by this wave: `test_cpu_x86_llamacpp_floor` (#2448),
  `sanitize-cpu` (#2435), `windows-msvc-*` (#584), `test_cpu_threadpool` (#631).

**This wave landed NO product code, and the reason is a measurement rather than
a choice.** It found one real correctness bug (#2527) and wrote the reconciled
fix, then could not compile it: the dev box carried a load average near 50 with
two other worktrees compiling and about 1 GiB free, and a third concurrent build
is this repository's recorded OOM failure. A grep asserts which change is
present; only a compiler says it compiles, so the fix was reverted out of the
branch and attached to its issue instead of landing unverified. A full
`agent-preflight.sh --staged` was started and produced no output in 25 minutes
under that load, so the commit-scoped record and style checkers were run
individually rather than reporting a timeout as a pass.

Because this wave changes documents and records rather than product code, its
gate is the record and style gate, not a token or throughput gate. Nothing here
claims a numeric parity result, so no golden dump is regenerated and no
benchmark is re-baselined. Step 6's re-measurement obligation is carried, not
discharged: it belongs to the wave that ports the queue.

## 6. Stop conditions

- **Do not advance the pin while the PORT-NOW queue is unworked.** A pin
  advanced over an unported queue is worse than no advance: every downstream
  gate then measures against an oracle the tree does not mirror.
- **Gateability is a precondition for step 7, not an afterthought.** An oracle
  is gateable only once it demonstrably builds and runs the model. Constructing
  a config proves nothing.
- Stop and report rather than widen scope if the queue re-derivation shows the
  report's lists need regenerating; that is a step-3 defect and belongs to a
  cycle, not to a patch.

## Owed

- The 315-item PORT-NOW queue, in the dependency order the report's §11 names
  (#2524).
- Re-derivation of every PORT-NOW disposition against this tree (#2524).
- Gateability of `cdefd9d499`: a measured build that runs a model, and the
  measured `vllm.__version__` carrying its `+g<sha>` segment (#2524, #2490).
  The first attempt ran in an `rc` lease on `orin:gpu0` and its partial evidence
  is recorded in the sync report §9.1: the clone is confirmed at the target with
  complete history (`REVCOUNT=20692`, `SHALLOW=false`), so #1185's
  `setuptools_scm` fallback does not apply, and `git describe` puts the target
  233 commits after `v0.28.1rc0`. No `vllm.__version__` was measured, so
  `gateable` stays `no` and that tag distance must NOT be transcribed into the
  `parity-pin` block.
  **The job then finished and found a harder blocker (§9.2): `BUILD_RC=0` but
  `IMPORT_RC=1`, because the target's runtime requirements do not install on
  aarch64.** `2d7f42b4f3` vllm#52801 promoted `instanttensor` from a test
  dependency to a runtime one without a platform marker, it ships no aarch64
  wheel, and it fails to build from source. Every fleet device is aarch64, so
  **advancing the pin would move the oracle to a revision this fleet cannot
  install.** That is a stronger reason to hold the pin than the unported queue.
- Step 6 re-measurement of any vLLM baseline at the target before comparison
  (#2524).
- Tiering the 356 INVENTORY entries (#2524).
- The three genuinely wrong upstream anchors the report's §8 group C names.
- [#2527](https://github.com/mudler/vllm.cpp/issues/2527), the priority-scheduler
  silent-skip bug this wave found while re-deriving the queue. The reconciled fix
  is written and attached to the issue, and it is UNCOMPILED: this wave could not
  build, so it did not land product code it could not gate. The issue owes a red
  test, a build and a focused green.
- [#2531](https://github.com/mudler/vllm.cpp/issues/2531), the async-scheduling
  placeholder underflow, the second live defect the re-derivation found. Its
  `assert` is compiled out under `NDEBUG`, so a release build goes negative
  silently. Same reason it did not land here: no build was possible.
