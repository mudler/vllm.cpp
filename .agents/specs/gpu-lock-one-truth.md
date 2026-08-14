# Spec — the GPU mutex has exactly one truth

Issue: [#777](https://github.com/mudler/vllm.cpp/issues/777)
Row: — (unplaced protocol/environment defect; intake table in
[`roadmap_v1.md`](../roadmap_v1.md))
State: `DONE`

## Scope

This repo carried **two GPU mutexes**, and the documented one coordinated with
nothing.

| Path | Reached by |
|---|---|
| `$HOME/gpu.lock` | `dgx-gdn-packed-bridge-ab.sh`, `laguna_longctx_bench.sh`, `lmcache/run_output_invariance.sh`, and the run command documented in 11 capture scripts |
| a private lock file | `dgx-gdn-packed-component.sh`, `dgx-online-serving.sh` (executing), plus the run command documented in 5 more scripts and one planned-command manifest |
| `${GPU_LOCK}` | `.env.example`, `.agents/environment.md`, `.agents/coordination.md`, `.github/pull_request_template.md` |

`.env.example` shipped `GPU_LOCK=` **empty**, so the value was whatever each
developer filled in, and `.agents/coordination.md` documented the developer
profile's value as the private path — i.e. the instruction surface named the
mutex that serialised with nobody. Follow the docs and you took one file; run a
harness and you took another.

Developer-directed resolution: **`$GPU_LOCK` is the documented path everywhere,
defaulting to `$HOME/gpu.lock`.**

## Design

One expression, two spellings, everywhere:

```sh
${GPU_LOCK:-$HOME/gpu.lock}
```

- `.env.example` ships the default rather than a blank, so a filled-in `.env`
  cannot quietly diverge from a fresh copy.
- Every script that takes a lock resolves it through that expression, so the
  variable and the scripts cannot disagree.
- Every script that *documents* a run command spells the same expression, so a
  copy-paste takes the one truth.
- The two instruction surfaces state the variable **and** its default, so
  someone whose `.env` predates this change can repair it from the docs alone.

## Risks

- **A `.env` is untracked and is the developer's file.** The default only helps a
  fresh copy; an existing `.env` naming another path must be changed by hand.
  Called out in `.env.example`, `.agents/environment.md` and the PR body.
- **Evidence-record semantics.** `dgx-gdn-packed-component.sh` records the lock
  path into `component-order.log`, and `gdn_packed_component.py` pinned the old
  literal. The path is now host-resolved, so the validator matches
  `path=(?P<path>\S+)` and additionally requires the acquire and the release to
  name the **same** file — a run that acquired one path and released another
  never held one lock, which the literal pin could not express. Mutation-proven
  below.

## Tests

`tests/scripts/test_gpu_lock_one_truth.py` — five structural rules:

| Rule | Asserts |
|---|---|
| A | no tracked `scripts/**` or `tools/**` `.sh`/`.py` names a bare lock path (the retired private path never; the `$HOME/gpu.lock` default only on a line that also names `GPU_LOCK`) |
| B | any such file that invokes `flock` or opens the fd one is taken on must spell `GPU_LOCK` |
| C | `.env.example` ships `GPU_LOCK=$HOME/gpu.lock` |
| D | `.agents/coordination.md`'s `## GPU scheduling` section names `${GPU_LOCK}` and `$HOME/gpu.lock`, and not the retired path |
| E | `.agents/environment.md`'s GPU-mutex bullet states the variable **and** its default |

Registered in `scripts/agent-preflight.sh` (`SUITES`) and in `.github/workflows/ci.yml`.

Historical records — `.agents/benchmark-record.md`, `.agents/parity-ledger.md`,
`.agents/completed/state-events/`, and every `.agents/specs/*.md` recording a past
run — are **deliberately not scanned and not edited**. They say what was actually
used at the time and are evidence; rewriting them would falsify the record.

## Gates

```sh
python3 tests/scripts/test_gpu_lock_one_truth.py
scripts/agent-preflight.sh --staged
python3 -m unittest discover -s tests/tools -p 'test_*.py'   # PYTHONPATH=repo root
```

No CUDA/GPU/SACRED/oracle gate is implicated: the change reaches no forward pass
and no product source under `src/` or `include/`.

## Outcome

**What the bug actually was.** Not "some scripts forgot the variable". The
instruction surface itself named the wrong file: `.agents/coordination.md`
recorded the private path as the developer profile's `${GPU_LOCK}`, and
`.env.example` shipped the variable blank, so following the documentation
correctly produced an unserialised run. `scripts/lmcache/run_output_invariance.sh`
already had the right idiom — and that safe default is defeated precisely when
the variable is set, which is what the docs told you to do.

**Why no runtime check could catch it.** A `flock` on the wrong file *succeeds* —
that is what a mutex does. The failure is silent and surfaces only as timing
noise, and it does not present as "my number is wrong", it presents as "someone
else misbehaved". It cost a full standalone Marlin series: every absolute timing
was downgraded to an upper bound and only the interleaved ratios survived,
because contention hits both arms alike. `.agents/specs/muse-glimmer.md` records
an earlier bite of the same defect.

**What the enumeration got wrong.** The issue counted 10-11 "scripts hardcoding
`$HOME/gpu.lock`". Most of those occurrences are *documented run commands in
header comments*, not lock acquisition — only three shell scripts actually took
that path. The larger half of the split was the opposite direction: two of the
biggest benchmark drivers executed against the private path. Fixing only the
`$HOME` half would have left the split fully intact and made the new test a fig
leaf, so both halves are in this change.

**The gate earned itself before it landed.** Rebasing this branch onto `main`
picked up `scripts/dspark-paired-e2e.sh` and `scripts/marlin-moe-standalone.py`,
merged the same day, both hardcoding `$HOME/gpu.lock` and both carrying a comment
warning the reader off the other path. They are *correct* about which file to
take and they still fail rule A, which is the point: the drift is not people
choosing the wrong lock, it is the path being writable without the variable that
resolves it. The suite caught them on its first run against a tree it had never
seen.

**Why the default is set the way it is.** `$HOME/gpu.lock` and not the private
path because it is what the harness scripts and the operator campaign scripts
already used, so it is the value with existing holders — moving to the other
path would have been the change that silently unserialised everyone else.
`.env.example`'s "leave it empty when your setup does not have the thing"
convention does not apply here: every box that runs GPU work needs a mutex, and
`$HOME/gpu.lock` always resolves.
