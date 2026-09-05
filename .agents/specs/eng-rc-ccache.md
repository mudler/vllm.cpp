# `ENG-RC-CCACHE` — the fleet's mandatory build cache is a no-op, because CIFS refuses `symlink(2)`

Issue: [#2473](https://github.com/mudler/vllm.cpp/issues/2473).

This row has **no matrix row and therefore no lifecycle state**. It repairs a
lease-script build recipe and the fleet document that recipe is read from. Its
state is the issue it closes and the gate that holds it.

The defect is carried under `## Owed` in
[`ltx25-render-confirm.md`](ltx25-render-confirm.md), which is where it was
found and where a 1404 s build was paid for nothing.

## Scope

IN SCOPE:

- Establish, by probe rather than by inference, why `ccache` records zero hits,
  zero misses and zero stores on this fleet when it is configured exactly as the
  host usage sheet requires.
- Move `CCACHE_DIR` off `/workspace` in the two lease scripts that carry the
  build recipe, and give the cache a persistence path that CIFS can actually
  serve.
- Make "the cache did nothing" a refusal instead of a log line, so the next
  occurrence stops a job rather than costing it 23 minutes silently.
- Record the finding and the corrected recipe in
  [`environment.md`](../environment.md), which is the document a lease author
  reads before writing a job.
- Measure the result: a cold build and a warm build of the same tree on
  `dgx:gpu0`, with the hit rate between them.

OUT OF SCOPE:

- **The host usage sheet itself.** `rc describe <device>` serves host state, not
  repository state. Its instruction to "keep its cache on the NAS" is what this
  row proves wrong, and correcting it is the operator's, which #2473 asks for.
  Nothing here edits anything outside this repository.
- Every other lease script under `scripts/`. Fifteen of them configure CMake in a
  lease and only `ltx25-render-confirm.sh` sets `CCACHE_DIR` today. The gate this
  row adds covers all of them; converting them is not this row's work.
- The binary-artefact cache keyed on `SRC_SHA` that `ltx25-render-confirm.sh`
  already carries. That skips the build entirely on an unchanged tree and is
  complementary: `ccache` is what makes a CHANGED tree cheap.

## What was measured, and how the four candidates were separated

Four causes were on the table, and each is answered by a measurement rather than
by argument. The whole probe is two `rc` jobs and under four minutes of compute.

| # | Candidate | Verdict | The measurement that decided it |
|---|---|---|---|
| 1 | The CMake launcher never reached the compile line | **REFUTED** | `-DCMAKE_CXX_COMPILER_LAUNCHER=ccache` on a nine-unit project put `ccache` on 9 lines of the verbose build and produced `Cacheable calls: 9 / 9 (100.0%)` |
| 2 | `ccache` was not on `PATH` when CMake resolved it | **REFUTED** | the same run resolved it to `/usr/bin/ccache`; the reference job also asserts `command -v ccache` and exits 37 otherwise |
| 3 | `CCACHE_DIR` on CIFS is unusable | **CONFIRMED** | below |
| 4 | A wrapper or generator bypassed the launcher | **REFUTED** | same as 1: the launcher is on the Ninja compile line and the counters move |

### Cause 3, and the exact mechanism

`ccache` 4.9.1 takes every cache and stats lock by creating a **symlink**. The
`/workspace` mount is CIFS with `nounix`, and `symlink(2)` on it returns
`EOPNOTSUPP`:

```text
mount:    //192.168.68.102/Data[/rc] cifs rw,...,soft,nounix,mapposix,...
symlink:  NO  (Operation not supported)
hardlink: YES
rename:   YES
flock:    YES
```

Every lock then fails, and the failure is per bucket:

```text
Acquiring /workspace/.../ccdir/4/c/stats.lock
Could not acquire /workspace/.../ccdir/4/c/stats.lock: Operation not supported
Failed to acquire lock /workspace/.../ccdir/4/c/stats.lock
Failed to acquire lock for /workspace/.../ccdir/4/c/stats
```

**The stats lock fails too, which is why the counters read zero.** This is the
part worth stating precisely, because the reading is misleading in a specific
direction: a cache that was consulted and empty records MISSES, so
zero-of-everything looks like proof that `ccache` was never invoked. It is not.
`ccache` was invoked on every translation unit, and every store and every
counter update was refused. The two readings are identical and the conclusions
are opposite.

**It is worse than a no-op.** Each failed acquisition costs a retry timeout of
roughly 0.4 to 0.6 s, and `ccache -s` walks all 256 stats buckets.
`ccache -s -v` against a `/workspace` cache produced **no output at all** in the
probe window after 278 consecutive lock failures, and the job had to be killed.
A cache on the NAS adds time to every build it cannot accelerate.

The residue is visible without running anything. `/workspace/ccache` holds a
complete 16x16 bucket tree, its second-level directories carry mtimes from
inside the 07:58-08:22 window of the 1404 s build, and it contains **one file**,
`tmp/.cleaned`, and zero bytes.

The same CIFS limitation is already on this repository's record: it is why a
staged CUDA toolkit loses its SONAME symlink (#2220). This row is that fact
arriving at a second consumer.

## Design

Two changes, and the second is what makes the first durable.

**1. The cache moves to local disk and persists through ccache's own remote
storage.**

```sh
export CCACHE_DIR=/root/ccache                              # local, not CIFS
export CCACHE_REMOTE_STORAGE=file:/workspace/ccache-remote  # survives the container
```

`CCACHE_DIR` must be local because that is the path that needs `symlink(2)`.
`CCACHE_REMOTE_STORAGE` may be on `/workspace` because ccache's `file` backend
stores through `open` plus `rename`, which this mount supports, and takes no
symlink lock. The distinction is not a guess: the probe drove both.

| Arm | `CCACHE_DIR` | Remote | Result |
|---|---|---|---|
| local | `/tmp` | none | 1 hit / 1 miss over 2 calls; ccache itself is healthy here |
| cifs | `/workspace` | none | 278 lock failures, `ccache -s -v` never returned |
| remote-seed | `/tmp` (empty) | `file:/workspace/...` | 18 s, 0/8 hits, 16 remote writes |
| remote-cold | `/tmp` (**fresh, empty**) | the same store | **5 s, 8/8 remote hits, 0 misses** |

The `remote-cold` arm is the one that carries the design: its local cache was
created empty, so every hit it recorded had to come off the NAS.

A tarball of the local cache on `/workspace` was measured as a working
alternative (seed 4 s, save 1 s, restore 0 s, warm 2 s) and rejected: it needs an
explicit save step, it has no eviction, and a job killed by `--max-runtime`
saves nothing. Remote storage needs no save step at all.

**2. Zero counters become a refusal.**

The reference job printed `ccache after: ...` and continued. That line is exactly
what a dead cache produces, and nobody read it for a month. The recipe now
asserts, after a build that actually compiled, that `ccache` recorded a non-zero
number of cacheable calls, and exits when it did not. It also refuses at setup
if `CCACHE_DIR` sits on a filesystem that cannot create a symlink — the precise
capability that fails, checked in milliseconds, named in the message.

**3. `scripts/check-lease-ccache.py`** refuses any shell script under `scripts/`
that points `CCACHE_DIR` at a `/workspace` path. Fifteen lease scripts configure
CMake and any of them can acquire the broken recipe by copying a neighbour, which
is how the recipe spread in the first place. The gate is a static read of
`scripts/*.sh`; it writes nothing and locks no shared file.

## Tests and gates

- `tests/scripts/test_check_lease_ccache.py` — the checker's mutation suite. It
  must go RED when a fixture script sets `CCACHE_DIR` under `/workspace`, and
  RED again when the checker's own refusal is deleted.
- `python3 scripts/check-lease-ccache.py` — green on the tree.
- `scripts/agent-preflight.sh` — the full gate, with the new checker and suite
  registered.
- `python3 scripts/check-tree-compiles.py` — `main` carries a compile gate and
  this change must not red it.
- The measurement gate: one `rc` job on `dgx:gpu0` that builds the same tree
  twice, cold and warm, and reports both wall clocks and the hit rate.

## Evidence

| What | Where |
|---|---|
| the defect, as found | `/workspace/ltx25-render-confirm/run/20260901T075837Z/ccache-{before,after}.txt`, `rc` job `93a60151-7d4d-4718-842c-ef724208be0e` |
| causes 1-4 separated, and the CIFS mechanism | `/workspace/eng-rc-ccache/probe/strix/`, `rc` job `9b287a1f-d94d-476d-9732-0f649509cdd8` |
| the remedy measured on a toy | `/workspace/eng-rc-ccache/probe2/strix/`, `rc` job `e4793984-10e8-4b2e-b6a7-7f931d6d40fe` |
| the remedy measured on this tree, CPU arm | `/workspace/eng-rc-ccache/build-ab-cpu/strix/`, `rc` job `dec8ab1f-4522-4dd7-8bcb-34e885c2c3e5` |
| the remedy measured on this tree, CUDA arm | `/workspace/eng-rc-ccache/build-ab/dgx/`, `rc` job `54b75579-3809-4856-922b-3e8b077859eb` |

## Stop conditions

- Stop and report `NEEDS_DECISION` if the remedy requires a change outside this
  repository. The usage sheet is the operator's; #2473 asks for it and this row
  does not edit it.
- Stop if `ccache`'s remote storage turns out to be unusable on `dgx:gpu0` even
  though it works on `strix:gpu0`. Both mount the same share, so a divergence
  would mean the mount options differ per host and the finding does not
  generalise.

## The measurement on this tree

`strix:gpu0`, `rc` job `dec8ab1f-4522-4dd7-8bcb-34e885c2c3e5`, base
`a56890f960cdc557727b7e7c1415d2067877ff09`, target `vllm`, 540 objects, `-j 16`.

| Arm | Local `CCACHE_DIR` | Build | `ccache` |
|---|---|---|---|
| cold | created empty, remote store created empty | **165 s** | 538 / 538 misses, 1076 writes |
| warm | **created empty again**, same remote store | **56 s** | **538 / 538 hits (100.0%)** |

**2.95x, and every hit came off the NAS.** The warm arm's local storage line
reads `Hits: 0 / 538, Misses: 538 / 538`: its own cache had nothing, so all 538
hits were served by remote storage. The build tree was deleted between the arms,
so ninja recompiled every object rather than skipping it. Nothing else differs
between the two arms, which is what makes the saving attributable.

The remote store is 1077 files and 32.0 MB for this target. `/workspace` is the
same folder on `dgx`, `thor`, `orin` and `strix`, so the persistence crosses
leases AND hosts: a job on one box can hit a cache another box filled, subject
to ccache's own key, which hashes the compiler binary and so cannot collide
across architectures.

### The three readings side by side, which is where the diagnosis closes

| Configuration | Hits | Misses | Stores |
|---|---:|---:|---:|
| NAS `CCACHE_DIR`, the usage sheet's instruction | 0 | **0** | 0 |
| local, cold | 0 | **538** | 538 |
| local, warm from the NAS remote store | **538** | 0 | -- |

**Zero misses was never "the cache was empty".** A working cold cache records
538 of them, on the same tree, in the same job shape. Zero-of-everything is a
reading the broken configuration is uniquely able to produce, and no healthy
configuration can.

This row was briefed with the opposite inference -- that zero misses meant
`ccache` was never invoked, because a cache consulted and empty would record
misses. The first half of that is sound and the conclusion does not follow: it
silently assumes the counters can be written. They are stored inside
`CCACHE_DIR` under the same symlink lock as the cache entries, so on this mount
the instrument and the thing it measures fail together. The measurement that
settled it was `Cacheable calls: 9 / 9 (100.0%)` from the CMake launcher probe,
which refuted "never invoked" directly rather than by argument.

**The generalisation worth keeping: when a counter reads zero, ask whether the
counter could have been written at all before reading the zero as a measurement.**
A dead instrument and a null result are the same bytes.

### The mount advertises a capability it does not have

The option line carries `symlink=native` while `nounix` is what actually
governs, so a reader who checks the options concludes symlinks work. Only
attempting one tells the truth, and the errno varies by client: `EOPNOTSUPP`
(`Operation not supported`) from the `vers=3.1.1` mount inside the worker,
`EIO` (`Input/output error`) from the `vers=3.0` mount on the devbox. Both are
the same absent capability. This is why the shipped guard ATTEMPTS a symlink
rather than parsing the mount options.

The equivalent CUDA measurement on `dgx:gpu0` -- the host where the 1404 s was
paid -- is `rc` job `54b75579-3809-4856-922b-3e8b077859eb`, queued behind other
work at the time of writing. `dgx` and `strix` mount the same share with the same
options, and the CIFS capability probe runs on both, so the finding is expected
to reproduce; until that job returns, the cold-versus-warm ratio above is
measured on the CPU arm and is stated as such.

## Now

`ACTIVE`. The cause is established, the remedy is measured on this tree at 2.95x,
and the gate is green. The CUDA arm on `dgx:gpu0` is the remaining measurement.

## Owed

- **The usage sheet still tells every reader to put the cache on the NAS.** It is
  host state and this repository cannot edit it. #2473 carries the request with
  the evidence. Until it changes, a lease author who follows the sheet rather
  than `environment.md` reproduces the defect. Owner: the operator.
- **Thirteen other lease scripts configure CMake with no launcher at all.** They
  pay a cold build every time. The gate this row adds stops them acquiring the
  WRONG recipe; it does not give them the right one. Owner: unowned; sizing is
  one pass with no lease.
- **The share is `soft`-mounted, so a streaming read off it can fail.** The
  first A/B attempt died with `gzip: stdin: Resource temporarily unavailable`
  mid-archive on a 100 MB tarball whose digest was intact (`rc` job `6a737c16`).
  The A/B harness now stages local with retries and verifies sha256 before
  unpacking. The lease scripts under `scripts/` still `tar xzf` straight off
  `/workspace` and carry the same hazard. Owner: unowned.
- **The remote store has no eviction.** `ccache --trim-dir` is not run by
  anything. It grows until somebody trims it. Owner: unowned.
