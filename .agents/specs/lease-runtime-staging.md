# Staging a CUDA runtime a lease can start

Row: `ENV-LEASE-RUNTIME-STAGING`.
Issue: [#1146](https://github.com/mudler/vllm.cpp/issues/1146).

## Scope

Record one measurement and correct the records it falsifies.

The measurement: a relocated CUDA Python runtime, staged on `/workspace`, starts
inside an `rc` lease on `thor:gpu0`. It imports `torch`, initializes CUDA, runs a
bf16 matmul, and compiles and executes a Triton kernel.

In scope:

- This spec, holding the four walls, the working recipe, and the evidence.
- The correction to `.agents/environment.md`, which records the leased worker as
  carrying no `python3`, no `pip` and no `gcc`.
- The correction to `.agents/specs/mtp-k-gt-1.md` and
  `.agents/specs/gpu-lease-methodology.md`, which both carry
  [#1129](https://github.com/mudler/vllm.cpp/issues/1129)'s reasoning as the live
  cause of the oracle blocker.
- One appended row in `.agents/issue-index.md` for #1146.

Out of scope:

- Any measurement on `dgx:gpu0`. See `## What this does NOT establish`.
- Staging the pinned vLLM oracle. Only `torch` and `triton` are staged, and the
  oracle is the thing #1129 actually blocks.
- Any product code. This row touches records and documents only.
- Any checker change. This row changes no checker semantics, so it owes no
  red-before mutation under `## Changing the rules or a checker`.
- Editing #1129's existing index row. That file is append-only.

## What #1129 recorded, and what is now false

#1129 is closed. It records that no lease-compliant path can run a CUDA Python
runtime on this fleet, because the leased worker "cannot start Python". Its
Direction 2 probe recorded `python3=ABSENT`, `pip=ABSENT`, `gcc=ABSENT`,
`curl=ABSENT` and `git=ABSENT` inside the worker.

That reading came from one probe on `dgx:gpu0`, job
`ff28ada1-0cd3-4867-bf9b-f67050d0608b`, taken on 2026-08-17. Three of its five
names are now contradicted, and the contradictions come from TWO different boxes.
Keep them apart, because a name proven on one worker proves nothing about the
other.

On `thor:gpu0`, job `8beba132` reports `python3` and `gcc` present. The worker
runs as `uid=0(root)`, carries `/usr/bin/python3`, `/usr/bin/gcc` and
`/usr/bin/apt-get`, and installs a package as root (`APT_UPDATE_RC=0`,
`APT_INSTALL_RC=0`). Job `fd5654c0` then ran `python3 -m pip install --target`
there to completion, so `pip` is present on `thor:gpu0` too.

On `dgx:gpu0`, job `609c4944-594b-4617-967b-fb3d3d8c09f6` invoked
`python3 -m pip install --quiet --target /workspace/oracle-probe/site torch`, so
`python3` and `pip` are present on that worker as well. Read that job no further:
it ended on `max_runtime exceeded (35m0s)`, so it is evidence that `pip` STARTS
on `dgx:gpu0` and not that this install finished.

`curl` and `git` were probed on neither worker after #1129, so this row says
nothing about them.

The measurement was taken correctly. The fleet changed under it. So the recorded
cause no longer holds, and the three fleet-side fixes #1129 names as the only
ways forward are not the only ways forward. The image is provisionable per job.

## Measured, on `thor:gpu0`, 2026-08-17

```
torch.__file__   = /workspace/oracle-probe/site/torch/__init__.py
torch.__version__= 2.13.0+cu130
cuda available   = True
device count     = 1
device 0         = NVIDIA Thor
capability       = (11, 0)
CUBLAS_OK          (bf16 1024x1024 matmul executed)
triton.__version__ = 3.7.1
TRITON_JIT_OK    = 4096.0 (expect 4096.0) PASS
PROBE5_RC=0
```

## The four walls, in the order they appear

1. **The runtime must be installed FROM the worker.** The submitting host here is
   `x86_64` and the workers are `aarch64`, so a local `pip install --target` onto
   the NAS writes the wrong architecture into the exact path the worker imports
   from, and it fails there as a confusing import error. Run the `pip --target`
   inside `rc run`.
2. **`Python.h` is absent.** Triton's JIT shells out to `/usr/bin/gcc` to build
   `cuda_utils.c` and dies with `fatal error: Python.h: No such file or
   directory`. `apt-get install -y python3-dev` succeeds as root, and the headers
   copy to `/workspace/oracle-probe/pyhdr` (1.6 MB) so the next job does not
   reinstall them. Put them on `CPATH`.
3. **The NAS grants no exec bit, and `TRITON_PTXAS_PATH` does not save you.** The
   mount presents `file_mode=0664`, so Triton cannot execute its own
   `ptxas-blackwell`. Setting `TRITON_PTXAS_PATH` to a `/tmp` copy is **not
   sufficient**: it redirects only the plain `ptxas`, while Triton selects the
   Blackwell variant from its own package directory. The whole `triton` package
   (651 MB) has to sit on a filesystem that grants exec bits.
4. **So `PYTHONPATH` is ordered, not single.** `PYTHONPATH=/tmp/tp:/workspace/oracle-probe/site`
   puts the exec-capable `triton` first and leaves the 4.5 GB of torch on the NAS
   where it costs nothing to re-stage.

## The working recipe

```sh
# once per worker container
apt-get update -qq && apt-get install -y -qq python3-dev
mkdir -p /tmp/tp && cp -a /workspace/oracle-probe/site/triton /tmp/tp/
chmod -R +x /tmp/tp/triton/backends/nvidia/bin/

export PYTHONPATH=/tmp/tp:/workspace/oracle-probe/site
export CPATH=/workspace/oracle-probe/pyhdr/python3.12:${CPATH:-}
```

## The prebuilt-wheel route is closed for our pin

The spec says the oracle needs `nvcc`. A reader can reasonably ask whether a
prebuilt wheel sidesteps that, so the question was measured on 2026-08-17 rather
than argued. It does not.

**An aarch64 vLLM wheel exists in general, so the architecture is not the
obstacle.** On the worker, `python3 -m pip download --no-deps vllm` fetched
`vllm-0.27.1-cp38-abi3-manylinux_2_28_aarch64.whl`, 307,180,998 bytes,
`VLLM_DL_RC=0`. The same job reported `aarch64 linux-aarch64` from
`platform.machine()` and `sysconfig.get_platform()`.

**Our pin is not reachable that way.** `https://wheels.vllm.ai/nightly/vllm/` is
a PEP 503 index, and it lists FOUR wheels for exactly ONE commit, `402547d7f`,
aarch64 and x86_64, each also in its percent-encoded form. It is a moving
pointer and not an archive, so the pin
`5559679229bc961848b121ccdeaa8fa5d79bec98` is absent from it. The pin is also a
development commit, `0.23.1rc1.dev1511+g555967922`, so it is not on PyPI.

**The 404s are NOT the evidence, and this is the part to read carefully.** Four
candidate filenames under `https://wheels.vllm.ai/<sha>/` returned 404 by GET
and by HEAD, on the pin and on a current `main` sha `c1e438728c55`. That URL
scheme was never confirmed against a known-good case, and the host's own root
404s while `/nightly` returns 200. A 404 from an unconfirmed path therefore
proves nothing about whether a per-commit wheel exists. The load-bearing
evidence is the nightly index listing one commit.

**So the consequence is narrow.** Reproducing the pinned oracle needs a source
build, which is why it needs `nvcc`, or a deliberate advance of the pin to a
commit a wheel exists for. A wheel cannot substitute for either. **This does NOT
establish that vLLM never retains per-commit wheels.** Nobody measured that, and
the unconfirmed URL scheme is exactly why.

**The source build was then run, and it worked.** On 2026-08-18, inside an `rc`
lease on `dgx:gpu0`, that build produced our own wheel from the pin against a
staged CUDA toolkit, and the installed package imports and sees the GB10
([#1185](https://github.com/mudler/vllm.cpp/issues/1185),
[`oracle-wheel-in-lease.md`](oracle-wheel-in-lease.md)). So this section's
conclusion holds and its cost estimate does not: the source build is the route,
and the route is open.

## What this does NOT establish

Read this section before you quote any line above it. Every claim here is
narrower than the sentence a reader wants to write from it.

- **This is measured on `thor:gpu0` at capability (11,0) ONLY.** The GB10 is
  `sm_121a` and is UNMEASURED. Nothing here licenses a claim about the Spark.
  A probe is queued at lowest priority behind a human hold on `dgx:gpu0`. Until
  that probe returns, `dgx:gpu0` keeps the reading its own last probe gave it.
- **The pinned vLLM oracle is not in THIS staged tree.** Only `torch`, `triton`
  and `numpy` are, so nothing measured on `thor:gpu0` shows that the oracle
  runs. vLLM at `555967922` is a source build with compiled extensions, and a
  prebuilt wheel does not remove that requirement, which is measured rather than
  assumed. See `## The prebuilt-wheel route is closed for our pin`.
- **The `nvcc` clause this section used to carry is FALSIFIED.** It read "it
  needs `nvcc`, which the worker still lacks". On 2026-08-18 a lease on
  `dgx:gpu0` ran the source build against a staged CUDA toolkit and produced a
  434 MiB wheel that imports and reports `cuda True NVIDIA GB10`
  ([#1185](https://github.com/mudler/vllm.cpp/issues/1185),
  [`oracle-wheel-in-lease.md`](oracle-wheel-in-lease.md)). #1129's consequence
  for the oracle-dependent rows narrows again and does not close: those rows are
  UNBLOCKED for the build step and STILL BLOCKED for a model run, which nobody
  has measured.
- **The CUDA version skew is recorded as observed, not adjudicated.** The torch
  wheel is `+cu130` while the staged `ptxas` reports `release 12.8, V12.8.93`. It
  compiled and ran a correct kernel here. Nobody has read whether the skew
  changes a numerical result, and this row does not.
- **`numpy` WAS absent from the staged tree, and no longer is.** Job `fd5654c0`
  installed it into the same tree on 2026-08-17: `NUMPY_RC=0`, then
  `numpy 2.5.2 /workspace/oracle-probe/site/numpy/__init__.py`,
  `NUMPY_IMPORT_RC=0`. The five jobs above ran BEFORE that install, which is why
  each of their logs carries torch's `Failed to initialize NumPy: No module
  named 'numpy'` warning. A job that stages the tree today should not see that
  warning, and a job that still does is reading a different tree.

## Risks

The one checker these edits can break is `test_gpu_lock_one_truth` (#777), which
requires exactly one `**GPU mutex:**` bullet in `.agents/environment.md`. These
edits add no second mutex statement and do not touch that bullet.

The corrected sections say what one worker image did on one day. The worker image
can change again, in either direction, which is how this row came to exist. Each
section therefore carries its box, its date and its job IDs, so the next reader
can tell a stale reading from a wrong one.

The narrower risk is a reader who takes the `thor` result for a fleet result. The
`## What this does NOT establish` section above exists for exactly that reader,
and the same scope statement rides in every record this row edits.

## Gates

```sh
scripts/agent-preflight.sh
```

The full preflight is the row gate. This row adds no test, because it adds no
behavior: it records a measurement and corrects three documents. The `rc` jobs
below are the evidence, and they are not reproducible in CI, which has no fleet
device.

## Evidence

Six `rc run` jobs on `thor:gpu0`, 2026-08-17. The first five staged and ran the
runtime. The sixth added `numpy` and asked the wheel question, and it was
submitted separately as `claude/mudler-ubuntu-box/vllm-probe`.

Every job after the first ran a script staged on the NAS rather than fed on
stdin. That is a harness detail and not one of the four walls: `@triton.jit`
calls `inspect.getsourcelines`, which raises `OSError: could not get source code`
on a function read from stdin. The sha256 values are taken over the staged files
at `/mnt/nas_share/rc/oracle-probe/`, which is the same folder the worker reads
as `/workspace/oracle-probe/`.

| Job | Script | sha256 prefix | Result |
|---|---|---|---|
| `6f4bdb03-0162-4ed3-a922-f1a42da10fbd` | `probe.sh` | `9831e189910c` | torch imports, CUDA available, bf16 matmul runs, Triton refuses a stdin kernel |
| `9c0ebeac-8edb-4317-9112-2583bf85f38d` | `probe2.sh` | `0e9cd0f6d323` | wall 2: `fatal error: Python.h: No such file or directory` |
| `8beba132-c682-4b11-aa47-a682f2cce096` | `hdr.sh` | `fb02d1a1f45f` | `uid=0(root)`, `apt-get` installs `python3-dev`, headers staged, wall 3 appears as `PermissionError` on `ptxas-blackwell` |
| `f60d945f-f5ec-49c5-9452-c214a50a0043` | `hdr2.sh` | `37c9dcad4c64` | `TRITON_PTXAS_PATH` to a `/tmp` copy is insufficient, and the staged `ptxas` reports `release 12.8, V12.8.93` |
| `63c60a90-29b8-4115-88c5-d82be2126136` | `hdr3.sh` | `e11a540d23b8` | wall 4 removed, `TRITON_JIT_OK = 4096.0 PASS`, `PROBE5_RC=0` |
| `fd5654c0-d522-498c-8800-ca4df9a36944` | `vllmprobe.sh` | `58fa03543810` | `NUMPY_RC=0`, `numpy 2.5.2` imports from the staged tree, `NUMPY_IMPORT_RC=0`, `VLLM_DL_RC=0` on a 307,180,998-byte aarch64 vLLM wheel, `aarch64 linux-aarch64` |

The kernel each Triton job ran is `tritontest.py`, sha256 prefix `df9861c9f86d`.

One `rc run` job on `dgx:gpu0` is cited above for a different purpose:
`609c4944-594b-4617-967b-fb3d3d8c09f6` invoked `python3 -m pip install --target`
and then ended on `max_runtime exceeded (35m0s)`. It is evidence that `python3`
and `pip` exist on that worker. It is not evidence that its install completed.

The `wheels.vllm.ai` reads have no `rc` job, because they are host-side network
reads rather than device work.

## Stop conditions

- Stop if a correction needs a second `**GPU mutex:**` bullet. Return
  `NEEDS_DECISION`. Never widen `test_gpu_lock_one_truth` to pass.
- Stop if a correction would state the `thor` result as a `dgx` result. The
  scope limit is the point of this row.
- Stop if a correction needs an edit to an existing `.agents/issue-index.md`
  row. That file is append-only.

## Owed

- [#1146](https://github.com/mudler/vllm.cpp/issues/1146) stays open. Re-run the
  identical staged probe on `dgx:gpu0` at `sm_121a`.
- PAID for the build step. The pinned oracle itself now builds, installs and
  imports inside a lease on `dgx:gpu0`, against a staged `nvcc`
  ([#1185](https://github.com/mudler/vllm.cpp/issues/1185),
  [`oracle-wheel-in-lease.md`](oracle-wheel-in-lease.md)). A MODEL RUN is still
  owed, and #1185 owns it.
- Read whether the `+cu130` and `12.8` skew changes a numerical result.
- Confirm the `https://wheels.vllm.ai/<sha>/` URL scheme against a known-good
  case before anyone reads the four 404s as evidence of absence. Until then those
  404s carry no weight, and the nightly index is the only load-bearing reading.

Paid since this spec was written: `numpy` is staged (job `fd5654c0`), so the
former `## Owed` line asking for it is removed rather than left to read as debt.

## Now

The four walls and the working recipe are recorded here.
`.agents/environment.md`, `.agents/specs/mtp-k-gt-1.md` and
`.agents/specs/gpu-lease-methodology.md` no longer carry "cannot start Python" as
the live cause, and each now names the box and the date its reading came from.
The staged tree now holds `torch`, `triton` and `numpy`. It does not hold the
oracle, and a prebuilt wheel cannot put it there for our pin, so a source build
against `nvcc` is the route. That build was then run, inside a lease on
`dgx:gpu0`, and it produced a wheel that imports and sees the GB10
([#1185](https://github.com/mudler/vllm.cpp/issues/1185),
[`oracle-wheel-in-lease.md`](oracle-wheel-in-lease.md)). A model run stays
untested. The next step for this row is the `dgx:gpu0` runtime probe at
`sm_121a`, which the coordinator has queued.
