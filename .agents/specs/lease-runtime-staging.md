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
`ff28ada1-0cd3-4867-bf9b-f67050d0608b`, taken on 2026-08-17. The worker image
measured here on `thor:gpu0` the same day reports the opposite for two of those
five names, `python3` and `gcc`. It runs as `uid=0(root)`, carries
`/usr/bin/python3`, `/usr/bin/gcc` and `/usr/bin/apt-get`, and installs a package
as root (`APT_UPDATE_RC=0`, `APT_INSTALL_RC=0`). `pip`, `curl` and `git` were not
probed on `thor:gpu0`, so this row says nothing about them.

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

## What this does NOT establish

Read this section before you quote any line above it. Every claim here is
narrower than the sentence a reader wants to write from it.

- **This is measured on `thor:gpu0` at capability (11,0) ONLY.** The GB10 is
  `sm_121a` and is UNMEASURED. Nothing here licenses a claim about the Spark.
  A probe is queued at lowest priority behind a human hold on `dgx:gpu0`. Until
  that probe returns, `dgx:gpu0` keeps the reading its own last probe gave it.
- **The pinned vLLM oracle is NOT staged.** Only `torch` and `triton` are, so
  this does not show that the oracle runs. vLLM at `555967922` is a source build
  with compiled extensions, and it needs `nvcc`, which the worker still lacks.
  #1129's consequence for the oracle-dependent rows is therefore NARROWED and not
  closed.
- **The CUDA version skew is recorded as observed, not adjudicated.** The torch
  wheel is `+cu130` while the staged `ptxas` reports `release 12.8, V12.8.93`. It
  compiled and ran a correct kernel here. Nobody has read whether the skew
  changes a numerical result, and this row does not.
- **`numpy` is absent** from the staged tree. Torch warns about it on every
  import, and vLLM would require it.

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

Five `rc run` jobs on `thor:gpu0`, 2026-08-17. Every job after the first ran a
script staged on the NAS rather than fed on stdin. That is a harness detail and
not one of the four walls: `@triton.jit` calls `inspect.getsourcelines`, which
raises `OSError: could not get source code` on a function read from stdin. The
sha256 values are taken over the staged files at
`/mnt/nas_share/rc/oracle-probe/`, which is the same folder the worker reads as
`/workspace/oracle-probe/`.

| Job | Script | sha256 prefix | Result |
|---|---|---|---|
| `6f4bdb03-0162-4ed3-a922-f1a42da10fbd` | `probe.sh` | `9831e189910c` | torch imports, CUDA available, bf16 matmul runs, Triton refuses a stdin kernel |
| `9c0ebeac-8edb-4317-9112-2583bf85f38d` | `probe2.sh` | `0e9cd0f6d323` | wall 2: `fatal error: Python.h: No such file or directory` |
| `8beba132-c682-4b11-aa47-a682f2cce096` | `hdr.sh` | `fb02d1a1f45f` | `uid=0(root)`, `apt-get` installs `python3-dev`, headers staged, wall 3 appears as `PermissionError` on `ptxas-blackwell` |
| `f60d945f-f5ec-49c5-9452-c214a50a0043` | `hdr2.sh` | `37c9dcad4c64` | `TRITON_PTXAS_PATH` to a `/tmp` copy is insufficient, and the staged `ptxas` reports `release 12.8, V12.8.93` |
| `63c60a90-29b8-4115-88c5-d82be2126136` | `hdr3.sh` | `e11a540d23b8` | wall 4 removed, `TRITON_JIT_OK = 4096.0 PASS`, `PROBE5_RC=0` |

The kernel each Triton job ran is `tritontest.py`, sha256 prefix `df9861c9f86d`.

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
- Stage the pinned vLLM oracle itself, which is what #1129 actually blocks. It
  needs `nvcc` first.
- Read whether the `+cu130` and `12.8` skew changes a numerical result.
- Stage `numpy`, which vLLM requires and the current tree lacks.

## Now

The four walls and the working recipe are recorded here.
`.agents/environment.md`, `.agents/specs/mtp-k-gt-1.md` and
`.agents/specs/gpu-lease-methodology.md` no longer carry "cannot start Python" as
the live cause, and each now names the box and the date its reading came from.
The next step is the `dgx:gpu0` probe, which the coordinator has queued.
