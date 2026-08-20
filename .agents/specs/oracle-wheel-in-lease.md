# Building the pinned vLLM oracle inside a lease

Row: `ENV-ORACLE-WHEEL-IN-LEASE`.
Issue: [#1185](https://github.com/mudler/vllm.cpp/issues/1185).
Follows: [#1146](https://github.com/mudler/vllm.cpp/issues/1146),
[#1129](https://github.com/mudler/vllm.cpp/issues/1129).

## Scope

Record one measurement and correct the records it falsifies.

The measurement: the pinned vLLM oracle builds from source, installs, imports,
and sees the GPU inside an `rc` lease on `dgx:gpu0`. The build runs the `nvcc`
that a previous row staged on the NAS.

In scope:

- This spec, holding the result, the four staging walls, the non-claims, and the
  evidence.
- The correction to [`lease-runtime-staging.md`](lease-runtime-staging.md),
  whose `## What this does NOT establish` records the oracle as unstaged because
  the worker lacks `nvcc`, and whose body argues that a source build is out of
  reach.
- The correction to [`.agents/environment.md`](../environment.md),
  [`mtp-k-gt-1.md`](mtp-k-gt-1.md), and
  [`gpu-lease-methodology.md`](gpu-lease-methodology.md), which each carry the
  same `nvcc` reasoning.
- One appended row in [`.agents/issue-index.md`](../issue-index.md) for #1185.
- One comment on #1146 and one on #1129 recording the result.

Out of scope:

- Any product code. This row touches records and documents only.
- Any model run. See `## What this does NOT establish`.
- Any change to the parity-pin block in
  [`.agents/upstream-sync.md`](../upstream-sync.md). The version-string
  discrepancy is recorded as open, not resolved by editing the pin.
- Any checker change. This row changes no checker semantics, so it owes no
  red-before mutation under `## Changing the rules or a checker`.
- Editing an existing `.agents/issue-index.md` row. That file is append-only.

## Measured, on `dgx:gpu0`, 2026-08-18

Both jobs ran through `rc run`. No raw `ssh` was used, so the fleet reported the
box as held for the whole window.

The build job ran `bash /workspace/oracle-probe/buildvllm.sh`, staged script
sha256 prefix `15e140d41f44e7c2`.

- The script asserted the checkout against the pin **before** it compiled
  anything: `HEAD=5559679229bc961848b121ccdeaa8fa5d79bec98`, it printed
  `PIN CONFIRMED`, and it aborts when the two differ. A build that cannot say
  which commit it compiled is not an oracle.
- `nvcc` came from the toolkit staged by row `MODEL-NEMOTRON-H-ABI-A3-E2E`.
  `NVCC_RC=0`, CUDA `release 13.3, V13.3.73`.
- `WHEEL_RC=0` and `PERSIST_RC=0`.
- The artifact is
  `/workspace/oracle-vllm/vllm-0.1.dev1+g555967922.cu133-cp312-cp312-linux_aarch64.whl`,
  434 MiB, sha256
  `7c58b339741a288fbb313f4f5196c9c92a9e3b3c3ebe2ea970b0ff50bb9bcba4`.

The identity job ran `oracleenv.sh`, staged script sha256 prefix
`6119f5223f5d818c`. It asserted the identity from `cd /`, so it read the
installed package and not a source tree.

```
vllm.__file__    = /tmp/oracleenv/lib/python3.12/site-packages/vllm/__init__.py
vllm.__version__ = 0.1.dev1+g555967922
IDENTITY OK: installed wheel, commit g555967922
IDENTITY_RC=0
cuda True NVIDIA GB10
CUDA_RC=0
```

## What this does NOT establish

Read this section before you quote any line earlier in this spec. Every claim
here is narrower than the sentence a reader wants to write from it.

- **Running a model is UNTESTED.** Only the build, the install, the import, and
  `torch.cuda.is_available()` are measured. [`mtp-k-gt-1.md`](mtp-k-gt-1.md)
  records that the previous time an oracle reached this far it consumed the host
  in the step **after** `torch.compile` and rebooted the box, at
  `gpu_memory_utilization=0.75` and again at `0.30`, so the fraction is not the
  lever. Nothing measured here contradicts that. The next attempt varies
  `max_num_batched_tokens` and `cudagraph_capture_sizes` one at a time with a
  `MemAvailable` sampler running.
- **The version string differs from the recorded pin, and the discrepancy is
  OPEN.** This build reports `0.1.dev1+g555967922`, and
  [`.agents/upstream-sync.md`](../upstream-sync.md) records
  `vllm_runtime_version = 0.23.1rc1.dev1511+g555967922`. The commit segment
  matches, and the pin's binding rule is that the version carries a `+g<sha>`
  naming `vllm_commit`, which it does. The cause of the prefix difference is the
  shallow fetch: `setuptools_scm` cannot count the commits since the last tag and
  falls back to a default. A gate that compares the FULL string needs either a
  deeper fetch or an explicit pretend-version with the reason attached. Do not
  edit the parity-pin block to match this build.
- **The virtual environment is NOT staged.** That job was killed at its
  90-minute ceiling in the middle of the copy and left a partial tree, which was
  removed. Only the wheel is durable. A later job rebuilds the virtual
  environment from it.
- **This is `dgx:gpu0` on 2026-08-18 and nothing else.** The worker image can
  change again, in either direction, which is how #1146 came to exist.

## The four walls, in the order they appear

Every wall is a staging artifact rather than a property of CUDA, of vLLM, or of
the lease. The next person who stages a toolchain hits all four.

1. **Exec bits.** `cp -a` from the NAS preserves `file_mode=0664`, so `nvcc`
   exited **126**.
2. **Directory symlinks.** CIFS `nounix` cannot store a symlink, so `include`
   and `lib64`, which are normally links into `targets/<arch>/`, vanished. CMake
   then reported
   `Could NOT find CUDA (missing: CUDA_INCLUDE_DIRS CUDA_CUDART_LIBRARY) (found version "13.3")`.
   That message names the version and denies the toolkit in one line, so it
   reads like a broken install rather than a lost link.
3. **Library symlinks.** One level deeper, only the `libfoo.so.X.Y.Z` real files
   survived. Every `libfoo.so` and `libfoo.so.MAJOR` link was gone, and 32 links
   had to be rebuilt.
4. **A truncated staged package.** `markupsafe` exists in
   `/workspace/oracle-probe/site` as a dist-info with NO package files, because
   the original `pip --target` was killed at a 35-minute ceiling. Marlin codegen
   died on `ModuleNotFoundError: No module named 'markupsafe'`. A partial
   `pip --target` tree looks installed to a directory listing and is not.

**The `rc` worker container is REUSED between jobs.** A repair placed inside a
staging branch is skipped on the next run, which reports `nvcc already in place`
and then fails for the reason the branch was meant to remove. An environment
repair must be unconditional, and it must assert its postcondition.

## Consequence for the blocked rows

[#1129](https://github.com/mudler/vllm.cpp/issues/1129)'s index row states that
no vLLM leg of any row can run on `dgx.casa` by a lease-compliant path, and it
names the rows that blocks: [#1003](https://github.com/mudler/vllm.cpp/issues/1003)'s
re-takes, [#915](https://github.com/mudler/vllm.cpp/issues/915)'s withheld
cells, [#821](https://github.com/mudler/vllm.cpp/issues/821), and the MTP
adjudication plus vLLM leg owed by
[#81](https://github.com/mudler/vllm.cpp/issues/81).

Those rows are now **unblocked for the build step and still blocked for a model
run**. Write it exactly that way. None of them can take a measurement until a
model run is demonstrated inside the lease, and the recorded failure mode of the
step after `torch.compile` is a host reboot.

## Risks

The corrected sections say what one worker image did on one day. The image can
change again in either direction, so each section carries its box, its date, and
its script hashes.

The narrower risk is a reader who takes "the oracle builds" for "the oracle
runs". `## What this does NOT establish` exists for that reader, and the same
non-claim rides in every record this row edits.

A third risk is the version string. A reader who compares the full
`vllm.__version__` against the recorded `vllm_runtime_version` sees a mismatch
and can conclude that the wrong commit was built. The commit segment is the
binding part, and this spec says so beside the cause.

## Gates

```sh
scripts/agent-preflight.sh --fail-on-skip
```

The full preflight is the row gate. This row adds no test, because it adds no
behavior: it records a measurement and corrects four documents. The `rc` jobs are
the evidence, and they are not reproducible in CI, which has no fleet device.

## Evidence

| Job | Script | sha256 prefix | Result |
|---|---|---|---|
| build | `buildvllm.sh` | `15e140d41f44e7c2` | `PIN CONFIRMED` at `5559679229bc961848b121ccdeaa8fa5d79bec98`, `NVCC_RC=0` on CUDA `release 13.3, V13.3.73`, `WHEEL_RC=0`, `PERSIST_RC=0`, 434 MiB wheel |
| identity | `oracleenv.sh` | `6119f5223f5d818c` | `IDENTITY_RC=0`, `vllm.__version__ = 0.1.dev1+g555967922` read from `cd /`, `cuda True NVIDIA GB10`, `CUDA_RC=0` |

Wheel sha256:
`7c58b339741a288fbb313f4f5196c9c92a9e3b3c3ebe2ea970b0ff50bb9bcba4`.

The virtual-environment staging job has no green result to cite. It was killed
at its 90-minute ceiling and its partial tree was removed.

## Stop conditions

- Stop if a correction would state the build result as a model-run result.
  Return `NEEDS_DECISION`. The distinction is the point of this row.
- Stop if a correction needs an edit to the parity-pin block in
  `.agents/upstream-sync.md`. The version discrepancy is recorded, not resolved.
- Stop if a correction needs an edit to an existing `.agents/issue-index.md`
  row. That file is append-only.

## The durable wheel is NOT installable under the name it was staged with

Measured 2026-08-19 on `dgx:gpu0`, run `/workspace/a2q1-neartie/20260819T215514Z`
([#1416](https://github.com/mudler/vllm.cpp/issues/1416)). The artifact this spec
calls durable is staged as
`vllm-0.1.dev1+g555967922-FLASHINFER-ONLY-cp312-cp312-linux_aarch64.whl`, and
`pip` refuses it before opening the file:

```
ERROR: Invalid wheel filename (wrong number of parts):
'vllm-0.1.dev1+g555967922-FLASHINFER-ONLY-cp312-cp312-linux_aarch64'
```

PEP 427 allows five `-`-separated parts, six with a build tag. The
`FLASHINFER-ONLY` marker adds two more, so
`packaging.utils.parse_wheel_filename` raises `InvalidWheelFilename`. The bytes
are unaffected: the wheel's `METADATA` version is `0.1.dev1+g555967922`, exactly
what a conforming name carries.

**Copy to a conforming name; do not rename the staged artifact**, whose name is
what identifies it to a reader:

```sh
WHEEL_OK=/tmp/vllm-0.1.dev1+g555967922-cp312-cp312-linux_aarch64.whl
cp "$WHEEL" "$WHEEL_OK" && pip install "$WHEEL_OK"
```

**Two facts about WHERE this failure lands, because they are what made it
expensive.** `pip install -q torch==2.13.0` must run first and takes about
thirteen minutes, so `RC[pip wheel]=1` arrives long after the job looks healthy;
and a driver that does not stop there reports the failure only as
`ModuleNotFoundError: No module named 'vllm'` at the measurement step. That is an
infrastructure failure presenting as a verdict about the model, which is the
shape this repository has been caught by before. A lease driver that installs
this wheel therefore asserts the identity and **exits** on failure rather than
continuing.

Everything else in that job was green, so nothing else here is in doubt:
`RC[apt-get install]=0`, `nvcc` already present at `cuda_13.0.r13.0` with no
install needed, `RC[pip torch]=0`, and `cuda True NVIDIA GB10`.

## Owed

- [#1185](https://github.com/mudler/vllm.cpp/issues/1185) stays open.
  Demonstrate a model run inside the lease, varying `max_num_batched_tokens` and
  `cudagraph_capture_sizes` one at a time with a `MemAvailable` sampler.
- Rebuild the virtual environment from the durable wheel. Only the wheel
  survived the 90-minute ceiling.
- Resolve the version-string discrepancy against `.agents/upstream-sync.md`,
  either with a deeper fetch or with a recorded pretend-version and its reason.
- [#1146](https://github.com/mudler/vllm.cpp/issues/1146) stays open for its own
  reason: the staged `torch` and `triton` result is `thor:gpu0` at capability
  (11,0) only.

## Now

The pinned oracle builds, installs, imports, and sees the GB10 inside a lease on
`dgx:gpu0`. The durable artifact is the wheel on `/workspace/oracle-vllm`.
`lease-runtime-staging.md`, `.agents/environment.md`, `mtp-k-gt-1.md`, and
`gpu-lease-methodology.md` no longer carry the missing `nvcc` as the live cause,
and each now says that the oracle-dependent rows are unblocked for the build step
and still blocked for a model run. The next step is a model run, and the recorded
failure mode of the step after `torch.compile` is a host reboot.
