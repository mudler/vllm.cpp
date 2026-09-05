# Sync cycle `e126687a9a`, wave RUNHALF

Row: `UPSTREAM-SYNC-HEADPIN`.
Issue: [#2611](https://github.com/mudler/vllm.cpp/issues/2611).
Predecessor: [#2593](https://github.com/mudler/vllm.cpp/issues/2593), wave
HEADPIN, which landed [#2594](https://github.com/mudler/vllm.cpp/pull/2594) and
[`../sync/2026-09-02-e126687.md`](../sync/2026-09-02-e126687.md).
Row spec: [`upstream-sync-headpin.md`](upstream-sync-headpin.md), whose `## Owed`
names this work as its first item.

## Now

**Measured. `e126687a9a` builds from source and runs a model on `thor:gpu0`** —
level 2 of §2.1, on vLLM's default backend, eager and compiled.
`SRCBUILD_RC=0`, `EXT_PRESENT=True`, `RUN_RC=0`, `COMPILED_RC=0`. The
`qwen4_exp` stretch leg did not run (`Q4EXP_RC=1`,
[#2626](https://github.com/mudler/vllm.cpp/issues/2626)).
Report: [`../sync/2026-09-03-e126687-runhalf.md`](../sync/2026-09-03-e126687-runhalf.md).

The pin did **not** advance and nothing measured here is a reason to move it.
The active parity pin remains `5559679229bc961848b121ccdeaa8fa5d79bec98`.

**SUPERSEDED 2026-09-03: the pin HAS advanced**, to
`e126687a9a828d513c01a07cd69f025f27d63280`, by the developer ruling recorded in
[`upstream-pin-advance-e126687.md`](upstream-pin-advance-e126687.md) §1
([#2817](https://github.com/mudler/vllm.cpp/issues/2817)). The sentence above was
true of THIS wave and is kept because it was that wave's own stop condition. It
is no longer true of the tree.

## 1. Scope

**One question.** Does `e126687a9a828d513c01a07cd69f025f27d63280` demonstrably
**run a model** on this fleet? AGENTS.md §"When vLLM has no implementation"
requires an oracle to "demonstrably build and run the model" before it is
gateable, and adds that "constructing a config proves nothing". #2594 measured
install, build and import at this target and said in as many words that this is
not the run half: no weights were loaded, no forward pass ran, no token was
compared. That is the sole reason the candidate's `gateable` stays `no`.

**What makes it reachable now, and what still blocks it.** `thor:gpu0`'s leased
container exposes the device (`CUDA_AVAIL=True`, `NVIDIA Thor, 595.78`) where
`orin`'s does not, and `nvidia-cuda-nvcc 13.3.73` installs as an aarch64 wheel
inside the target's own `requirements/cuda.txt`
([`../sync/2026-09-02-e126687.md`](../sync/2026-09-02-e126687.md) §5.5, §5.7).
The blocker is that every #2594 job ran `VLLM_USE_PRECOMPILED=1`, which on
aarch64 yields an editable wheel and no `vllm._C` (`EXT_RC=1`). **A package
without its extension cannot run a model**, so this wave needs a source build,
and no job has ever run one at this target. Locating a compiler is not compiling
with it.

In scope:

1. One `rc` lease on `thor:gpu0`. Source-build vLLM at the target from a
   verified tree, install the wheel, and report `SRCBUILD_RC`, `EXT_PRESENT`
   and `RUN_RC` separately and literally.
2. Generate tokens greedily from a real checkpoint through vLLM's normal path,
   and record the model, prompts, sampling parameters, engine configuration and
   the output token ids.
3. Record the result, and record what gateability it establishes and what it
   does not.

Out of scope:

- **Advancing the pin.** The 290-entry PORT-NOW queue for
  `5559679229..e126687a9a` is unworked and is not this wave's.
- Working that queue, re-deriving its dispositions, or editing
  [`../porting-inventory.md`](../porting-inventory.md).
- Any product code. This wave touches records and documents only.
- Any throughput, latency or memory number. A run is a correctness precondition
  here, not a benchmark, and nothing measured in this wave may be quoted as a
  speed axis.

## 2. Design

### 2.1 Three levels of answer, and which one this wave targets

The question "does it run the model" has three levels on this fleet, and the
report says which one it reached rather than blurring them:

1. **`qwen4_exp` itself**, from a real checkpoint. **Not reachable.** Every
   published safetensors arm of Qwen3.8-Flash-Next exceeds the largest fleet
   box, and upstream's own `tests/models/registry.py` marks all three Qwen4Exp
   architectures `is_available_online=False` at this very revision. Nothing this
   wave can do changes that.
2. **Any model** generating tokens through vLLM's normal path at this revision.
   **This wave reads AGENTS.md's "builds and runs the model" permissively, as
   any model, and targets that. It does not claim the reading is settled.** A
   strict reading — that the model an oracle must run is the model the candidate
   is wanted for — is defensible and would not be satisfied by level 2. Under
   either reading the pin does not move, and §6 of the report records that
   `qwen4_exp` itself does not run on this device.
3. A forward pass producing logits without generation. Weaker, and recorded as
   such if it is all that is reached.

A fourth reading sits between 1 and 2 and this wave attempts it as a stretch:
the `qwen4_exp` graph executing on **random** weights, from the model's own
published `config.json` shrunk in **depth and count only** — width is left
alone, because halving `head_dim` breaks
`sum(mrope_section) == rotary_dim // 2` — under `load_format="dummy"`.
That is not a parity statement and not a token gate — the tokens carry no
information — but it is the strongest statement about this architecture the
fleet can currently support, and it is reported with that qualifier attached.

### 2.2 The build

`VLLM_USE_PRECOMPILED=0`, `VLLM_TARGET_DEVICE=cuda`, `MAX_JOBS=4`,
`NVCC_THREADS=1`, and `TORCH_CUDA_ARCH_LIST` read from the device rather than
written down. The shape follows the only recorded source build of upstream vLLM
inside a lease, `.agents/benchmark-record.md`'s DFlash2 oracle wheel on
`dgx:gpu0`: `pip wheel --no-deps --no-build-isolation -w dist .`, then install
the wheel into a venv. `MAX_JOBS=4` is AGENTS.md's limit; unconstrained
parallelism has OOM-rebooted a fleet box.

**`VLLM_FA_CMAKE_GPU_ARCHES` is deliberately NOT set.** Upstream's
`vllm-project/flash-attention` hard-codes `FA2_ARCHS "8.0+PTX"`, so the built
FlashAttention reaches this device only through a driver JIT of `compute_80`
PTX, which is the mode that failed on GB10 with
`cudaErrorUnsupportedPtxVersion` (`/workspace/oracle-vllm/README-WHEELS.md`).
Overriding the arch list might produce native SASS and might instead fail the
whole compile for a device nobody has built FA2 for. The question this wave owes
is whether a model runs, not which attention backend is fastest, so the build
takes the low-risk path and the run walks a backend ladder: the default first,
then `TRITON_ATTN`, then `FLASHINFER`, reporting which one answered.

### 2.3 Nothing the build needs is fetched from the network

The worker's `github.com` egress is not guaranteed and its absence reads as an
authentication failure rather than a network one
([`../environment.md`](../environment.md)); a `thor:gpu0` job failed `git fetch`
that way on 2026-09-02, the same day #2594's jobs cloned successfully. The
target tree, the `FetchContent` dependencies CMake would clone and the
checkpoint are therefore staged on `/workspace` from the developer box, and the
build is pointed at them with the `*_SRC_DIR` overrides upstream provides.

**This spec planned for FOUR such dependencies and the measurement found NINE.**
A CUDA build clones all nine at configure time, before any architecture gate can
skip one, and the fifth of them — `deepseek-ai/DeepGEMM` — is what job A died
on. The report's §3 carries the full table with revisions and override names.
Because the count was wrong once, the job also prints every `GIT_REPOSITORY`
the tree still declares beside every override it set, so a tenth names itself in
the log rather than costing another lease.

The tree is staged as a **git bundle**, not a tarball, for two reasons. A
release tarball cannot build vLLM, because `setuptools_scm` needs git. And the
bundle carries the tags, so `git describe` and the version string are derived in
the built environment rather than transcribed from the developer box, which is
the failure #2589 §7.3 measured when a locally-only tag rewrote a version
prefix.

### 2.4 The identity is asserted, not named

The job reads `HEAD` from the restored tree and **aborts** when it is not the
target, before it installs or compiles anything. `vllm.__version__` is then read
from `cd /`, so it is the installed package and not a source tree, and it must
carry `e126687a9`. A build that cannot say which commit it compiled is not an
oracle.

### 2.5 Instruments

This row's recurring failure is an instrument whose failure looks like a result,
and #2594's own `BUILDDEPS_RC=1` was one. Three rules follow, and the job
implements each:

- **Every rc is printed separately and literally**, immediately after its
  command and never after a pipe. A leg that could not run prints
  `SKIPPED_<reason>`, never a `0` and never a red that would read as a target
  defect.
- **The run is watchdogged on `MemAvailable`, and the floor is recorded.**
  `thor` is a unified-memory box on which `gpu_memory_utilization` reserves HOST
  RAM, and the recorded failure mode of this fleet is a host consumed in the
  step after `torch.compile` — a reboot on `dgx` at both `0.75` and `0.30`, so
  the fraction is not the lever. The engine therefore runs at `0.10` of 132 GB,
  which is far more than a 125M model needs and far below any fraction that has
  taken a box down, and the watchdog floor sits at 20,000 MB, which is outside
  that configuration's operating point. A guard whose threshold sits inside the
  guarded thing's operating point manufactures the finding it was meant to
  detect.
- **The first leg is `enforce_eager=True`**, which removes `torch.compile` and
  graph capture from the variables. Only after an eager leg has proved the
  engine runs does a compiled leg follow. Neither is a benchmark denominator and
  neither may be quoted as one.

### 2.6 The workload is the one already committed here

Prompts, dtype, sampling parameters and the per-prompt batch=1 regime are copied
from [`../../scripts/opt-oracle-capture.py`](../../scripts/opt-oracle-capture.py),
which captured `tests/parity/goldens/opt_greedy` at the **pin** on `dgx`. Laying
this wave's token ids beside that golden is **informative and not a gate**: the
revision differs by 1465 commits and the silicon differs, so a divergence is not
a defect and an agreement is not a parity result.

## 3. Risks

- **The source build may fail.** Nobody has built upstream vLLM on `thor` — the
  repository has no `TORCH_CUDA_ARCH_LIST` value for `sm_110` anywhere, and no
  recorded build duration or toolkit choice for one. A failure is a legitimate
  result and is reported as the **first** error rather than the last line.
- **The box may be lost.** `thor` reboots rather than OOM-kills, and it has gone
  `worker_lost` mid-build before. The wheel is persisted to `/workspace` the
  moment it exists, so a build that succeeded is not repaid by a run that dies.
- **`e126687a9a` may run a model and still not be pinnable.** It is, and §5
  says so: this wave measures a precondition, not a licence.

## 4. Gates

- `SRCBUILD_RC`, `EXT_PRESENT`, `IMPORT_RC` and `RUN_RC` reported separately,
  each read literally, from one `rc` lease on `thor:gpu0`.
- The run recipe recorded in full: model identity, prompts, sampling parameters,
  engine configuration, backend, and the output token ids.
- `git diff` is not a gate here. This wave's gate is the job's own output, and
  the report carries it verbatim.

## 5. Stop conditions

- **Do not advance the pin.** Not on any result in this wave.
- **One lease, on `thor:gpu0`.** Never a second device, never two at once.
- **Do not fake a run.** If the source build fails, report the failure precisely
  and stop. A build that did not produce a wheel makes every leg below it an
  absence, and the job prints them as absences.
- Touch [`../oracles/vllm.md`](../oracles/vllm.md) only if the run succeeds, and
  then only to record what this candidate established. The `oracle-pin` block
  keeps naming `5559679229`.

## Owed

Carried from [`upstream-sync-headpin.md`](upstream-sync-headpin.md) and not
discharged here:

- The PORT-NOW queue for `5559679229..e126687a9a`, 290 entries
  ([#2611](https://github.com/mudler/vllm.cpp/issues/2611)).
- Whether upstream `main` installs under the same repair (#2611).
- Nothing about the job scripts. They are committed at
  `.agents/scripts/runhalf-e126687-*`. An earlier draft owed their absence to
  `scripts/check-pr-size.py` not classifying `.agents/scripts/**`
  ([#2607](https://github.com/mudler/vllm.cpp/issues/2607)); that was false when
  written — the classifier has accepted those paths since before this wave's
  merge-base, and #2607 closed before the wave ran. See the report's §7 C7.

## Outcome

**What was measured.** Six `rc` jobs on `thor:gpu0`, one at a time. The source
build takes 94 minutes at `MAX_JOBS=4` for a single architecture and produces a
198 MB wheel with seven compiled extensions. The run reproduced on two separate
leases and matched `tests/parity/goldens/opt_greedy` on all 96 token ids, which
the report records as informative rather than as a gate.

**What was rejected, and why.**

- **Setting `VLLM_FA_CMAKE_GPU_ARCHES=110-real`** to get native FlashAttention
  SASS. Rejected before the build: a failed FA compile would have cost the whole
  90-minute build for a question the wave does not owe, and the backend ladder
  covered the risk at run time for the price of one extra minute. It turned out
  to be unnecessary — the stock `FA2_ARCHS "8.0+PTX"` build ran on Thor — so the
  cheap path also answered more than it promised.
- **A CPU-target build** (`VLLM_TARGET_DEVICE=cpu`), which compiles ~71 files
  instead of the CUDA set and would have reached a token in perhaps 20 minutes.
  Rejected because the oracle's whole purpose is GPU parity, and a CPU run would
  have answered a question nobody asked while leaving `gateable` exactly where it
  was.
- **Two hypotheses that measurement refuted**, both recorded rather than
  quietly dropped: that job A's clone failure was a 401 over HTTP/2 addressable
  with `git -c http.version=HTTP/1.1` (§3 of the report: it is a credential
  fall-through on a container with no egress, and staging is the fix), and that
  `qwen4_exp`'s cluster failure is specifically the 16-CTA arm (§6: the A/B did
  not move `num_rows`, so it is inconclusive).

**Why each default has its value.**

- `gpu_memory_utilization=0.10`. Thor is unified memory, so the fraction
  reserves host RAM, and this fleet's recorded failure is a host consumed after
  `torch.compile` at both 0.75 and 0.30. 0.10 of 126 GB is far more than a 125M
  model needs and far below any fraction that has taken a box down.
- Watchdog floor 20,000 MB. Chosen to sit **outside** the guarded
  configuration's operating point, which is the rule a tripped guard on this
  fleet already cost a measurement to learn. It never fired.
- `MAX_JOBS=4`. AGENTS.md's limit. Load held at ~8 on 14 CPUs and
  `MemAvailable` never fell below 104 GB, so the box was never near the edge.
- The tree staged as a git **bundle** rather than a tarball: `setuptools_scm`
  needs git, and the bundle carries the tags, so `GIT_DESCRIBE` and the version
  string are derived on the worker rather than transcribed. Both reproduced
  #2594's independently measured values.
