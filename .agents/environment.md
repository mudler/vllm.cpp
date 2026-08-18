# Environment & assets

This file is a factual registry of known development and gate environments. It
does not authorize connecting to a host, managing its services, using its
credentials, or assuming its paths exist. The untracked
`developer-preferences.md` selects which entries are available in the current
workspace and supplies local path/lock overrides. If no profile is selected,
use only the current local host and mark unavailable hardware gates `PENDING`.

## Reaching a GPU: claim a lease, never `ssh`

The shared GPUs are managed by
[resource-controller](https://github.com/mudler/resource-controller), whose
client is `rc`. **Claim a device with `rc run` or `rc hold` before any GPU work,
and never `ssh` to a GPU box to run work directly.** A bypass makes the fleet
report the box free while somebody is on it, which is the exact failure the
lease exists to remove. The procedure is in the `leasing-a-gpu` skill, which
this repository deliberately does not copy, because a copy goes stale without
saying so.

`AGENTS.md` §`Work on a GPU happens inside a lease` holds the rule, and its
condition is the DEVICE rather than the shell you are typing in. `dgx:gpu0`,
`thor:gpu0` and `orin:gpu0` are the fleet devices, so a lease is the required
path to each of them and it replaces the file mutex as the default. The three
names are listed in both files so that membership stays checkable when the
client is not at hand, and they are a lower bound rather than an upper one: a
device that `rc devices` reports is a fleet device even when this list has not
caught up. On a GPU that is not one of them, take
`${GPU_LOCK:-$HOME/gpu.lock}` as before.

`rc devices` lists the fleet when your shell has the client AND the controller
answers, so it reports your access and not the device's membership. It fails in
at least three ways that a reader must not collapse into one: `command not
found` means this shell lacks the client, and a timeout or a refused
authentication means the controller is not answering. `thor:gpu0` read `unknown
(no contact 1m0s)` on 2026-08-17, so lost contact is a live state. On a fleet
device every one of those answers means get the client or report the controller
down. None of them means take the file mutex over `ssh`, because that is the
collision below, in which two mutexes could not see each other.

**This REPLACED the `ssh <host>` plus `flock` mechanism that the profiles later
in this file still describe.** Read a historical recipe as evidence of what ran
at the time, not as an instruction for what to run now. The file mutex is still
real and still required, and it now lives INSIDE a lease rather than instead of
one.

**The bypass has already voided a measurement, so this is a measured cost and
not a rule for its own sake.** `.agents/specs/minimax-music3.md` §13.10 records
a whole speed axis retained as VOID on 2026-08-17: those runs went in by `ssh`
plus `docker run` serialised by the old mutex, while a concurrent session held
the SAME box through `rc`. The two sessions took different mutexes and neither
excluded the other, which is verbatim the #777 failure, and it is the likely
cause of a 3x swing in the samples. `.agents/benchmark-record.md` records the
other half of that row taking a real `rc hold` on `thor:gpu0` and reports the
window in which the fleet showed `thor:gpu0` FREE while it was in use.

The fleet, read from `rc devices` and `rc describe` on 2026-08-17:

| Device | Labels | `/workspace` on it |
|---|---|---|
| `dgx:gpu0` | `gpu_model=GB10`, `class=train`, `k8s=true`, driver 580.173.02, `cpus=20`, 128 GB | the house NAS |
| `thor:gpu0` | `gpu_model=NVIDIA-Thor`, `class=train`, `k8s=true`, driver 595.78, `cpus=14`, 132 GB | the house NAS, the SAME folder as `dgx` |
| `orin:gpu0` | `gpu_model=AGX-Orin`, `class=train`, `k8s=true`, `cpus=12`, 32 GB, and NO detected GPU labels because Jetson carries no `nvidia-smi` | LOCAL disk, invisible from `dgx` and `thor` |

**Select on `class` or `gpu_model`, never on `vram`.** `rc describe` reports
`vram=[N/A]M` and `vram_free=[N/A]M` on this fleet. That is a probe reporting
"unknown", not a value, so a selector such as `vram>=40G` matches nothing and
the job is rejected with `no_matching_device`. A device that carries no label
never matches, INCLUDING for `!=`, and `orin:gpu0` carries no detected GPU
labels at all. `class=train` and `gpu_model=GB10` do match. `rc run` has no
`--image` flag. Its flags are `--as`, `--cwd`, `-d`, `--explain`,
`--idle-timeout`, `--max-runtime`, `--no-wait`, `--priority`, `--select` and
`--timeout`.

### What the `dgx:gpu0` leased worker can and cannot do, measured 2026-08-17

Probed with one `rc run -d dgx:gpu0 --max-runtime 2m` job
(`ff28ada1-0cd3-4867-bf9b-f67050d0608b`). Verify this again before you plan work
around it, because the worker image can change under you. **It did change.** The
`thor:gpu0` worker measured later the same day carries `python3` and `gcc`, which
this list calls absent, so read this section as one box on one day. The `thor`
reading is in "A relocated CUDA runtime starts on `thor:gpu0`" further down.

- The command runs as user `rc` in a **k3s pod**, hostname `rc-worker-<id>`.
  `/.dockerenv` is absent and 8 `KUBERNETES_*` variables are set, so it is a pod
  rather than a docker container. The toolchain question is therefore a
  worker-image question, not a per-job one.
- Present: `bash`, `sh`, `ls`, **`nvidia-smi`** (which reports the GB10 by UUID),
  `flock`, and **`/workspace`**.
- **Absent: `gcc`, `cc`, `clang`, `nvcc`, `ninja`, `cmake`, `make`, `python3`,
  `python`, `pip`, `docker`, `sudo`, `git`, `ssh`, `curl`,
  `/usr/include/stdio.h`, and any `/usr/local/cuda*` toolkit.** This `dgx:gpu0`
  worker cannot compile, cannot start Python, and cannot install anything.
  **Do not carry that clause to another device.** On `thor:gpu0` the same day the
  worker ran as `uid=0(root)` with `/usr/bin/gcc`, `/usr/bin/python3` and a
  working `apt-get` ([#1146](https://github.com/mudler/vllm.cpp/issues/1146)).
- **The host filesystem is not visible.** `/home/mudler` does not exist inside
  the worker.
- `/workspace` is the house NAS, measured as `//192.168.68.102/Data 7.3T total,
  4.0T available, 46% used`, writable from the job, mounted on the dgx host at
  `/usr/local/nas_share/rc` (SMB, NodePort 31516, subfolder `rc/`). It is the
  SAME folder from `dgx` and from `thor`, and it is the one surface both ends
  can see. It is NOT shared with `orin`.

**The consequence, and it is now narrower than a blocker.** The pinned oracle
venv lives at `~/venvs/vllm-oracle-pin-555967922` on the dgx HOST, and a leased
worker cannot see it. The dgx host has carried no toolchain since the 2026-08-14
reimage, so host-side oracle work needs `sudo -n docker run` against
`vllmcpp-build:gb10` or `nvidia/cuda:13.0.1-devel-ubuntu24.04`, reached over
`ssh`, which is the bypass. **The sentence this paragraph used to carry, "no
vLLM leg of any row can currently run on `dgx.casa` by a lease-compliant path",
is FALSIFIED for the BUILD step and still holds for a MODEL RUN.** On 2026-08-18
two `rc run` jobs built the pin from source inside a lease, installed the wheel,
imported it, and reported `cuda True NVIDIA GB10`
([#1185](https://github.com/mudler/vllm.cpp/issues/1185), and "The pinned oracle
builds inside a lease on `dgx:gpu0`" further down). Nobody has run a model that
way, so no oracle-side MEASUREMENT is unblocked yet. Read the old reason
carefully before you quote it, because it was never the worker's missing
toolchain. "The lease carries bytes, and the exec bit is a mount option" below
measures staged content starting under the dynamic loader and after a copy to
`/tmp`. That is why recent GPU work reached for `ssh`, and the bypass is a
symptom of this gap rather than a discipline problem. Do not design the
migration here. `ENV-LEASE-RUNTIME-STAGING` owns the runtime staging, and
[`lease-runtime-staging.md`](specs/lease-runtime-staging.md) holds its working
recipe for `torch` and `triton`. `ENV-ORACLE-WHEEL-IN-LEASE` owns the oracle
build, in [`oracle-wheel-in-lease.md`](specs/oracle-wheel-in-lease.md).

**This confirms and extends a finding that already landed, rather than making a
new one.** `.agents/specs/minimax-music3.md` §13.10 probed `thor`'s worker on
2026-08-17 and found the same absence (`no gcc / g++ / cmake / ninja / nvcc /
make`), reported that the `$HOME` build tree is not mounted inside the worker,
and named what a valid re-measurement needs: either a worker image carrying the
CUDA devel toolchain, or the build placed on `/workspace` by something that
already has one. This row measured `dgx`'s worker and adds the part that turns
an open gap into a blocker for the parity gates, which is that the ORACLE VENV
is also unreachable from a lease.

### The lease carries bytes, and the exec bit is a mount option, measured 2026-08-17

Probed with two `rc run -d dgx:gpu0 --max-runtime 3m` jobs,
`1cb56f84-62bf-4c90-b138-9bd4c3b0617a` and
`c692d5a0-ec3d-4498-86e4-e86a2864e91a`. Verify this again before you plan work
around it, because the worker image and the mount options can change under you.

`/workspace` in the worker and `/mnt/nas_share/rc/` on a local host are the same
folder. A file written from the worker appeared locally under that path, and a
file staged locally was read by the worker. The worker's `df` reported
`//192.168.68.102/Data`, 7.3T total, 4.0T available, 46% used.

**Direct execution off `/workspace` is refused, and the mount causes it rather
than a `noexec` flag.** The worker mounts the share with `file_mode=0664`,
`dir_mode=0775`, `nounix`, `forceuid` and `forcegid`, and the option list holds
no `noexec`. The same bytes read `-rwxr-xr-x` on the local host, which mounts
the share with `file_mode=0755`, and `-rw-rw-r--` in the worker, so the exec bit
is a presentation of each mount and not stored state. In the worker, `chmod +x`
failed with `Operation not permitted`, and a staged shell script and a copied
ELF binary each failed to start with `Permission denied` and exit code 126.

**Three routes ran staged content anyway, and each measured green.** Record the
distinction, because a missing exec bit reads like a wall and is not one.

| Route | Measured |
|---|---|
| `sh /workspace/staged.sh` | printed the script's output, exit 0 |
| `/lib/ld-linux-aarch64.so.1 /workspace/echo_copy` | ran the ELF, exit 0 |
| `cp` to `/tmp`, then `chmod +x`, then run | ran the script and the ELF, exit 0 |

`/tmp`, `/var/tmp` and `$HOME`, which is `/home/rc`, are writable, take a real
exec bit, and sit on a 3.6T overlay with 2.5T available. `/dev/shm` is 64M. The
job's working directory `/` is not writable.

**So the lease carries bytes, and bytes are enough to run.** A runtime staged on
`/workspace` can start under the dynamic loader, or after a copy to `/tmp`. What
this `dgx:gpu0` worker cannot do is produce or fetch that runtime, because it has
no `curl`, `wget`, `git`, `gcc`, `nvcc`, `cmake` or `python3`. Present and useful
for staging: `cp`, `cat`, `tar`, `chmod`, `perl`, `flock` and `nvidia-smi`. **The
`thor:gpu0` worker does produce one**, because it is root and carries `apt-get`
and `gcc`. That is the section below.

**This narrows [#1129](https://github.com/mudler/vllm.cpp/issues/1129) and does
not close it.** The HOST venv at `~/venvs/vllm-oracle-pin-555967922` stays
unreachable from a lease, and only a host-side actor reached over `ssh` can
place a copy of it on the NAS. That route is no longer the only one, because a
lease BUILT the pin on 2026-08-18 rather than copying it
([#1185](https://github.com/mudler/vllm.cpp/issues/1185)). **Whether a relocated CUDA runtime then starts is no longer
UNMEASURED. It starts, on `thor:gpu0`.** The section below has the reading. A
CUDA virtual environment still holds absolute paths in its shebangs and its
`RECORD` files, so a `pip install --target` tree is the shape that was measured
and a copied venv is not.

**Three fleet-side changes would each remove the staging problem, and none of
them is ours to make.** Whoever owns the fleet picks one. **A fourth route was
then measured, and it needs nobody's permission:** the `thor:gpu0` worker runs
as root with a working `apt-get`, so a job provisions its own container.

1. The worker image gains a toolchain and a Python interpreter.
2. `rc run` gains an `--image` flag, so a job selects an image that has them.
3. `/workspace` is mounted so that a file there can carry an exec bit. This one
   removes the copy step only, because the two routes above already execute.

### A relocated CUDA runtime starts on `thor:gpu0`, measured 2026-08-17

Probed with six `rc run` jobs on `thor:gpu0`: `6f4bdb03`, `9c0ebeac`, `8beba132`,
`f60d945f`, `63c60a90` and `fd5654c0`. A `torch`, `triton` and `numpy` tree
staged on `/workspace` imports, initializes CUDA, runs a bf16 matmul, and
compiles and executes a Triton kernel. The job IDs in full, the staged-script
sha256 values, the four walls and the working recipe are in
[`lease-runtime-staging.md`](specs/lease-runtime-staging.md)
([#1146](https://github.com/mudler/vllm.cpp/issues/1146)).

```
torch.__version__= 2.13.0+cu130      cuda available = True
device 0         = NVIDIA Thor       capability     = (11, 0)
triton.__version__ = 3.7.1           TRITON_JIT_OK  = 4096.0 PASS
```

The recipe, once per worker container:

```sh
apt-get update -qq && apt-get install -y -qq python3-dev
mkdir -p /tmp/tp && cp -a /workspace/oracle-probe/site/triton /tmp/tp/
chmod -R +x /tmp/tp/triton/backends/nvidia/bin/

export PYTHONPATH=/tmp/tp:/workspace/oracle-probe/site
export CPATH=/workspace/oracle-probe/pyhdr/python3.12:${CPATH:-}
```

**Read the scope before you quote it.** This is `thor:gpu0` at capability (11,0)
and nothing else. The GB10 is `sm_121a`, and this tree was never staged there,
so nothing here licenses a claim about the Spark. Only `torch`, `triton` and
`numpy` are in this tree, so the pinned vLLM oracle is not shown to run by it.
The clause this paragraph used to carry, that the oracle "needs `nvcc`, which
the worker lacks", is FALSIFIED: a `dgx:gpu0` lease built the pin against a
staged CUDA toolkit on 2026-08-18, in the section after the next one. The torch
wheel is `+cu130` while the staged `ptxas` reports `release 12.8, V12.8.93`, and
that skew is recorded as observed rather than adjudicated.

**A prebuilt wheel does not remove the `nvcc` requirement, and that is measured.**
An aarch64 vLLM wheel exists in general: `pip download --no-deps vllm` on the
worker fetched `vllm-0.27.1-cp38-abi3-manylinux_2_28_aarch64.whl`, 307,180,998
bytes. Our pin is not reachable that way, because
`https://wheels.vllm.ai/nightly/vllm/` lists wheels for exactly ONE commit and is
a moving pointer rather than an archive, and because the pin is a development
version that is not on PyPI. Four 404s under a per-commit URL scheme were also
seen, and they prove nothing, because that scheme was never confirmed against a
known-good case. So reproducing the pinned oracle needs a source build or a
deliberate pin advance. Nobody established that vLLM never retains per-commit
wheels. That source build was then run inside a lease and produced our own
wheel, so the route this paragraph names is open.

### The pinned oracle builds inside a lease on `dgx:gpu0`, measured 2026-08-18

Two `rc run` jobs on `dgx:gpu0`. The pinned vLLM oracle builds from source,
installs, imports, and sees the GPU inside a lease. No raw `ssh` was used, so
the fleet reported the box as held for the whole window. The job details, the
staged-script hashes, the four staging walls and the non-claims are in
[`oracle-wheel-in-lease.md`](specs/oracle-wheel-in-lease.md)
([#1185](https://github.com/mudler/vllm.cpp/issues/1185)).

```
HEAD             = 5559679229bc961848b121ccdeaa8fa5d79bec98   PIN CONFIRMED
nvcc             = release 13.3, V13.3.73    NVCC_RC=0
wheel            = vllm-0.1.dev1+g555967922.cu133-cp312-cp312-linux_aarch64.whl
vllm.__version__ = 0.1.dev1+g555967922       IDENTITY_RC=0
cuda True NVIDIA GB10                        CUDA_RC=0
```

The `nvcc` came from the CUDA toolkit that row `MODEL-NEMOTRON-H-ABI-A3-E2E`
staged on the NAS. The build script asserts `HEAD` against the pin before it
compiles anything, and it aborts when the two differ.

**Read the scope before you quote it, because three things are UNMEASURED.**
Running a model is untested: only the build, the install, the import and
`torch.cuda.is_available()` were measured, and the recorded failure mode of the
step after `torch.compile` on this host is a reboot of the box
(`.agents/specs/mtp-k-gt-1.md`). The wheel reports `0.1.dev1+g555967922` while
`.agents/upstream-sync.md` records
`vllm_runtime_version = 0.23.1rc1.dev1511+g555967922`, an OPEN discrepancy whose
cause is the shallow fetch that stops `setuptools_scm` counting the commits
since the last tag. The virtual environment is NOT staged, because that job was
killed at its 90-minute ceiling and its partial tree was removed, so only the
WHEEL is durable.

**Two staging traps are worth carrying forward.** A `cp -a` from the NAS
preserves `file_mode=0664`, so `nvcc` exits 126, and CIFS `nounix` stores no
symlink, so a copied CUDA toolkit loses `include`, `lib64` and 32 library links
and CMake then reports
`Could NOT find CUDA (missing: CUDA_INCLUDE_DIRS CUDA_CUDART_LIBRARY) (found version "13.3")`,
which names the version and denies the toolkit in one line. **The `rc` worker
container is REUSED between jobs**, so a repair inside a staging branch is
skipped on the next run and reports `nvcc already in place`. Write an
environment repair unconditionally, and assert its postcondition.

### The `flock` orphan hazard that motivated the replacement

The harness family in this repository puts the `flock` handle on a **subshell**,
not on the `timeout` or wrapper process a reader would check. Kill the wrapper
and an ORPHAN survives holding the mutex, with its output pipe severed, and it
looks perfectly idle to every instrument a reader reaches for.

Measured 2026-08-17 (`.agents/specs/mtp-k-gt-1.md`, "What held the mutex"):
`nvidia-smi --query-compute-apps` was EMPTY and loadavg stayed near 1.1, which
reads as a finished holder. The holder was PID 333128, `bash -s 8000`, `PPid: 1`,
holding `fd 3` on the lock file, with `fd 1` and `fd 2` still pointing at the
pipe its dead `tee` had been reading. It was not idle. It held a live container
and was inside a readiness poll, and it blocked its own owner's restart for about
50 minutes as well as the queued gate. Read the whole process chain and
`/proc/<pid>/fd` before you call a lock stale, and never kill an unowned PID.

## Registering your own environment

The profiles below are per-developer facts, not requirements: nothing here is
usable unless your own setup provides it. To make your hardware a gate
environment:

1. Copy `.env.example` to the untracked `.env` at the repository root and fill
   in what your setup has (reference checkouts, oracle, gate host, GPU lock,
   device arch and toolchain). Empty means unavailable, and the gates that
   need it stay `PENDING` for you.
2. Copy `developer-preferences.example.md` to the untracked
   `developer-preferences.md` for the policy choices (Git integration, which
   remote hosts you may use, contention policy).
3. Set `CHECKPOINT_ROOT` in your `.env` if you have shared or network storage,
   and download model weights there rather than onto a box's system disk. A
   30B bf16 checkpoint is ~60 GB; a build tree is ~169 GiB on its own, and a
   full disk surfaces as unrelated test failures rather than an obvious disk
   error. Fetching once to shared storage means every host, worktree and agent
   reuses it instead of each pulling its own copy. It states an INTENT and
   nothing more: no code in the tree reads `CHECKPOINT_ROOT`, so it neither
   redirects a download nor resolves a bare directory name — you place the
   weights under it and pass the full path onward. A setup with no shared mount
   leaves it empty and uses whatever the tool defaults to (usually the Hugging
   Face cache under `$HOME`).

   Two rules travel with it. Pin an explicit revision when you fetch:
   publishers re-quantize in place under an unchanged repo name, so a bare
   branch name is not reproducible and a checkpoint you gated against can
   change under you. And setting the variable authorizes nothing on its own —
   a large asset download still needs authority for the task.

4. Add a profile entry to this file, in the same shape as the entries below:
   hardware, arch, toolkit versions, oracle availability, and the box's
   quirks. A PR for it is welcome, so the shared record says where each gate
   can run. New accelerator classes (an AMD/ROCm box, an Intel GPU) register
   the same way and become the gate environment for their backend rows.

- **Rich local development/GPU profile (re-verified 2026-07-25):** NVIDIA GeForce RTX
  5070 Ti, 16 GiB, compute capability 12.0 (`sm_120`), driver 595.71.05. The
  cached `Qwen/Qwen3.5-4B` snapshot is the only model large enough for the
  local direct-load performance diagnostic; it cannot run the 27B/35B gates.
  The CUDA Nix shell must put `/run/opengl-driver/lib` before toolkit stubs.
  `flake.nix` now does so directly and no longer emits a malformed literal
  `LD_LIBRARY_PATH` expansion. A clean `nix develop .#cuda` reports
  `torch.cuda.is_available() == True` and Triton
  `GPUTarget(backend='cuda', arch=120, warp_size=32)` without a manual
  override. Build the current diagnostic with CMake CUDA arch `120a`, Triton
  vendored target `sm_120`, CUTLASS and FlashAttention-2 enabled.
- **CPU development path:** CPU reference backend + engine logic + CI
  development.
- **Ettore DGX release-gate profile**: device `dgx:gpu0`, host `dgx.casa` (claim
  it with `rc`, and read "Reaching a GPU" earlier in this file before you use the
  `ssh` recipes recorded here). DGX Spark, GB10 (Blackwell, **sm_121**),
  ~119 GB unified memory, 20 cores, CUDA toolkit 13.0.88 (nvcc); compute
  capability 12.1 → sm_121. Unified memory: both gate models fit
  in bf16; the machine is memory-bandwidth-bound (~273 GB/s class) — decode
  parity is a bandwidth/launch-overhead game, hence CUDA graphs + fused
  kernels in T0. Give each active claim its own `~/work/<claim>/` directory;
  never share a build tree between agents.
  - Non-interactive SSH does not put nvcc on PATH — prepend
    `export PATH=/usr/local/cuda/bin:$PATH` in remote build commands.
  - **The NAS mounts at `/usr/local/nas_share`, and `/mnt/nas_share` is GONE
    (re-verified 2026-08-16).** `.env` sets
    `CHECKPOINT_ROOT=/usr/local/nas_share/checkpoints`, where 18 checkpoint
    directories resolve, `nemotron-3.5-lightning-30b-nvfp4` and
    `nemotron-3.5-lightning-30b-gguf` among them. **Do not restore the old path
    as a convenience symlink.** `/mnt` is on the EPHEMERAL root overlay of this
    immutable Kairos OS, so anything created there is gone after the next
    reboot; `/usr/local` is `COS_PERSISTENT` and survives. That is the same
    property that made an earlier `/oem` `rootfs`-stage change cost a boot (see
    [[kairos-oem-rw-paths-change-cost-a-boot]]). Measured 2026-08-16, after the
    box returned from an 8 h 19 min outage: the mount itself came back because
    the `/oem` boot-stage unit worked and `findmnt /usr/local/nas_share` was
    clean, and `/mnt/nas_share` did not come back. Every path built on `/mnt`
    broke while `.env` still declared it, which blocks a checkpoint-loading gate
    silently — a gate that reads a path `.env` does not declare is not the gate
    its spec names. Check `findmnt /usr/local/nas_share` before you conclude
    that a checkpoint is missing (#1073).
  - **MANDATORY gate-build flags on this box (re-proven 2026-07-29).** A model
    gate configured WITHOUT `-DVLLM_CPP_CUTLASS_DIR=$HOME/cutlass-4.5.0` and
    `-DVLLM_CPP_TRITON=ON` is NOT the production stack: cutlass-off silently
    disables the sm120a NVFP4 fp4×fp4 GEMM *and* FlashAttention-2, triton-off
    swaps the vendored Triton-AOT GDN kernels for the hand kernels. On the 27B
    that flips the documented tok6 near-tie and turns the SACRED
    `test_qwen27_paged_engine` red (234/235 or 233/235 vs 235/235) with the
    source untouched — measured three-arm from ONE tree at main `d4492c03`. Hard-
    verify the configure log prints `CUTLASS found … enabling sm120a NVFP4
    cutlass GEMM`, `FlashAttention-2 prefill/decode: ENABLED`, and the
    `Triton AOT: … <- vendored … sm_121a` lines before trusting any gate or A/B.
    Never read the arch from `CMakeCache.txt` (`CMAKE_CUDA_ARCHITECTURES` there
    legitimately reads `75`, the `enable_language(CUDA)` probe default shadowed
    in `CMakeLists.txt`); read `VLLM_CPP_CUDA_ARCHITECTURES`, `flags.make`, or
    `cuobjdump -lelf`. This defect has voided work three times (2026-07-16 ×2,
    2026-07-28); the 27B gate now refuses to run without both flags.
  - Oracle venv: **PIN ADVANCED 2026-07-26** (see
    [specs/pin-advance.md](specs/pin-advance.md)). `~/venvs/vllm-oracle` is now a
    canonical symlink to the from-source **`~/venvs/vllm-oracle-next`** — the new
    active stack is **vLLM 0.26.0.dev0+g5559679** (source `55596792`, built for
    sm_121a: the exact commit has NO aarch64 wheel, so the oracle is a ~1.3 h
    from-source build not a pip install), **Transformers 5.14.1, Torch
    2.13.0+cu130, FlashInfer 0.6.15.post1, CUTLASS DSL 4.6.0, Triton 3.7.1,
    torchvision 0.28.0**. This advance unblocks DFlash (vllm#40898 mixed-attn fix,
    under `VLLM_USE_V2_MODEL_RUNNER=1`), Gemma-4 (`transformers.models.gemma4`),
    and OLMo-3 (nested rope). It was validated by the W0–W4 pin-advance re-gate:
    **zero real golden drift** (27B-W4A4 + 32B-NVFP4A16 bit-identical, 35B/Coder
    byte-stable — the W0-W2 "27B drift" was a capture-config near-tie, not the
    oracle), full `ctest` 296/299 GREEN on GB10 (the 3 fails pre-exist on main,
    unrelated). **ROLLBACK (immediately restorable):** `ln -sfn
    ~/venvs/vllm-oracle-v0.25.0-stage ~/venvs/vllm-oracle` restores the prior pip
    vLLM 0.25.0 stack (FlashInfer 0.6.13, Torch 2.11.0+cu130, CUTLASS DSL 4.5.2,
    Transformers 5.13.1, Ninja 1.13.0; install/serving SHA-256 `ab786eee…c297` /
    `536385d8…f506`, vLLM/Ninja `ec6d76ff…96c` / `abf71487…10b`, freeze
    `cf1636cc…fa5f`); the v0.24.0 dir remains at
    `~/venvs/vllm-oracle-v0.24.0-retired`. The §2D mechanical re-sync of
    upstream-changed mirrored files (rmsnorm-fusion #46998, ReplaySSM #48018,
    MoeWNA16 #44120, olmo3.py, …) is a DEFERRED follow-on (gate-correctness does
    not depend on it — goldens are bit-identical).
  - The only dependency-check exception is NVIDIA's
    `nvidia-cusparselt-cu13==0.8.0` wheel: PyPI served the aarch64 wheel
    (`sha256:400c6ed1…77c`), its library is an AArch64 ELF and direct
    `ctypes.CDLL`/Torch imports pass, but its internal WHEEL tag is
    `manylinux2014_sbsa`, so `pip check` reports it unsupported. This is a
    recorded vendor-tag defect, not silently treated as a green `pip check`.
  - Lock-held production-graph validation on the exact 27B snapshot passed both
    offline generation (16 input IDs, one output ID) and the actual text-only
    server: `/health` 200 plus `/v1/completions` 200 with exact 1+1 usage and
    `finish_reason=length`. Server log/response SHA-256 are
    `f56be69a…3787` / `82307db4…8e1` under
    `~/work/vllm-oracle-v0.25.0-stage-validation/2026-07-12-server-smoke`.
    The smoke rate is non-binding. Its first offline inference emitted one
    causal-conv Triton JIT warning, which remains a warmup/trace audit item.
    Online-gate manifests hash pandas package/distribution files plus Ninja and
    reject missing/drifted dependencies before the GPU lock; profiler launches
    prepend the venv `bin` to spawned EngineCore `PATH`.
  - **Run the CUDA `ctest` suite with `-j 1`.** GB10 memory is UNIFIED, so a
    `gpu_memory_utilization` reservation is HOST RAM: concurrent model gates
    stack into the same ~119 GB and the kernel starts killing. Measured
    2026-08-09 on a default-ON Triton build, `ctest -j 4` **OOM-rebooted this
    box** (`NVRM ... Out of memory [NV_ERR_NO_MEMORY]`), which is why the
    parallel-flake advice in the Apple/Metal profile below does not transfer
    here. Serialising also means every other probe queues behind the suite, so
    run attribution arms BEFORE a full suite, never during one.
  - **GPU mutex:** this runs INSIDE an `rc` lease, never instead of one. The
    lease decides who gets the box. The mutex serialises the work of whoever
    holds it. Every CUDA test/model/serve/benchmark/profile holds the
    `${GPU_LOCK}` file mutex — **`$HOME/gpu.lock`**, which is what `.env.example`
    ships and what every script here falls back to via
    `${GPU_LOCK:-$HOME/gpu.lock}` — for the whole job or multi-arm series WHEN
    other agents may run GPU work concurrently (sole owner verified idle via
    `nvidia-smi` may skip). Mechanism: run GPU work as
    `flock ${GPU_LOCK:-$HOME/gpu.lock} -c '<command>'`, or take the lock once
    around an entire benchmark series so arms are never interleaved; waiting on
    the lock is normal, stealing it is not. Compilation, source inspection and
    file transfer do not need the lock. Never kill an unowned PID.
    **Check your `.env` before you measure anything:** a `GPU_LOCK` naming any
    other path takes a mutex nobody else holds, and `flock` succeeds on it, so
    the run is unserialised and only looks like someone else misbehaving. That
    cost a whole Marlin series (#777); an existing `.env` predating that fix
    must be repaired by hand.
  - Disk cleanup 2026-07-10 reclaimed ~368 GB from unrelated cached model sets,
    April-era autoresearch logits/F16-GGUF cache artifacts, the vLLM compile
    cache and stale rebuildable CUDA build trees. Active latency/PR workspaces,
    gate checkpoints, APEX GGUF evidence and sources were preserved; the volume
    had 359 GB free afterward. Maintain at least 200 GB headroom before adding
    competitor images.
- **Ettore Jetson Thor profile (sm_110 CUDA runtime gate)**: device `thor:gpu0`,
  host `192.168.68.23` (claim it with `rc` first)
  — NVIDIA Jetson Thor (Blackwell, **sm_110**), aarch64, 14 CPU cores, ~122 GB
  UNIFIED memory. `nvidia-smi --query-gpu=compute_cap` returns **11.0**. Host of
  the first non-GB10 runtime proof (`CLAIM-CUDA-SM110-RUNTIME`, 2026-07-27).
  - **REIMAGED, re-verified 2026-08-11.** The box is now hostname
    **`kairos-4db2`** (Ubuntu 24.04 under Kairos), driver **595.78**, and there is
    **NO host CUDA toolkit, no cmake, no nvcc, no huggingface_hub** — the JetPack
    R38 / `/usr/local/cuda-13.0` profile described here before is GONE. **The GPU
    is usable only from inside a container** (developer statement, confirmed on
    box). There is also no `local-ai-worker` container and no `~/gpu.lock`; the
    worker-restore discipline below does not apply in this state.
  - **Working container recipe** (each element was required; all three failed
    first):
    - `docker` needs `sudo` (the user is in group `admin`, not `docker`).
    - Use **`--runtime=nvidia`**, NOT `--gpus all`: the hook refuses the latter
      outright ("invoking the NVIDIA Container Runtime Hook directly ... is not
      supported").
    - Add **`-e NVIDIA_DISABLE_REQUIRE=1`**. The image's `NVIDIA_REQUIRE_CUDA`
      enumerates BOUNDED driver ranges topping out at `driver<576`, so driver
      595.78 — which is NEWER and forward-compatible — reads as unsupported.
    - Image already present: `nvidia/cuda:13.0.1-devel-ubuntu24.04` (nvcc
      **13.0.88**). Install `cmake ninja-build` inside; build
      `-DVLLM_CPP_CUDA_ARCHITECTURES=110 -DVLLM_CPP_CUDA=ON -DVLLM_CPP_TRITON=OFF`,
      no cutlass (every sm_110 fast-path cell resolves EMPTY). **VERIFIED
      2026-08-11: configure + build of `vllm-cli` both exit 0.**
    - `nsys` from `nsight-systems-cli` in that image is **2024.2.3 and cannot
      trace CUDA here** ("does not contain CUDA trace data"). Do not plan a
      graph/kernel-count measurement on Thor without first installing a newer one.
  - **★ THIS BOX REBOOTS INSTEAD OF OOM-KILLING — size every load for it.**
    `vm.overcommit_memory=1` ("always overcommit") with **zero swap**: the kernel
    grants memory it cannot back, and touching those pages takes the WHOLE MACHINE
    down. Signature: container `exit=255`, `OOMKilled=false`, NO `dmesg`/journal OOM
    line, host reboot (`uptime` resets). Observed **three times on 2026-08-11**
    loading a 27B: bf16 (52 GB target + 8.2 GB draft) twice, and NVFP4 (25 GB +
    8.2 GB) once, the last of which left the box unreachable pending a power cycle.
    A 27B target alone loads and runs fine; it is target+draft that crosses the
    line, consistent with the documented transient double-hold in the load path
    (`114→67 GiB`, see [[gb10-nvfp4-load-recipe-context-first-shard-release]]).
    **Do not run a >25 GB model plus a second checkpoint here.** Measured OK on
    this box: Qwen3-4B bf16 spec-off warm ~24.6 tok/s; Qwen3.6-27B bf16 spec-off
    warm 4.42 tok/s (portable kernels, no fast paths — absolute numbers are NOT
    comparable to GB10).
  - `k3s` runs here and is `enabled`; `sudo systemctl stop k3s` frees its pods
    (5 containerd shims survive the stop). It was not the crash cause.
  - Transfer code with `git archive` (NOT rsync — see
    [[dgx-transfer-git-archive-not-rsync]]). Model weights move dgx→Thor over the
    LAN with `tar -ch | ssh ... tar -x` into a FRESH directory; dgx reaches
    192.168.68.23 directly. The reimage CHANGED THE HOST KEY, so dgx's
    `known_hosts` needs `ssh-keygen -R 192.168.68.23` once.
  - **Oracle CAVEAT UPDATE (2026-08-12, `CLAIM-MM-SPEED-AUDIO-ENC-FA2`):** both venvs
    import cleanly now — `vllm-oracle-next` reports `0.23.1rc1.dev1511+g555967922`.
    That string is a **`setuptools_scm` nearest-ancestor-tag artefact, NOT an identity
    mismatch**: HEAD is `5559679229bc`, which IS the pin, and setuptools_scm names a
    dev build after the newest tag reachable from the commit, not after the release the
    commit belongs to. **Assert the oracle BY COMMIT, never by version string** — the
    recorded `0.26.0.dev0` string will not match and nothing is wrong. Its source tree
    `~/work/vllm-src-5559679` is present again (890 MB).
  - **`soundfile==0.14.0` is required in `~/venvs/vllm-oracle-next` for Voxtral
    (2026-08-12).** Without it the pin cannot tokenize the audio prompt at all, so the
    pinned oracle is not gateable for the Voxtral vehicle. Installed; recorded here and
    against issue #375. `~/venvs/vllm-oracle-v0.25.0-stage`
    (vLLM 0.25.0 + mistral_common 1.11.5 — the Voxtral golden-capture stack) **ran a full
    teacher-force to completion**, so the "crashes in EngineCore KV-cache/model init"
    below is at least partly a PATH artefact: in a non-login shell Triton's JIT dies
    `RuntimeError: Failed to find C compiler` AFTER the weights load, which surfaces as
    `EngineCore failed to start`. Export **`CC=/usr/bin/gcc`** alongside the documented
    `ninja` PATH fix and it runs. The `vllm-oracle` symlink still points at the 0.25.0
    rollback rather than the pin (issue #375, open).
  - **★ `CC=/usr/bin/gcc` is STALE for the reimaged host, and the correction is to
    run the oracle IN A CONTAINER (2026-08-17, `SPEC-MTP-K-GT-1`).** The bullet above
    is right about the failure and wrong about the cure on this host. Measured on
    `kairos-17dd`: there is no `gcc`, no `cc`, no `clang`, no `ninja` and no `nvcc`
    anywhere on the host, and `/usr/include` carries neither `stdio.h` nor
    `python3.12/Python.h`, so there are no glibc headers and no crt objects either.
    Exporting `CC=/usr/bin/gcc` therefore names a file that does not exist. Triton
    3.7.1 in the pinned venv ships only `ptxas`/`cuobjdump`/`nvdisasm`, no C
    compiler, so nothing in the venv supplies one.
    **What this looks like if you do not know it:** the weights load, the engine
    then dies `RuntimeError: Failed to find C compiler`, and vLLM reports
    `Engine core initialization failed. See root cause above. Failed core proc(s): {}`.
    That is an INSTRUMENT failure wearing the shape of a verdict about the model.
    Do NOT reach for `enforce_eager` to get past it: it is forbidden as a
    denominator, and it would silently change the thing being measured.
    **The cure**, and the shape `~/rs35b/run_oracle.sh` already used: run the host
    venv inside `nvidia/cuda:13.0.1-devel-ubuntu24.04` with
    `python3 python3-dev ninja-build build-essential libnuma1` installed, `-v
    $HOME:$HOME`, `CC=/usr/bin/gcc` and `/usr/local/cuda/bin` on `PATH`. The image
    ships python **3.12.3**, which matches the venv's `pyvenv.cfg` exactly, so the
    HOST venv resolves inside the container. Bake the toolchain into an image
    (`~/mtpgate/Dockerfile.oracle`, `mtpgate-oracle:1`) rather than `apt-get`ing it
    per leg: a leg that must reach the network to start can fail for a reason that
    has nothing to do with the measurement. Assert `gcc` and `ninja` INSIDE the
    container before the model loads, so a broken image aborts by name instead of
    four minutes later as an engine error. Container egress WAS available on
    2026-08-17; the box has been recorded without it before, which is the argument
    for baking rather than installing.
    **★ AND THE PINNED ORACLE CANNOT CURRENTLY LOAD A 27B HERE AT ALL: it eats the
    WHOLE MACHINE in the step after `torch.compile`, and `gpu_memory_utilization`
    does NOT control it.** Measured the same day, once the toolchain fix let an
    oracle get that far for the first time. At `gpu_memory_utilization=0.75` the
    engine held about **110 GiB of HOST RAM** while `nvidia-smi` reported only
    26 GiB on the device, hung 45 minutes at loadavg **260** with **0 GiB
    available**, and `sshd` stopped completing a banner exchange while the box
    still answered ICMP. Killing the container took it from 118 of 119 GiB used to
    4 of 119 in under ten seconds.
    **The obvious attribution to that 0.75 was tested and REFUTED.** A second run
    at **0.30**, with a 5-second host-memory sampler running, collapsed the same
    way: `avail_mb` 87683 at 09:00:47 and **0** at 09:02:25, loadavg 1.19 to 39.90.
    Weight loading finished with 66 GiB free and `torch.compile` finished with
    88 GiB free, so the collapse is neither of those. It is the step immediately
    AFTER compilation and it is insensitive to the KV-pool fraction, which points
    at the profiling forward and the graph capture (`max_num_batched_tokens=8192`,
    `cudagraph_capture_sizes: [1, 2, 4, 8]`, every allocation host-backed here).
    **That last part is a hypothesis with a located step, not a result.** Vary
    those one at a time with the sampler running and believe nothing without an
    A/B. **The 0.75 run THRASHED for 42 minutes and survived (`boot_id` and
    `uptime` unchanged); the 0.30 run REBOOTED THE BOX** — `boot_id` moved
    `5bbdc432…` to `bd5c6e7a…` and `journalctl --list-boots` shows boot `-1`
    ending 09:10:15Z against boot `0` beginning 09:13:55Z. So a lower fraction is
    NOT a safety margin: assume the box is at risk on every attempt. Always run a
    `MemAvailable` sampler beside any load here; `nvidia-smi` is blind to all of
    it, and the sampler is what turned a 45-minute mystery into a timestamped
    100-second collapse. While sshd was answering intermittently one connection
    returned `Permission denied (publickey)`; that is a memory-pressure artefact,
    not a credential problem, and the same key worked seconds after the reboot.
    **A cleanup trap is not a stop button.** Both DGX drivers used
    `trap cleanup EXIT INT TERM` where `cleanup` resets the clocks and RETURNS, so
    `SIGTERM` reset the clocks and the script then started its NEXT leg on a box
    with no memory left. Put an `exit` on the signal path, and `docker kill` the
    current named container inside the handler: `timeout` signals `docker run`, and
    the container outlives it.
  - **Oracle CAVEAT (2026-07-27):** the pinned vLLM oracle on dgx.casa was found
    DEGRADED — `~/venvs/vllm-oracle`→`vllm-oracle-next` (0.26.0.dev0) is an editable
    install whose source tree `~/work/vllm-src-5559679` was pruned (dangling; `import
    vllm` fails), and `~/venvs/vllm-oracle-v0.25.0-stage` (vLLM 0.25.0) now crashes in
    EngineCore KV-cache/model init. A fresh teacher-force could not be run this
    session; the sm_110 near-tie verdict rests on the COMMITTED gap-0 golden. Repair
    the oracle before the next gate that needs a fresh capture.
- **Ettore Apple/Metal profile**: `ssh 192.168.68.103` — Mac mini, Apple M4 (10 CPU
  cores), 16 GB unified memory, arm64, macOS 26.5.2. Use it for the MLX-backed
  `vt::` backend, Metal op parity, and small-model bring-up. It cannot hold the
  27B/35B gate models; gate-scale Apple performance needs a larger-memory Mac.
  **Re-verified 2026-07-22 (`CLAIM-BACKEND-FANOUT-1`), correcting the stale
  2026-07-10 line:** only the **Command Line Tools** are installed, NOT full
  Xcode — so the offline `metal` shader compiler is absent (`xcrun -sdk macosx
  metal` fails). That does **not** block MSL: runtime compilation via
  `newLibraryWithSource:` was verified working, together with a numerically
  correct dispatched compute kernel. **CMake IS already installed** (brew
  4.1.0 at `/opt/homebrew/bin`, missing from the non-interactive PATH — always
  `export PATH=/opt/homebrew/bin:$PATH` in remote commands); `ninja` is not
  (make works). **MLX is NOT installed** (`brew install mlx` -> 0.32.0, pulls
  `python@3.14`) and is not required for native-MSL bring-up. Device facts:
  `hasUnifiedMemory=YES`, `MTLGPUFamilyApple9` + `Metal3`, SIMD width 32,
  32 KiB threadgroup memory, 11.84 GiB recommended max working set, ~30 GiB
  free disk. Our tree configures AND builds there under AppleClang 21 with
  three Clang-only `-Werror` fixes, and 108,952 portable-tier assertions pass.
  **Updated 2026-07-22 (W0 landed):** the FULL tree (library + every test) now
  builds `-Werror`-clean on the M4 with the Metal backend ON, and the fix count
  is **seven**, not three — a full build surfaced four more than the spike's
  lib-only probe (see the fan-out spec § Work breakdown "W0 landed"). Configure
  with plain `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`: `VLLM_CPP_METAL`
  defaults to `AUTO` and turns itself ON for an Apple host with an ObjC++
  compiler. Add `-DVLLM_CPP_METAL=OFF` for a CPU-only A/B.

  **TWO PRE-EXISTING macOS TEST GAPS — expected failures, not regressions.**
  Both fail identically with `-DVLLM_CPP_METAL=OFF`, i.e. they are unrelated to
  Metal, and neither is fixed yet:
  - `test_serve_low_tools` — the Python bench tooling calls Linux-only
    `os.sched_getaffinity` (`tests/tools/test_gdn_packed_component.py`) and
    `POSIX_FADV_DONTNEED` (`tools/bench/drop_file_cache.py`).
  - `test_safetensors` — `MappingRssKb` reads `/proc/self/smaps`, which macOS
    does not provide, so it returns 0 and the RSS assertions cannot hold.

  `test_capi` and `test_openai_conformance` are ctest-PARALLELISM flakes on this
  box (and on Linux); they pass on rerun. Prefer `ctest -j 3` **here, on this
  16 GB Mac mini only** — it is not general advice, and in particular the DGX
  profile above requires `-j 1` because its unified memory OOM-reboots the box
  under a parallel CUDA suite.
  `test_engine_core_proc` is likewise a timing flake under heavy parallel ctest:
  the case "EngineCoreProc: abort-mode shutdown aborts in-flight requests"
  (`test_engine_core_proc.cpp:315`) races the busy-loop teardown against the
  abort-output enqueue and intermittently misses `abort_seen` (measured ~1/5 in
  isolation under load, 2026-07-28); it is a pre-existing test-side timing race
  (untouched by the C7 sampling work) and passes on rerun. A dedicated fix would
  make the loop block for the abort frame (bounded wait) instead of a
  best-effort non-blocking `try_get` sweep.

  **LOCALAI WORKER — must be DOWN for any timing/benchmark work on this box
  (user-directed 2026-07-22).** It is a **root LaunchDaemon**, not a container
  and not a user LaunchAgent:

  | | |
  |---|---|
  | Unit | `system/com.localai.worker` |
  | Plist | `/Library/LaunchDaemons/com.localai.worker.plist` (root:wheel) |
  | Program | `/Users/mudler/local-ai/local-ai worker` (a NATS-driven worker) |
  | Properties | `keepalive | runatload` — so `kill`ing the PID is NOT enough, launchd restarts it |
  | Log | `/Users/mudler/local-ai/worker.log` |
  | State observed 2026-07-22 | **running**, PID 327, RSS ~51 MB, up 1d08h, **MEASURED idle: 0.0% CPU, and `ioreg IOAccelerator PerformanceStatistics` reports `Device Utilization % = 0`, `Renderer Utilization % = 0`, `Tiler Utilization % = 0` — it holds NO GPU work.** Log shows only periodic `NATS backend.list` events (~1 per 6 h); no model loaded |

  ```sh
  # inspect (works WITHOUT root)
  launchctl print system/com.localai.worker
  # stop  (NEEDS root; bootout, because KeepAlive would restart a killed process)
  sudo launchctl bootout system/com.localai.worker
  # restore to the observed state
  sudo launchctl bootstrap system /Library/LaunchDaemons/com.localai.worker.plist
  launchctl print system/com.localai.worker | grep state   # expect: running
  ```

  **NOT STOPPED during W0**, for two reasons, both recorded deliberately:
  (1) stopping it needs an interactive `sudo` password and this box has no
  passwordless sudo (`sudo -n true` -> "a password is required"), so an agent
  cannot do it unattended; (2) W0 took **no timing measurement whatsoever** —
  every gate is a functional/correctness assertion — so contention could not
  affect any recorded result. **The next agent doing MLX-vs-ours benchmarking
  MUST get the user to run the bootout above first; any Metal timing taken with
  this daemon up is VOID.** Note also three `actions.runner.localai-org-*` GitHub
  Actions runners as user LaunchAgents (PIDs 599/600/601) which can start CI jobs
  on this box at any time — quiesce those too before a benchmark series
  (`launchctl bootout gui/$UID/actions.runner.localai-org-<name>.<label>`).

  **STILL NOT STOPPED as of the 2026-07-22 MLX baseline run** — same reason
  (no passwordless sudo). The MLX numbers in
  [docs/BENCHMARKS.md](../docs/BENCHMARKS.md) are therefore recorded
  **`BLOCKED-ON-SUDO` / INDICATIVE, not binding**; the recipe is a one-command
  re-run once the user boots the daemon out. **Second contender found the same
  session and not anticipated by the earlier note:** the desktop **aerial video
  wallpaper** — `WallpaperAerialsExtension` (PID 472, **8.2% CPU**) plus
  `VTDecoderXPCService` (PID 518, 2.2%) — decodes video continuously and touches
  the GPU; it is the actual source of the ~1.47 load average on an otherwise idle
  box. Disable it (System Settings -> Wallpaper, or log the console user out)
  before any binding run. It was left untouched.

  **MLX IS NOW INSTALLED (2026-07-22), via the venv route as recommended** —
  brew was NOT used, so `python@3.14` never entered `/opt/homebrew/bin` and the
  PATH our macOS builds use is unchanged:

  ```sh
  /usr/bin/python3 -m venv ~/mlx-venv && ~/mlx-venv/bin/pip install -U pip mlx-lm
  ```

  | | |
  |---|---|
  | Resolved versions | **`mlx` 0.29.3, `mlx-metal` 0.29.3, `mlx-lm` 0.29.1** — the CLT python 3.9.6 caps the resolve BELOW brew's 0.32.0. Record this: an unpinned competitor arm is not a floor |
  | Location | `~/mlx-venv` (off every build PATH), `~/hf-cache` (3.2 GB model cache) |
  | Model | `mlx-community/Qwen3-1.7B-bf16` @ rev `9cd6692855d3e06772228e9a962b2606359b2d24` |
  | Ships prebuilt | `mlx/lib/mlx.metallib` **104,894,650 bytes** + `libmlx.dylib` — so CONSUMING MlX needs no Xcode, but BUILDING it from source does (`xcrun metal`), which this box cannot do |
  | Device probe | `mx.metal.device_info()` -> `applegpu_g16g`, `max_recommended_working_set_size` 12,713,115,648 (11.84 GiB), `max_buffer_length` 9,534,832,640 |
  | Removal | `rm -rf ~/mlx-venv ~/hf-cache` — neither is on any PATH our builds consult |

## Benchmark models on Ettore's dgx.casa profile

- `~/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4`
  (snapshot complete, ~22G, 3 safetensors shards — re-downloaded 2026-07-03
  after the original snapshot was found incomplete)
- `~/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-NVFP4`
- `~/work/apex/qwen36_35b/Qwen3.6-35B-A3B-APEX-*.gguf` (GGUF-gate inputs)
- `~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B` (fast tests)

## Gate model architecture (from GGUF metadata, arch `qwen35moe`)

40 blocks = 10 × (3 GDN + 1 full-attn); hidden 2048; full-attn GQA 16q/2kv,
partial RoPE 64 dims (MRoPE sections [11,11,10,0]), rope base 1e7; MoE 256
experts top-8 + 1 shared (expert FFN 512); GDN: conv kernel 4, 16 groups,
inner 4096, state 128; context 262144.

## Prior art on Ettore's dgx.casa profile (mudler's llama.cpp patch series — mine for GB10 kernels)

- `~/killgate_series/` — NVFP4 W4A4 FP4 MMA prefill, qwen35moe NVFP4
  quant/dedup, MoE decode regraph
- `~/llama-phase93-qwen3next-gqa-bcast`
- `~/llama-phase84-attn-only-source`

## TODO

- Binding immutable `3f256ab` remains **55/124 axes pass, 69 fail** against
  vLLM v0.25.0. Finalized c2 root `179a0fc` already maps the executed path and
  selects the complete **193 vs 97** GDN projection mismatch. W1 merged BA is
  implemented/`GATING`. Clean pushed `581d335` under
  `~/work/vllm.cpp-gdn-ba/immutable-581d335…` passes the exact CUDA 13.0.88 /
  CUTLASS / Triton-AOT build, packed F32/BF16 capture/replay, strict memcheck,
  merged/split 27B and inert native-35B gates; the isolated BF16/decomposed
  control fails the token near-tie. Immutable `0091cd1`, finalized by pushed `8a1f923`, is
  `complete-structural`: both exact-c2 arms pass all 24 local range contracts at
  merged 963/145 versus split 1,011/193, with 48 BF16-only removals and unchanged
  selected non-BF16 families. Clean `f344dec` closes packed W1D2/G2 for
  default+rollback 27B **235/235**, 35B/GGUF inertness and strict safety.
  `benchmark_binding=false`. Close paired node traces and the c2/c16 component
  before qkvz.
  Independently remove **22.920
  GiB** host-weight mirror and overlapping source pages. No 35B performance
  command runs before all 27B axes pass.
- Keep the existing SGLang v0.5.13 P1 evidence immutable. The distinct
  shared-prefix gate pins v0.5.15 `f63458b` and image digest `d0a667e`; its PX1
  deterministic 64k/256k harness/counter work is ready after the priority
  cache-off closure. Write the dedicated `KV-MAMBA-ALIGN` spike before PX2,
  then require matched BF16/no-spec capacity, native hit/no-eviction evidence,
  full axes and traces. Never mutate the vLLM oracle while provisioning SGLang.
- ~~Bootstrap CMake + MLX on the M4 host before the Metal backend bring-up.~~
  **RESOLVED/SUPERSEDED 2026-07-22** by the [backend fan-out
  spike](specs/backend-fanout-metal-vulkan-xpu.md): CMake is already present,
  and MLX is **not** a bring-up prerequisite (native MSL compiles at runtime
  with CLT only, so E2 precedes E1). The real prerequisite is spike work item
  `W0` — chiefly the `CMakeLists.txt:304-306` Apple `-force_load` fix, without
  which every static registrar is silently dropped on macOS and even the CPU
  backend fails to register. `brew install mlx` is deferred to work row `M5`.
  **CLOSED 2026-07-22: `W0` LANDED** — the `-force_load` fix is in and
  `test_backend` is 7/7 on the M4, so the M4 is fully usable for backend work.
  **REOPENED in a different role:** MLX must now be installed on the M4 as the
  **competitor BENCHMARK arm** (user directive; `BACKEND-GATE-METAL-MLXLM`),
  which is independent of its demotion as an implementation path. Use the venv
  route recorded in the M4 entry above, and stop the LocalAI worker daemon
  first.
- **Vulkan runtime is already usable and needs no acquisition.** dgx GB10
  enumerates as a real Vulkan `INTEGRATED_GPU` at API 1.4.312 (loader 1.4.328 +
  NVIDIA ICD) with `VK_KHR_cooperative_matrix` v2 and `VK_NV_cooperative_matrix2`;
  the dev box enumerates `llvmpipe` (Vulkan 1.4.318, CPU) for GPU-free CI.
  Optional still: `libvulkan-dev` and `vulkan-tools` (neither is needed to build
  or gate — the backend `dlopen`s the loader and vendors the Khronos TYPE headers).
- **Vulkan shader toolchain — glslang 16.5.0, installed 2026-08-06 (`VK-A1`).**
  `$HOME/tools/glslang-16.5.0/bin/glslang`, from the upstream prebuilt Linux
  x86_64 release tarball; no root, nothing linked (it is a build-time tool, never
  a dependency — `.agents/discipline.md`). Put that directory on `PATH` to
  regenerate committed SPIR-V with `scripts/gen-vulkan-spirv.py`.
  **Two measured facts about the pin.** (1) `src/vt/vulkan/vulkan_spirv.h:16`
  records `Glslang Version: 11:16.4.0`, but **16.4.0 ships NO release assets** —
  only `16.5.0` and `main-tot` do, and Ubuntu packages `15.1.0` — so the recorded
  version cannot be fetched and cannot back a CI gate. (2) The committed SPIR-V
  nonetheless reproduces **byte-for-byte under 16.5.0** (`--check` passes, exit 0),
  which proves the committed artifact is what it claims AND that the emitted
  SPIR-V is stable across a glslang minor bump. The freshness gate therefore pins
  the DOWNLOAD URL rather than asserting a version string.
  The older note here — that Ubuntu's shaderc 2023.8 `glslc` is too old for the
  coopmat2 feature probe — still holds and is why the system package is not used.
- **dgx Vulkan/llama.cpp comparison toolchain (2026-08-07, `VK-E`).** apt:
  `glslc`, `glslang-tools`, `libvulkan-dev`, `spirv-headers`. **Ubuntu's `glslc`
  is shaderc 2023.8 and is NOT USABLE for a fair llama.cpp-Vulkan build** — it
  disables FOUR of five fast paths (`GL_NV_cooperative_matrix2`,
  `..._decode_vector`, `GL_EXT_integer_dot_product`, `GL_EXT_bfloat16`), leaving
  only `GL_KHR_cooperative_matrix`, and the build still configures and runs. A
  source-built shaderc `v2026.4-dev` lives at `/tmp/shaderc/b/glslc/glslc`; pass
  `-DVulkan_GLSLC_EXECUTABLE=` to it. **Verify the runtime banner says
  `matrix cores: NV_coopmat2` before trusting any number.** llama.cpp is
  unpacked at `~/lcpp-vk` with `build-vk/bin/llama-bench` built, **and that tree
  is the SUPERSEDED fork `237ad9b96`, not the pin.** Do not reuse it. The
  llama.cpp oracle is stock `b10451` since 2026-08-16
  ([`oracles/llama-cpp.md`](oracles/llama-cpp.md)). `237ad9b96` is a local-only
  commit on the developer's `localai-paged` branch, 65 of our own performance
  commits past upstream `b9827`, built from a working tree with 27 uncommitted
  entries. `~/lcpp-vk` therefore reproduces neither the pin nor any identifiable
  object. The `BENCH-VK-LLAMA` decode `4.36 vs 4.35 MET` measured with it is the
  **most fragile verdict** in the enumeration, a 0.23% margin inside a 0.69%
  spread, and re-taking it is owed under
  [#1003](https://github.com/mudler/vllm.cpp/issues/1003). Unpack the pinned SHA
  fresh and assert `git status --porcelain` empty before recording any number.
  Enumeration and the clean-tree rule:
  [`specs/oracle-llamacpp-repin-stock.md`](specs/oracle-llamacpp-repin-stock.md).

- **No Intel GPU exists on any box here**, so `BACKEND-XPU` end-to-end work is
  HW-BLOCKED; only policy-port, compile coverage and oneAPI CPU-device unit
  numerics are available.
