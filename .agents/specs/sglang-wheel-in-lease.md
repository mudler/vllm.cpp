# Reaching the pinned SGLang oracle from an `rc` lease, through a PyPI wheel

Row: `SGLANG-ORACLE-LEASE-WHEEL`.
Issue: [#1265](https://github.com/mudler/vllm.cpp/issues/1265).
Follows: [#1185](https://github.com/mudler/vllm.cpp/issues/1185),
[#1213](https://github.com/mudler/vllm.cpp/issues/1213),
[#1354](https://github.com/mudler/vllm.cpp/issues/1354).
Shaped on [`oracle-wheel-in-lease.md`](oracle-wheel-in-lease.md), which solved
the same problem for the vLLM oracle.

## Scope

Plan one route, commit the evidence it rests on, and correct the two records
this evidence falsifies.

The route: install the pinned SGLang oracle from PyPI wheels into a virtual
environment inside an `rc` lease on `dgx:gpu0`, assert its identity against the
pin by content, and serve the gate model. No container image, no `ssh`, no file
mutex outside a lease.

**W1 planned the route and W2 walked it.** The scope below is written in two
parts for that reason: W1 was deliberately read-only, and saying so was the
point of the row. W2 added the driver and two `rc` jobs. A reader who quotes a
W1 sentence about a W2 result has crossed the line this row exists to hold.

In scope, W1 (2026-08-19):

- This spec, holding the route, the identity gate, the non-claims, and the
  static evidence measured on 2026-08-19.
- [`sglang-wheel-in-lease.json`](sglang-wheel-in-lease.json), the committed
  per-file manifest the identity gate compares against.
- One row `SGLANG-ORACLE-LEASE-WHEEL` in
  [`../sglang-matrix.md`](../sglang-matrix.md), under "Oracle stand-up".
- The correction to [`../oracles/sglang.md`](../oracles/sglang.md), which calls
  the aarch64 kernel-wheel coverage untested. It is tested and it passed.
- The record that `scripts/dgx-sglang-low-concurrency.sh` needs a replacement
  rather than a patch.

In scope, W2 (2026-08-23):

- [`../../scripts/rc-sglang-oracle-lease.sh`](../../scripts/rc-sglang-oracle-lease.sh),
  the replacement driver, and
  [`../../scripts/sglang_lease_identity.py`](../../scripts/sglang_lease_identity.py),
  the identity gate split out of it.
- [`../../tests/scripts/test_sglang_lease_identity.py`](../../tests/scripts/test_sglang_lease_identity.py),
  registered in `scripts/agent-preflight.sh` and on the `agent-record` CI lane.
- Two `rc` jobs on `dgx:gpu0` and everything in
  `## W2, and what the lease measured on 2026-08-23`.
- The `gateable` decision, against the five conditions in
  `## The exit criterion` and against nothing else.

Out of scope:

- **W1 only:** any `rc` job, any lease, any install, any GPU. Every number in
  `## The evidence` comes from static analysis of published artifacts. W2 lifts
  this exclusion for itself and for nothing earlier in the file.
- Any change to the `pin` key in [`../oracles/sglang.md`](../oracles/sglang.md).
  One pinned commit gains a second delivery artifact. The commit does not move.
- Any flip of `gateable` on partial evidence. See `## The exit criterion`.
- Any second oracle id. See `## One pin, two delivery artifacts`.
- Any product code. This row touches records, documents, and the operator
  scripts that drive its own jobs. Nothing under `src/` or `include/`.
- Editing an existing [`../issue-index.md`](../issue-index.md) row. That file is
  append-only, #1265 already has a row, and a second row for the same issue
  would duplicate a key rather than append a new one.

## The wall this row removes

[`../oracles/sglang.md`](../oracles/sglang.md) records `gateable = no` since
2026-08-18. The oracle ran once, on 2026-07-28, and the method it ran by is now
forbidden: `ssh` to `dgx.casa`, `docker run` on the
`lmsysorg/sglang:v0.5.15-cu130@sha256:d0a667e` arm64 image, and one
`flock $GPU_LOCK`. `AGENTS.md` makes the `rc` lease the required path to a fleet
device, and `rc run` has no `--image` flag.

Three read-only checks on 2026-08-18 found no permitted substitute: `rc run
--help` carries no `--image`; the `rc describe dgx:gpu0` usage sheet names no
container runtime; and `kubectl` is not the answer, because a sibling pod holds
the GPU outside the accounting the lease exists to keep.

The vLLM oracle already crossed the same wall without an image. It builds,
installs, imports, and sees the GB10 inside a lease
([#1185](https://github.com/mudler/vllm.cpp/issues/1185)), and on 2026-08-19 it
served a 52 GiB bf16 checkpoint through three clean benchmark legs from a lease
([`../environment.md`](../environment.md), "Clock pinning does NOT work inside
an `rc` lease"). SGLang needs no source build at all, which makes its route
shorter than the one vLLM already walked.

## What is measured here, and what is not

Everything in `## The evidence` is **static analysis of published artifacts**,
measured on 2026-08-19 from the developer workstation. Nothing in THAT section
ran on a GPU, in a lease, or against a model. Read every claim in it as a
property of a file, never as a property of a run.

The one thing that mattered and was **not** established by W1: that the route
works. Reachability is a present-tense property. A wheel whose contents are
correct is not an oracle until a job installs it, serves the gate model, and
completes a leg. **That is what W2 measured**, and it is reported separately in
`## W2, and what the lease measured on 2026-08-23` so that the two never merge
into one undifferentiated claim.

## The evidence

### The wheels, by content

| Artifact | Size (bytes) | sha256 |
|---|---:|---|
| `sglang-0.5.15-cp312-cp312-manylinux_2_34_aarch64.whl` | 12,716,006 | `1c2d2602b4ba04c6a71d2f3bf2e3654da53987536f0d65dbe4f57cdc65c9812e` |
| `sglang_kernel-0.4.4-cp310-abi3-manylinux2014_aarch64.whl` | 34,243,333 | `727e4bc53abeade20260186f99199200320b9fa51f8de7af90c01524cff73e5d` |
| `sglang_kernel-0.4.4-cp310-abi3-manylinux2014_x86_64.whl` | 615,071,908 | `558bc7035e3c0a795e8c3eca3cc29e189c7d29f1db2471a0a71b9156a8ee7fa1` |

Both aarch64 files were downloaded and hashed. Both hashes match the values the
PyPI JSON index reports and the values [#1265](https://github.com/mudler/vllm.cpp/issues/1265)
recorded.

### `sm_121a` is covered, and this was the question that could have killed the route

The aarch64 kernel wheel is 5.6% of the x86_64 one. That ratio is what
[#1265](https://github.com/mudler/vllm.cpp/issues/1265) recorded as leaving the
`sm_121a` coverage unestablished. It is established now, and the coverage is
complete AND accelerated.

`sgl_kernel/sm100/common_ops.abi3.so`, sha256
`672a1cb425d2d2815b8138c34fd8a1e89603bac9a8986dc9db7883e04be535d3` as the wheel
stores it, carries a `.nv_fatbin` section at offset 3,901,952 of 14,996,776
bytes. Walked container by container it holds **56 fatbin containers, each with
exactly 6 entries, 336 in total**, every entry of kind 2 (cubin) and none of
kind 1 (PTX). All 56 containers declare the same architecture set,
`{90, 100, 103, 110, 120, 121}`, and **zero containers lack 121**.

NVIDIA's own listing tool names each of those 336. `cuobjdump -lelf`, release
13.0, V13.0.85, taken from the CUDA redistributable archive
`cuda_cuobjdump-linux-x86_64-13.0.85-archive.tar.xz`, sha256
`bd624dcb2089842add8f293efab1d21aa076c98b36d8dcb64347fe08fb03315d` as NVIDIA's
`redistrib_13.0.2.json` publishes it, emits 336 lines whose target histogram is
exactly even:

```text
56 sm_100a   56 sm_103a   56 sm_110a   56 sm_120a   56 sm_121a   56 sm_90
```

**The 121 cubins are `sm_121a`, the architecture-accelerated variant, and every
one of the 56 containers carries one.** The 90 cubins are the plain ones. The
route's conclusion therefore improves rather than narrows: GB10's coverage is
complete AND accelerated. `cuobjdump -lptx` on the same file prints `No PTX file
found to extract`, which agrees with the entry kinds above, so nothing depends
on run-time compilation from PTX either.

The build recipe agrees, and it explains why the shape is exactly six.
`sgl-kernel/CMakeLists.txt` at the pin:

| Anchor | Target | Condition |
|---|---|---|
| `:127` | `-gencode=arch=compute_90,code=sm_90` | unconditional, in the base flag list |
| `:209-210` | `sm_100a`, `sm_120a` | `CUDA_VERSION >= 12.8` |
| `:215` | `sm_103a` | `CUDA_VERSION >= 13.0` |
| `:220-221` | `sm_110a`, **`sm_121a`** | `CUDA_VERSION >= 13.0` **and** `aarch64` |

`:99-101` sets `ENABLE_BELOW_SM90` off on aarch64, so `sm_80`, `sm_89` and
`sm_87` (`:196-201`) are never emitted. `sm_90a` (`:235`) is gated on
`SGL_KERNEL_ENABLE_FA3`, which `:108-114` defaults **off** on aarch64. Six
targets in, six cubins per container out, 1:1 — and the one switch that leaves
`sm_90` unaccelerated is the same switch that drops `flash_ops.abi3.so` from the
wheel.

The declared architecture was also cross-checked against the payload rather than
trusted, and that cross-check stands, for exactly what it measures. Each of the
336 zstd payloads was decompressed and its ELF `e_flags` read:
`(e_flags >> 8) & 0xFF` agrees with the fatbin's declared architecture in **336
of 336** cubins, with zero disagreements. **Declared architecture equals payload
architecture. That is the whole of what `e_flags` proves here.**

**`e_flags` does not carry the base-versus-`a` distinction on these files, and
an earlier reading of this row concluded the exact inverse by assuming it did.**
All 336 cubins are `EI_ABIVERSION = 8`, the new ABI, in which
`llvm/include/llvm/BinaryFormat/ELF.h` puts the architecture in
`EF_CUDA_SM_MASK = 0xff00` at `EF_CUDA_SM_OFFSET = 8` and the accelerator marker
at `EF_CUDA_ACCELERATORS = 0x8`. That header is llvm-project `main` read on
2026-08-19 and this repository pins no LLVM revision, so the constant names are
the anchor here and the line numbers deliberately are not. The six distinct values, one per target, are:

| `cuobjdump` target | `e_flags` | `(f >> 8) & 0xff` | `f & 0x8` | low byte |
|---|---|---:|---|---|
| `sm_90` | `0x06005a04` | 90 | clear | `0x04` |
| `sm_100a` | `0x06006402` | 100 | clear | `0x02` |
| `sm_103a` | `0x06006702` | 103 | clear | `0x02` |
| `sm_110a` | `0x06006e02` | 110 | clear | `0x02` |
| `sm_120a` | `0x06007802` | 120 | clear | `0x02` |
| `sm_121a` | `0x06007902` | 121 | clear | `0x02` |

`EF_CUDA_ACCELERATORS` is clear in all 336, including in every cubin `cuobjdump`
names accelerated, so it does not separate them. The other candidate is worse
than useless. `EF_CUDA_ACCELERATORS_V1 = 0x800` is the OLD ABI's
marker, from when the architecture lived in the low byte under
`EF_CUDA_SM = 0xff`, and under the new ABI `0x800` falls INSIDE
`EF_CUDA_SM_MASK`. It is an architecture bit here rather than a flag: it reads
SET in `0x06007902` (`sm_121a`, architecture `0x79`) and equally SET in
`0x06005a04` (plain `sm_90`, architecture `0x5a`). Neither bit ever supported a
claim about the variant. The low byte separates `0x02` from `0x04` by
coincidence of which target is which, and reading it as the accelerator marker
inverts the answer in both directions. Read the variant from `cuobjdump` and
from the gencode line. Never from `e_flags`.

`sgl_kernel/load_utils.py::_load_architecture_specific_ops` selects `sm90/` when
the compute capability is exactly 90 and `sm100/` in every other case, including
when no GPU is present. GB10 reports 121, so it loads `sm100/`. The same file
already names this machine: `_preload_cuda_library` comments that "On CUDA 13
systems (e.g., DGX Spark), only libcudart.so.13 exists".

### The 5.6% size gap is one absent library, not thinner coverage

The aarch64 wheel holds 80 entries against 81 for x86_64. The missing entry is
`sgl_kernel/flash_ops.abi3.so`, the FlashAttention-3 extension. In the x86_64
wheel it is **578,045,190 bytes compressed** and **1,758,623,416 bytes
uncompressed**; both numbers come from that wheel's zip central directory, read
by HTTP range request, so the 615 MB artifact itself was never downloaded. Its
absence is a build decision and not a packaging accident:
`sgl-kernel/CMakeLists.txt:108-114` defaults `SGL_KERNEL_ENABLE_FA3` off on
`aarch64`, the same switch that withholds the `sm_90a` gencode at `:235`.

It is unreachable on GB10 in any case: `sgl_kernel/flash_attn.py:25-28` returns
`True` only when `torch.cuda.get_device_capability()[0]` is 8 or 9, and GB10 is
12. `sgl_kernel/flash_attn.py:7-12` raises `ImportError` at module import when
`flash_ops` is absent, so the module is unusable on aarch64 as a whole, not only
its predicate.

`grep -rn "sgl_kernel.flash_attn" python/sglang/` at the pin returns **eight**
lines. Seven of them import the module; the eighth is a `torch.ops` dispatch
name that never touches it. This table is the complete set, not a sample:

| Site | Shape | On the GB10 dense text path? |
|---|---|---|
| `srt/layers/attention/flashattention_backend.py:307` | deferred, inside the `fa_impl_ver == 3` branch | No. `fa3` is not the resolved backend on this device. |
| `srt/layers/attention/xpu_backend.py:25` | **module level, unguarded** | No. The module is imported only when the XPU backend is selected. |
| `multimodal_gen/runtime/layers/attention/backends/xpu_backend.py:11` | module level, in a `try` that re-raises | No. Same, in the `multimodal_gen` runtime. |
| `srt/layers/attention/vision.py:62` | module level under `if _is_xpu:` | No. `_is_xpu` is `False` on CUDA. |
| `srt/models/mimo_audio.py:28` | module level under `if is_cuda():` | No. A model module. It WOULD raise on GB10, but only if MiMo-Audio were loaded. |
| `srt/models/mimo_v2_asr.py:42` | deferred, inside `try` / `except ImportError` | No. A model module, and it already handles the absence. |
| `jit_kernel/flash_attention_v3.py:72` | deferred, in `_load_fa3_kernel_from_sgl` | No. Only on the FA3 fallback path. |
| `srt/layers/attention/vision.py:54` | `torch.ops.sgl_kernel.flash_attn_varlen_func`, under `_is_cpu and _is_cpu_amx_available` | Not an import of the module at all. |

Four of the seven imports are module level, so on aarch64 the missing
`flash_ops` is a hard `ImportError` for anything that imports those four
modules. Nothing on the Qwen3.8 dense text path imports any of them.

### The resolved backends on `sm_121`

`srt/utils/common.py:285-289` defines `is_sm100_supported` as compute-capability
major in `[10]`. GB10 is major 12, so it is `False`.
`srt/utils/common.py:280-284` defines `is_sm120_supported` as major in `[12]`
with CUDA at least 12.8, so that one is `True`.

For a non-MLA model, `srt/server_args.py:4337-4361` falls through the Hopper
branch, the `is_sm100_supported` branch, the ROCm branch and the MPS branch, and
reaches `:4359`, which reads
`if is_flashinfer_available() and not model_config.has_attention_sinks:`. The
branch has **two** conjuncts, not one: it returns `flashinfer` only when both
hold, and `triton` at `:4361` otherwise. Qwen3.8 dense declares no attention
sinks, so the second conjunct is immaterial for this model. Quote the branch
whole regardless, because a model that does declare them resolves differently. A
container run resolves the same way, because the predicate reads the device and
not the packaging.

**Capture this from the server log in the lease. Do not assume it.** The branch
above is what the source says; what the server prints is what ran.

### Identity is assertable by content

`sglang/_version.py` in the wheel sets `__commit_id__ = commit_id = None`. The
installed package therefore carries **no** runtime assertion of the commit it
was built from, which is the reason this row commits a manifest instead of
printing a version string and calling it identity.

The wheel's `sglang/` tree was compared per file against the GitHub source
tarball at `f63458b5beaceabbd9d749b9fc956370e1b649e6`:

| Measure | Value |
|---|---:|
| files in the wheel's `sglang/` tree | 3338 |
| regular files under `python/sglang/` in the source | 3335 |
| common | 3335 |
| byte-identical | 3335 |
| differing | 0 |
| only in the source | 0 |

The three wheel-only entries are accounted for, not waved past.
`sglang/srt/mem_cache/cpp_radix_tree/.clang-format` is a symlink in the
repository to `sgl-kernel/.clang-format`, which the build resolves into a real
file; its content matches the link target byte for byte, which makes the honest
count **3336 of 3336**. The remaining two are build products with no source
counterpart: the generated `sglang/_version.py` and the compiled Rust gRPC
extension `sglang/srt/grpc/_core.cpython-312-aarch64-linux-gnu.so`.

### No source build, and no dependency wall

The pinned vLLM oracle needs a source build because its version is a development
version absent from PyPI. SGLang v0.5.15 is a released version with a prebuilt
aarch64 wheel, so the SGLang route has no compile step at all.

The wheel declares 70 hard requirements. Every one of them resolves for
linux / aarch64 / cp312 from a prebuilt wheel, either an aarch64 wheel or a
`py3-none-any` wheel. No hard requirement is sdist-only on this platform, so
nothing in the dependency set needs a compiler either. The entries that a reader
would expect to be the wall all have aarch64 wheels: `flash-attn-4==4.0.0b15`
(`py3-none-any`), `tilelang==0.1.11` (`cp38-abi3-manylinux_2_34_aarch64`),
`sgl-deep-gemm==0.1.4`, `tokenspeed_mla==0.1.7`, `torch_memory_saver`,
`llguidance`, `apache-tvm-ffi==0.1.11` and `nvidia-cutlass-dsl==4.5.2`.
`torch==2.11.0` has `torch-2.11.0-cp312-cp312-manylinux_2_28_aarch64.whl`,
419,731,115 bytes.

Upstream installs the same two things the same way. At the pin,
`docker/Dockerfile:1` sets `ARG CUDA_VERSION=13.0.1` over the base image
`nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04`, `:13` sets
`ARG SGL_KERNEL_VERSION=0.4.4`, `:210` runs
`pip install sglang-kernel==${SGL_KERNEL_VERSION} --force-reinstall --no-deps`
from PyPI for that CUDA version, and `:242` runs
`pip install --extra-index-url https://download.pytorch.org/whl/cu130 ".[all]"`.
The staging script uses the same extra index for the same reason: PyPI's default
`torch` build variant is not the one upstream resolves.

### The download budget

Fourteen artifacts alone come to 0.94 GiB, before the transitive NVIDIA runtime
wheels `torch` pulls. The two largest are
`flashinfer_cubin-0.6.12-py3-none-any.whl` at 447,533,460 bytes and the `torch`
wheel at 419,731,115 bytes. Size the job's runtime ceiling against a gigabyte-
scale download over the worker's egress, not against the 12 MB sglang wheel.

## The staging script

The script lives on `/workspace` and the job runs it by path. **Never inline the
work into `rc run --`.** A detaching client kills the job, and the Bash tool's
ten-minute cap makes detaching the normal case.

Required properties, each one paid for by a job this fleet has already lost:

1. **Unconditional environment repair with an asserted postcondition.** The `rc`
   worker container is reused between jobs. A `command -v nvcc` guard skipped an
   install once and cost the job that followed it, which reported
   `nvcc already in place` and then failed for the reason the guard was meant to
   remove. Install, then assert, then print the assertion's exit code.
2. **Build and install in container-local `/tmp`, then `cp -rL` to
   `/workspace`.** `/workspace` is CIFS with `nounix`, so it stores no symlink,
   and it presents `file_mode=0664`, so a file there has no exec bit and
   `chmod +x` returns `Operation not permitted`. A virtual environment created
   directly on the share loses its links and cannot run its own entry points.
   `-L` is the part that matters: it copies link targets as files.
3. **`apt-get install -y python3-dev` before any install.** `Python.h` is absent
   from the worker image, and a package that needs it fails late.
4. **A runtime ceiling with room for the download.** A 90-minute ceiling already
   killed a virtual-environment copy mid-flight and left a partial tree that
   looked installed to a directory listing. A partial `pip --target` tree has
   done the same: `markupsafe` existed as a dist-info with no package files.
   Print the exit code of the copy, and verify the destination after it.
5. **One log per run, and no step that can hang while holding the lease.**
6. **Every step prints `<NAME>_RC=$?` on its own line**, so a reader can tell a
   skipped step from a passed one. `cmd | tail` reports the exit code of `tail`.

## The identity gate

The identity gate is the **first** thing that runs after the install and before
anything touches the GPU. It mirrors `IDENTITY_RC` from
[`oracle-wheel-in-lease.md`](oracle-wheel-in-lease.md).

1. Resolve the installed package from outside any source tree, as `cd /` then
   `python3 -c 'import sglang; print(sglang.__file__)'`. A path inside a
   checkout means the gate read the wrong tree.
2. Re-derive the per-file sha256 of every file under the installed `sglang/`
   directory, excluding `__pycache__/`, which pip writes and the wheel does not
   carry.
3. Compare the result against
   [`sglang-wheel-in-lease.json`](sglang-wheel-in-lease.json) in this
   repository. **Abort non-zero on any mismatch, any missing file, and any extra
   file.** Print the first ten offenders, never a bare count.
4. Print `sglang.__version__`, `torch.version.cuda`,
   `torch.cuda.get_device_capability()`, and the `sgl_kernel` load line naming
   which of `sm90/` and `sm100/` was selected.
5. Print `IDENTITY_RC=$?` on its own line.

The manifest is cp312 and linux-aarch64 only, because the gRPC extension's file
name carries the interpreter tag and the machine tag. Another interpreter or
another architecture owes its own manifest and must not be checked against this
one.

`sgl_kernel` is **not** in the manifest. It is a separately versioned package
and it is asserted by its own wheel sha256, recorded in `## The evidence` and in
[`../oracles/sglang.md`](../oracles/sglang.md).

## One pin, two delivery artifacts

The wheel sha256 values and the manifest become a **second artifact under the
existing `sglang` pin**. They do not become a second oracle id, and they do not
move `pin`.

The argument, in three parts:

- **An oracle id names an upstream, not a delivery.** `AGENTS.md` admits a fixed
  set of oracles by upstream repository, and
  `scripts/check-oracle-pins.py` gates the AGENTS.md table and the
  `.agents/oracles/` directory against each other in both directions. A second
  id for the same repository would need a second row in that table, which would
  assert that SGLang is two oracles. It is one.
- **The commit is the same commit.** The wheel's `sglang/` tree is byte-identical
  to `f63458b5` for every file that exists in both. A delivery that reproduces
  the pinned tree exactly is the pin, arriving by a different road.
- **Two artifacts under one pin is the honest shape, because they are not
  interchangeable.** The image ran and is now unreachable. The wheel is
  reachable and has not run. Recording both under one pin keeps that difference
  visible; recording two pins would imply two upstreams and hide it.

## The exit criterion

**`gateable` was `no` until a job served a model and completed a leg. On
2026-08-23 one did.** The conditions below are the bar this row set for itself
in advance, and the verdict table at the end of this section reports against
them one by one.

`AGENTS.md` defines gateability as present reachability: the oracle "must
demonstrably build and run the model". Static analysis of a wheel is not a run.
`scripts/check-oracle-pins.py` admits no third value, so a caveat carried only
in prose would leave the machine-readable value reading `yes` for a reader who
greps the block and budgets an arm nobody has produced.

The flip is an explicit exit criterion of this row rather than something a later
reader infers:

**`SGLANG-ORACLE-LEASE-WHEEL` may set `gateable = yes` when, and only when, all
five hold in one recorded run:**

1. An `rc` job on `dgx:gpu0` installed the two wheels named in `## The evidence`
   and reported their sha256 values from inside the job.
2. `IDENTITY_RC=0` against the committed manifest, asserted from `cd /`.
3. The SGLang server reached readiness on the gate model, and the log names the
   resolved attention backend and the resolved MoE runner.
4. One benchmark leg completed with zero errors and the recorded output-token
   count.
5. `evidence` in the pin block moved from `#1265` to a path that exists in this
   tree, which is what `check-oracle-pins.py` requires of `gateable = yes`.

Anything short of all five leaves `gateable = no` and leaves #1265 open. A
partial result is recorded as a partial result.

**The verdict, 2026-08-23:**

| # | Condition | Verdict |
|---|---|---|
| 1 | an `rc` job on `dgx:gpu0` installed both wheels and reported their sha256 from inside the job | **MET.** Job `86282a1a`, `12716006`/`1c2d2602…` and `34243333`/`727e4bc5…`, both `WHEEL_SHA_OK=1`. |
| 2 | `IDENTITY_RC=0` against the committed manifest, asserted from `cd /` | **MET, twice.** 3338 of 3338, 0 missing, 0 extra, 0 differing, in job `86282a1a` and again in job `b9e7709d`. |
| 3 | the server reached readiness and the log names the resolved attention backend and MoE runner | **MET.** `READY=1` at 454 s on `/health_generate`; `attention_backend='flashinfer'` and `moe_runner_backend='auto'`, both read from the server's own log. |
| 4 | one leg completed with zero errors and the recorded output-token count | **MET.** Timed c1: 6 of 6 completed, 0 failed, 768 output tokens = 6 x 128 exactly. |
| 5 | `evidence` moved off `#1265` to a path that exists in this tree | **MET.** `evidence = .agents/specs/sglang-wheel-in-lease.md`. |

Five of five. **`gateable` moves to `yes`.**

**What the flip does NOT say.** It says the pinned oracle builds nothing, installs
from a wheel, loads the gate model and completes a clean leg inside a lease. It
does not say a number taken this way is a floor, and it does not close #1265's
remaining debt. The c8 leg is VOID, the job never printed its own teardown
assertion, the clock spread breached the 5% ceiling at 7.59%, and no vllm.cpp arm
ran beside any of it. Every one of those is in `## Owed`.

**Why `evidence` points at this spec rather than at a new file under
`docs/bench-evidence/`.** `check-oracle-pins.py` wants a path that exists. The
whole W2 record is here, and a second copy under `docs/` would be a measurement
of one file stored inside another, which `AGENTS.md` §Records forbids for exactly
the drift reason: the copy goes stale and nothing notices. `transformers.md`
already points its `evidence` at a spec.

## W2, and what the lease measured on 2026-08-23

W1 planned this. W2 wrote the driver and ran it. Everything in this section is a
job result on `dgx:gpu0`, reached only through `rc run`. No `ssh`, no container
image, and no file mutex taken outside the lease.

### The driver

`scripts/dgx-sglang-low-concurrency.sh` is NOT patched, for the reason
`## Two record corrections this change makes` gives: its missing half is the
forbidden path. The replacement it owed is
[`../../scripts/rc-sglang-oracle-lease.sh`](../../scripts/rc-sglang-oracle-lease.sh),
with the identity gate split out as
[`../../scripts/sglang_lease_identity.py`](../../scripts/sglang_lease_identity.py)
so that CI can hold the gate's own logic without a fleet device.

Both were staged onto the shared `/workspace` and the job ran them BY PATH.
Nothing was inlined into `rc run --`, because a detaching client kills the job.
The two phases are separate submissions on purpose: `install` stops at the
identity gate and touches no model, so a failed identity costs no model time,
and `serve` re-asserts identity before it loads a weight, because the worker
container is reused and the virtual environment in `/tmp` can be gone.

### 1. Both wheels, hashed inside the job

Job `86282a1a-6e07-4099-b2e8-f4768aa714e8`, worker `rc-worker-4b8lj`,
2026-08-23T20:35:04Z to 21:03:34Z, 28 min 30 s, exit 0. Evidence at
`/mnt/nas_share/rc/sglang-w2/out/install-20260823T203504Z/`, which the worker
sees as `/workspace/sglang-w2/out/install-20260823T203504Z/`.

```text
WHEEL sglang-0.5.15-cp312-cp312-manylinux_2_34_aarch64.whl
  size=12716006
  sha256=1c2d2602b4ba04c6a71d2f3bf2e3654da53987536f0d65dbe4f57cdc65c9812e
  WHEEL_SHA_OK=1
WHEEL sglang_kernel-0.4.4-cp310-abi3-manylinux2014_aarch64.whl
  size=34243333
  sha256=727e4bc53abeade20260186f99199200320b9fa51f8de7af90c01524cff73e5d
  WHEEL_SHA_OK=1
```

`pip download` resolved both from PyPI and the job hashed the bytes that landed,
against the values `## The evidence` committed on 2026-08-19. A remote hash
would have proved nothing about the file on this disk. `SGLANG_INSTALL_RC=0`
and `KERNEL_INSTALL_RC=0`, in upstream's own order and from upstream's own
index: the sglang wheel with `--extra-index-url https://download.pytorch.org/whl/cu130`
(`docker/Dockerfile:242`), then the kernel wheel with `--force-reinstall
--no-deps` (`docker/Dockerfile:210`).

**`JITCACHE_STATE=INSTALLED`.** `flashinfer-jit-cache==0.6.12+cu130` resolved
from `https://flashinfer.ai/whl/cu130` and installed, `JITCACHE_RC=0`. That
closes the difference `## What this route does NOT establish` marked as
conditional and unverified rather than accepting it.

The resolved stack, from the job's own `pip freeze` (199 lines, archived beside
the log):

| Package | Version |
|---|---|
| `sglang` | `0.5.15` (the wheel above, by sha256) |
| `sglang-kernel` | `0.4.4` (the wheel above, by sha256) |
| `torch` | `2.11.0+cu130` |
| `transformers` | `5.12.1` |
| `flashinfer-python` | `0.6.12` |
| `flashinfer-cubin` | `0.6.12` |
| `flashinfer-jit-cache` | `0.6.12+cu130` |
| `nvidia-cutlass-dsl` | `4.5.2` |
| `sgl-deep-gemm` | `0.1.4` |
| `triton` | `3.6.0` |
| `flash-attn-4` | `4.0.0b15` |

No compiler ran for any of them. The CUDA toolkit was installed anyway, because
FlashInfer JITs at run time and the worker image carries none; `cuda-toolkit-13-0`
from NVIDIA's `sbsa` repository, `NVCC_POSTCONDITION_OK=1` asserted on
`cuda_runtime.h` and `libcudart`, not on the presence of the `nvcc` binary.

**The install is base dependencies, not `[all]`.** `## The evidence` verified the
70 HARD requirements, and those are what this installs. `docker/Dockerfile:242`
installs `".[${BUILD_TYPE}]"` with `BUILD_TYPE=all`, which adds `ray`, tracing
and the diffusion set. None of them is on the dense text path, and the
difference is stated here rather than discovered later.

### 2. `IDENTITY_RC=0` against the committed manifest

```text
sglang.__file__ = /tmp/sgenv/lib/python3.12/site-packages/sglang/__init__.py
sglang.__version__ = 0.5.15
manifest_files=3338 derived_files=3338
missing=0 extra=0 differing=0
IDENTITY OK: 3338 files match the manifest for pin f63458b5beaceabbd9d749b9fc956370e1b649e6
IDENTITY_RC=0
```

Asserted from `cd /`, so it read the installed package and not a source tree.
3338 of 3338, zero missing, zero extra, zero differing. This is the assertion
that stands in for the runtime commit check `sglang/_version.py` does not carry.

The gate is mutation-proven by `tests/scripts/test_sglang_lease_identity.py`
rather than only read: one changed byte, a missing file, an extra file, a run
from a cwd that is not `/`, and a byte-identical SOURCE TREE each turn it red,
and the fixture returns green after each revert.

### The environment the numbers carry

```text
torch.__version__  = 2.11.0+cu130
torch.version.cuda = 13.0
cuda_available     = True
device_name        = NVIDIA GB10
capability         = (12, 1)
is_sm100_supported = False
is_sm120_supported = True
is_flashinfer_available = True
LGC_RC=4      The current user does not have permission to change clocks
```

The three predicates are measured on the device, and they are the ones
`srt/server_args.py:4337-4361` branches on. `LGC_RC=4` reproduces
[#1354](https://github.com/mudler/vllm.cpp/issues/1354) on a fourth job: the SM
clock is sampled and never pinned inside a lease.

### 3 and 4: the model run

Job `b9e7709d-cc96-4247-9d01-c611bce707ac`, `dgx:gpu0`, started
2026-08-23T21:51:56Z. Evidence at
`/mnt/nas_share/rc/sglang-w2/out/serve-20260823T215156Z/`.

The virtual environment from the install job was still there, so this job
re-asserted identity in 11 s and never touched PyPI: `IDENTITY_RC=0`, 3338 of
3338, a second time and on a second job. It then copied the gate checkpoint off
CIFS into container-local `/tmp` (`CKPT_COPY_RC=0`, 1166 s, 55,586,040,114
bytes) and served it.

The gate model is `/workspace/ckpt/qwen3.8-27b-hf`, architecture
`Qwen3_5ForConditionalGeneration`, `model_type` `qwen3_5`, bf16. The
configuration is chosen to COMPLETE, and it is NOT the recorded denominator:

```text
--mem-fraction-static 0.80  --max-running-requests 32  --context-length 2048
--chunked-prefill-size 8192 --disable-radix-cache --random-seed 0
```

**Condition 3: readiness, and the backends as THE LOG NAMES THEM.**
`READY=1`, `SECONDS_TO_READY=454` against `/health_generate`, which generates
rather than merely answering a liveness probe.

```text
attention_backend='flashinfer'
moe_runner_backend='auto'
mamba_backend='triton'
linear_attn_backend='triton'
[22:12:13] Attention backend not specified. Use flashinfer backend by default.
[22:12:28] Multimodal attention backend not set. Use triton_attn.
[22:12:13] Breakable CUDA graph is incompatible with multimodal model; disabling prefill CUDA graph.
[22:18:22] Load weight end. elapsed=353.55 s, type=Qwen3_5ForConditionalGeneration, avail mem=59.87 GB, mem usage=51.86 GB.
[22:18:26] KV Cache is allocated. dtype: torch.bfloat16, #tokens: 535124, K size: 16.33 GB, V size: 16.33 GB
[22:18:27] Using hybrid linear attention backend for hybrid GDN models.
[22:18:58] Capture target decode CUDA graph end. elapsed=30.54 s, mem usage=1.38 GB, avail mem=19.66 GB.
[22:18:58] max_total_num_tokens=535124, chunked_prefill_size=8192, max_prefill_tokens=16384, max_running_requests=32, context_len=2048, available_gpu_mem=19.66 GB
```

`flashinfer` is what `## The evidence` predicted from
`srt/server_args.py:4337-4361`, and the server says so in its own words rather
than the prediction being taken for the answer. The three device predicates the
branch reads were also measured, in the install job.

**`moe_runner_backend='auto'` is NAMED and UNEXERCISED, and those are different
things.** Qwen3.8-27B is dense-with-hybrid-GDN: `layer_types` alternates
`linear_attention` and `full_attention` and there is no expert layer, so `auto`
is the declared default of a code path this model never enters. Condition 3 asks
what the log names, and this is what it names. **A MoE arm is owed** before any
claim about SGLang's MoE runner on this device.

Decode CUDA graphs ARE captured, at batch sizes 1, 2, 4, 8, 12, 16, 24 and 32.
Prefill graphs are disabled by the server's own rule for a multimodal model, and
that is upstream behaviour rather than a flag of ours.

**Condition 4: one leg, complete, zero errors, exact token count.**

| Leg | Prompts | Completed | Failed | Input tokens | Output tokens | Duration (s) | Output tok/s | Median TTFT (ms) | Median TPOT (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| warmup c1, DISCARDED | 6 | 6 | 0 | 6144 | 768 | 173.31 | 4.431 | 1063.2 | 218.73 |
| **timed c1** | 6 | **6** | **0** | 6144 | **768** | 173.23 | 4.433 | 1078.1 | 218.81 |
| c8 | 48 | -- | -- | -- | -- | -- | -- | -- | -- |

768 output tokens is exactly `6 x 128`, which is what `ignore_eos` guarantees and
what makes the count an assertion rather than an observation. `sglang-oai`, so
`/v1/completions`, the same client path the 2026-07-28 image run drove.

**The warm cache worked, and the two c1 legs are the measurement of that.**
4.431 against 4.433 output tok/s is a 0.05% difference between an unwarmed first
leg and the leg after it. Without `flashinfer-jit-cache` the first leg compiles
kernels in-process and would not resemble the second. This is why the spec
insisted on closing that difference rather than accepting it.

**The c8 leg is VOID, because the BOX WENT DOWN DURING it.** During, and not
under: the leg had reached only its single warmup request, and the paragraph
below the reboot table shows that concurrency 8 was never issued. At
2026-08-23T22:27:00Z everything stopped at once: the server log's last line is a
decode batch at 22:26:54, the clock sampler's last sample is 22:26:58, the
memory sampler's is 22:27:00, and no teardown, `TEARDOWN_VERDICT` or
`DONE_MARKER` was printed.

The first reading of this was that the job had been killed — the backgrounded
`rc run` client streaming it had been killed shortly before, and
`.agents/environment.md` records that an `rc run` submit dies with its detaching
client. **That reading was wrong**, and the thing that falsified it is a field
neither the job nor the reader was looking at. The NEXT job to run on `dgx:gpu0`
started at 22:32:47Z and printed a different `boot_id`:

Every row below names the file it was read from, because two of the four rows
are another row's artifact and a reader cannot check a reading whose tree is not
named.

| Job | Time | `boot_id` | last PID in `/proc/loadavg` | read from |
|---|---|---|---|---|
| this row's install | 20:35:04Z | `02d5a76f-697c-4adb-830d-7465d49aa792` | -- | `/mnt/nas_share/rc/sglang-w2/out/install-20260823T203504Z/job.log`, line 8 |
| this row's serve | 21:51:56Z | `02d5a76f-697c-4adb-830d-7465d49aa792` | -- | `/mnt/nas_share/rc/sglang-w2/out/serve-20260823T215156Z/job.log`, line 8 |
| another session's, on the same box | 21:04:04Z | `02d5a76f-697c-4adb-830d-7465d49aa792` | 3510 | `/mnt/nas_share/rc/gdn-moe-packed-ba/logs/gate-ab.log`, lines 1-3 |
| another session's, on the same box | **22:32:47Z** | **`26394f62-37c5-4fc9-885a-c8faba9d35ac`** | **594** | `/mnt/nas_share/rc/gdn-moe-packed-ba/logs/gate-ab.log`, lines 18-20 |

The two `gate-ab.log` rows are the load-bearing ones, and they are six lines of
one file. Both are `### START` blocks of the same three-line shape, so the
reading is a comparison of like with like:

```text
### START 2026-08-23T21:04:04Z host=rc-worker-4b8lj arch=aarch64
boot_id=02d5a76f-697c-4adb-830d-7465d49aa792
loadavg=0.59 0.44 0.60 2/865 3510
### START 2026-08-23T22:32:47Z host=rc-worker-4b8lj arch=aarch64
boot_id=26394f62-37c5-4fc9-885a-c8faba9d35ac
loadavg=1.31 0.57 0.21 1/870 594
```

**The machine rebooted between 22:27:00Z and 22:32:47Z.** The `boot_id` changed
and the kernel's PID counter fell from 3510 to 594, which is a fresh boot and not
a recreated pod: the pod name `rc-worker-4b8lj` is the same on both sides of it.
`.agents/environment.md` records a GB10 unified-memory collapse that reboots this
box rather than OOM-killing a process, and this is that signature. Attribution
reaches this far and no further: this row held the lease, so the only load on the
box was ours, and a 52 GB model was resident. Nothing in the job's own output
names a cause, which is the hole in this result.

**Concurrency 8 was NEVER ISSUED, and the leg's name is the whole reason anyone
would think otherwise.** The leg is called `sglang-c8` because it was invoked as
`--num-prompts 48 --max-concurrency 8`. It never got there.
`sglang-c8.log` is **0 bytes**. The server log's last four entries are the entire
life of that leg:

```text
[2026-08-23 22:26:41] INFO: 127.0.0.1:43140 - "GET /v1/models HTTP/1.1" 200 OK
[2026-08-23 22:26:47] Prefill batch, #new-seq: 1, ... #running-req: 0, #queue-req: 0
[2026-08-23 22:26:47] INFO: 127.0.0.1:55142 - "POST /v1/completions HTTP/1.1" 200 OK
[2026-08-23 22:26:54] Decode batch, #running-req: 1, ... gen throughput (token/s): 1.64
```

`/mnt/nas_share/rc/sglang-w2/out/serve-20260823T215156Z/sglang-server.log`, last
four non-blank lines. The `GET /v1/models` is the client's readiness check, and
what follows it is `warmup_requests=1` — the default this benchmark runs before
its main loop, visible in the `benchmark_args` line of every leg log and printed
by the c1 legs as `Starting warmup with 1 sequences...`. **One request was
running at the moment of death**, on a 6-prompt-per-point corpus that would have
put at most 8 in flight had the main loop ever started. `#queue-req: 0` on the
last prefill says nothing was even waiting.

So the collapse landed on a single in-flight request against a resident 52 GB
model, not under an 8-way load. What distinguishes that moment from the eleven
minutes of identical single-request decode before it is **not known**, and the
falling `gen throughput` of that last line — 1.64 tok/s against the 4.57 the c1
legs held — is one sample and is not an explanation. Why the box went down stays
open below.

**The watchdog did not fire, and could not have.** Its floor was 5,000 MB.
The last eight `MemAvailable` samples, two seconds apart, read 17074, 17251,
17269, 17296, 17323, 17338, 14935 and 15449 MB. **The box went down with 15 GB
still available.**

**This reproduces a rule this repository already wrote down**, and the
reproduction is the only new part. [`../environment.md`](../environment.md)
records that "a sampling watchdog cannot guard the REBOOT CLASS of failure at
all", because a userspace sampler dies with the kernel, and it records the same
5,000 MB floor never being crossed on an earlier worker loss. A future run
wanting protection needs a different instrument; the floor in
`scripts/rc-sglang-oracle-lease.sh` is left in place because it costs nothing and
it is NOT protection. That same file also told this row to print `boot_id` in
every leased job that loads anything large, which is the only reason the cause of
this one is known at all.

**The reboot also answers the teardown question**, which is the one good
consequence. A reboot returns every resource, and the job at 22:32:47Z ran on the
box afterwards, so nothing of this row's was stranded holding the GPU. The
assertion is still made rather than inferred, by
[`../../scripts/rc-sglang-lease-reap.sh`](../../scripts/rc-sglang-lease-reap.sh),
staged as `/workspace/sglang-w2/reap.sh`, which matches on the venv interpreter
path `/tmp/sgenv/bin/python` and never on a launcher name. A broad `pkill -f` on
a script name is the recorded failure that stranded an EngineCore holding 23 GB
across three jobs.

**That job ran, and it says so — but the job archived nothing, and that is a
defect in the script rather than in the reading.** The version of
`rc-sglang-lease-reap.sh` that ran wrote only to stdout. Unlike `install` and
`serve`, which both `exec > >(tee -a "$OUT/job.log")`, it never set `$OUT`, so
`/mnt/nas_share/rc/sglang-w2/out/` held no `reap-*` directory and the verdict
below existed nowhere on the shared `/workspace`. Read that sentence before the
numbers: for a day the four lines under it were prose in this file and nothing
else. The script now tees like its siblings, so a future reap archives itself.

**Where the output actually survived, and how to re-derive it.** The resource
controller keeps its own copy of every job's output, and that copy is the
primary source here:

```sh
rc logs 0f84b66d-1c30-4de5-bdb8-ee7b058f284a
```

Job `0f84b66d-1c30-4de5-bdb8-ee7b058f284a` on `dgx:gpu0`, worker
`rc-worker-4b8lj`, 2026-08-23T23:10:48Z. Its complete output, 412 bytes,
verbatim:

```text
### reap Sun Aug 23 23:10:48 UTC 2026
rc-worker-4b8lj
26394f62-37c5-4fc9-885a-c8faba9d35ac
pid, process_name, used_gpu_memory [MiB]
COMPUTE_APPS=0
SGENV_PROCS=0
NOTHING STRANDED: no process is running our venv interpreter.
Mem:             119           4          58           0          58         115
pid, process_name, used_gpu_memory [MiB]
overlay         3.6T  1.3T  2.2T  36% /
DONE_MARKER_SGLANG_W2_REAP
```

The third line is the `boot_id`, printed bare by `cat
/proc/sys/kernel/random/boot_id`. Both `nvidia-smi --query-compute-apps` calls
print their header and no row, which is what `COMPUTE_APPS=0` counts.

**A durable second copy now exists**, recovered from the controller on
2026-08-24 and written to
`/mnt/nas_share/rc/sglang-w2/out/reap-20260823T231048Z/job.log`, with
`PROVENANCE.txt` beside it stating that it is a post-hoc copy and not a
job-written artifact. Cite the `rc logs` command first; the file is the backup
for the day the controller's store rolls over.

Three things follow. The GPU holds no compute process, so the resource IS
returned and no longer only inferred. Nothing is running this row's virtual
environment. And the `boot_id` it read is the POST-reboot one, which is a THIRD
job independently confirming the reboot rather than the same reading twice.

**`/tmp/sgenv` and `/tmp/ckpt38` did not survive.** The `du -sh` of both printed
nothing, so a re-run pays the full install again — about 28 minutes and roughly
a gigabyte of downloads for the wheels plus 1.75 GB for the JIT cache — and then
another 20 minutes to re-copy the 52 GB checkpoint off CIFS. The `serve` phase of
`scripts/rc-sglang-oracle-lease.sh` reinstalls when the environment is absent for
exactly this reason.

**Clock attribution, derived from the RAW sample file.** The step that would
have summarised it never ran, so the summary below is derived from
`clocks.samples` directly. A decimated summary hides exactly the rows it appears
to rule out, which is why every distinct throttle value is enumerated with its
count rather than sampled.

```text
clock_samples=214  window_s=441
sm_clock_mhz median=2489 min=2346 max=2535 spread=7.59%
BREACHES_5PCT
gpu_temp_c min=50 max=84
throttle_reason x210: 0x0000000000000000
throttle_reason x4:   0x0000000000000020   (SW thermal slowdown)
```

**7.59% against a 5% ceiling.** The clock could not be pinned (`LGC_RC=4`), the
GB10 reached 84 C, and software thermal slowdown was active in 4 of 214 samples.
This is the same regime `#1354` recorded at 12.92% to 26.36% on the vLLM
campaign. **No ratio may be divided out of these numbers**, and none is offered.

**Host memory, the first-class artifact.** 435 samples over 897 s.
`MemAvailable` fell from 117,688 MB to a minimum of **14,935 MB**, which is the
headroom `--mem-fraction-static 0.80` leaves on this box for this model. The
watchdog floor was 5,000 MB and never fired. A later reader setting a floor for
this configuration should start from 14,935 and not from a guess.

## What this route does NOT establish

Read this section before quoting any line above it.

**The virtual environment is NOT shown to be equivalent to the container.**
Nobody has inspected the aarch64 contents of
`lmsysorg/sglang:v0.5.15-cu130@sha256:d0a667e`. What is established is narrower
and stated exactly: the sglang Python tree in the wheel is identical to the
pinned source tree, and upstream's own cu130 Dockerfile installs the same kernel
wheel from the same index. Everything around those two facts is different and
unmeasured.

The named differences, and whether the Qwen3.8 path touches them:

| Difference | Reaches the Qwen3.8 dense text path? |
|---|---|
| Base image `nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04` versus the `rc` worker's Ubuntu 24.04 plus a per-job CUDA install | **Yes.** The CUDA runtime and cuDNN under `torch` differ in provenance. Record both `torch.version.cuda` and the driver version in the job. |
| The image's FlashInfer JIT cache (`Dockerfile:329`, stage `flashinfer_cache`) — **unverified, and off by default** | **Conditional, and closable rather than acceptable.** `Dockerfile:21` is `ARG INSTALL_FLASHINFER_JIT_CACHE=0`, so the stage at `:329-349` is a no-op unless the build passes `=1`, and the `COPY` at `:462` then copies an empty directory. Whether `sha256:d0a667e` was built with `=1` is **unknown**, because nobody has read that image. Do not accept this one as a stated difference: close it. `flashinfer_jit_cache-0.6.12+cu130-cp39-abi3-manylinux_2_28_aarch64.whl` **exists** on `https://flashinfer.ai/whl/cu130/flashinfer-jit-cache/`, sha256 `097ee785b4886719761cda20d8363957d4b158e937c6dcde3ba84ea774065048`, 1,750,271,858 bytes, and `Dockerfile:22` pins `FLASHINFER_VERSION=0.6.12`, so the venv can install the identical artifact the image would have installed. Budget the 1.75 GB in the job ceiling. Without the cache the first request compiles kernels in-process, and the first leg measures the JIT compiler rather than the model. |
| `sgl-deep-gemm` and the CUTLASS DSL | **Possibly.** Both are hard requirements and install from wheels. Whether the dense path calls them is a question for the server log, not for this spec. |
| DeepEP (`Dockerfile:257`, stage `deepep_builder`) | **No.** Expert-parallel all-to-all across ranks, and this is one device with no expert parallelism. |
| mooncake and NIXL KV connectors | **No.** Disaggregated prefill and remote KV storage, neither of which a single-box leg enables. |
| The gateway and devtools stages (`Dockerfile:354`, `:406`) | **No.** Serving front-end packaging and developer tooling. |
| FlashAttention-3 (`flash_ops`) | **No.** Absent from the aarch64 kernel wheel and refused by capability on GB10 in any case. |

The two rows marked **Yes** and the one marked **Possibly** are an **accepted,
stated difference**, not an equivalence. A number taken through this route is a
number taken through this route. Write it that way.

**One more content-level fact, inherited and not re-derived here.**
`flashinfer_cubin-0.6.12` carries the architecture tokens `sm100f`, `sm103a`,
`sm100a` and `sm110a`, and **zero** `sm_120` or `sm_121` tokens, so the
TRTLLM-gen cubin paths are dead on GB10 by content and not only by the
`is_sm100_supported()` gate. That artifact is 447,533,460 bytes and was not
downloaded for this row. Re-derive it before citing it as measured here.

## Measurement expectations, set in advance

Two lease-wide constraints already paid for on 2026-08-19 bind this arm, and a
reader must not discover them after the GPU time is spent.

- **Clock pinning is refused inside a lease**
  ([#1354](https://github.com/mudler/vllm.cpp/issues/1354)). `nvidia-smi -lgc`
  returns `LGC_RC=4` in a job running as root. The SM clock can be sampled and
  not pinned.
- **`gpu_clock_state compare` returned `PAIRING_VERDICT=DISCARD` on every c1
  pairing of the Qwen3.8-27B campaign**, on within-run spreads of 12.92% to
  26.36% against a 5% ceiling, even though the cross-arm rule passed perfectly
  at 0.0% offset on the same boot.

**Budget for absolutes.** A SGLang-versus-ours ratio can be undividable for
exactly the same reason, and this row does not promise one. Plan the campaign so
that two clean complete absolutes are a result rather than a consolation, and
report the pairing verdict beside every ratio that does survive.

Do not widen the clock assertion to turn a red green. #1354 records the three
admissible fixes, and each is its own row.

## Two record corrections this change makes

1. [`../oracles/sglang.md`](../oracles/sglang.md) calls the aarch64 kernel-wheel
   coverage "untested". It is tested and it passed: 56 fatbins, 6 cubins each,
   an `sm_121a` cubin in every one, and 336 of 336 payloads agreeing with their
   declared architecture. That file also carried the inverted reading, that the
   121 cubins were plain `sm_121` and gave GB10 no architecture-accelerated
   path. Both statements are corrected in the same change, and the corrected one
   is the stronger claim.
2. `scripts/dgx-sglang-low-concurrency.sh` is unrunnable and stays that way.
   `:10-11` hard-codes
   `docker.io/lmsysorg/sglang:v0.5.13-cu130-runtime@sha256:9631280f…` and passes
   it to `run_serve_low.py plan` as `--image`; `:55-57` refuses every mode
   except `--dry-run`; and its default pins v0.5.13 rather than the oracle's
   v0.5.15. Being precise about what is there: the script has **no execution
   half at all**. Its `--dry-run` half emits a plan, and `:5-7` says the image
   pull and the `docker run` are what **P2 would have added**. So the venv route
   needs a **new driver** rather than a patch, because the half that was never
   written is exactly the forbidden path, and the part worth keeping is the
   corpus definition rather than the launcher.

## Risks

- **The static reading ages.** PyPI files are immutable, so the two wheel hashes
  cannot drift. The worker image can, in either direction, which is how
  [#1146](https://github.com/mudler/vllm.cpp/issues/1146) came to exist. Every
  claim about the worker carries its box and its date.
- **A reader takes "the wheel is correct" for "the oracle runs".**
  `## What is measured here, and what is not` exists for that reader, and the
  same non-claim rides in every record this row edits.
- **The first leg pays for JIT compilation.** Whether the container ships a warm
  FlashInfer cache is unverified — `Dockerfile:21` defaults that stage off — but
  the venv certainly has none until one is installed. An unwarmed first leg
  measures the compiler. Install `flashinfer-jit-cache==0.6.12` from
  `https://flashinfer.ai/whl/cu130`, which is 1.75 GB on aarch64, or warm it
  explicitly and discard the first leg.
- **The job ceiling versus a gigabyte of downloads.** A ceiling that kills the
  install mid-copy leaves a tree that looks installed. Assert the postcondition;
  do not read the directory.
- **`gateable` drifts to `yes` on partial evidence.** `## The exit criterion`
  names five conditions and requires all five.

## Dependencies

- An `rc` lease on `dgx:gpu0`, and network egress from the worker to PyPI and to
  `download.pytorch.org`.
- Authority for a gigabyte-scale download inside the job. This row does not
  carry it.
- The gate checkpoint on the NAS, reachable from `/workspace`.
- Nothing in this tree. This row adds no product code and no test.

## Work breakdown

| Item | Owner | Output |
|---|---|---|
| W1 | this row | The spec, the manifest, the matrix row, and the two record corrections. **Done, 2026-08-19.** |
| W2 | this row | `scripts/rc-sglang-oracle-lease.sh`, `scripts/sglang_lease_identity.py`, `tests/scripts/test_sglang_lease_identity.py`. **Done, 2026-08-23.** |
| W3 | this row | One `rc` job that installs, asserts identity, and reports the environment. No model. **Done, 2026-08-23**, job `86282a1a`. |
| W4 | this row | One `rc` job that serves the gate model and completes a leg, with the resolved backends read from the server log. **Done, 2026-08-23**, job `b9e7709d`; c1 clean, c8 VOID. |
| W5 | this row | The `gateable` flip, if and only if all five conditions in `## The exit criterion` hold. **Taken, 2026-08-23.** Five of five. |

## Upstream chain

Read at `f63458b5beaceabbd9d749b9fc956370e1b649e6`, tag `v0.5.15`:

- `docker/Dockerfile:1,13,210,242` — the CUDA version, the kernel version, the
  PyPI kernel install, and the cu130 extra index.
- `python/sglang/srt/utils/common.py:280-284,285-289` — `is_sm120_supported` and
  `is_sm100_supported`.
- `python/sglang/srt/server_args.py:4337-4361` — attention-backend resolution for
  a non-MLA model.
- `python/sglang/srt/layers/attention/flashattention_backend.py:307`,
  `srt/layers/attention/xpu_backend.py:25`,
  `srt/layers/attention/vision.py:54,62`, `srt/models/mimo_audio.py:28`,
  `srt/models/mimo_v2_asr.py:42`, `jit_kernel/flash_attention_v3.py:72` and
  `multimodal_gen/runtime/layers/attention/backends/xpu_backend.py:11` — every
  `sgl_kernel.flash_attn` line the pin carries, four of them module level.
- `sgl-kernel/CMakeLists.txt:99-101,108-114,127,196-201,209-210,215,220-221,235,390-476`
  — the gencode list, the two aarch64 defaults, and the whole FA3 target.
- `docker/Dockerfile:21-22,329-349,462` — the FlashInfer JIT cache argument, the
  stage it gates, and the copy that carries whatever that stage produced.

Read in `sglang_kernel-0.4.4-cp310-abi3-manylinux2014_aarch64.whl`:

- `sgl_kernel/load_utils.py::_load_architecture_specific_ops` — the `sm90`
  versus `sm100` selection.
- `sgl_kernel/flash_attn.py:7-12,25-28` — the module-level import and
  `is_fa3_supported`.

## Our baseline

Ours is `main`, measured through the same corpus and the same concurrency as the
SGLang arm, on the same box, in the same lease window. SGLang binds as a
performance floor only, and only for a model whose own correctness gate already
passed. The greedy token cross-check is `SGLANG-ORACLE-CORRECT` and stays
`INVENTORIED`.

## Port map

None. This row ports no upstream code. It records a route to an oracle.

## Tests to port

None from upstream. SGLang has no test of a manifest this repository wrote.

**W2 adds one of its own, and the earlier "no test" reasoning was too wide.**
The paragraph below used to end this section: the gate runs inside an `rc` lease
against a fleet device, CI has no fleet device, so the gate is not reproducible
there. That is true of the LEASE HALF and false of the gate's own logic.
`tests/scripts/test_sglang_lease_identity.py` runs `scripts/sglang_lease_identity.py`
against a scratch fixture and asserts it goes RED on each defect it promises to
catch -- one changed byte, a missing file, an extra file, a run from a cwd that
is not `/`, and a source tree whose bytes match but which is not an installed
package. Each mutation asserts that it APPLIED before it reads the verdict, and
the suite re-asserts the green case after reverting, because a mutation that
never applied reads as a passing test.

The same suite is the only executing code in this tree that opens
[`sglang-wheel-in-lease.json`](sglang-wheel-in-lease.json). It checks the
manifest against the pin block it stands in for: same commit, same oracle id,
`file_count` equal to the table it counts, every key under the declared root,
no key naming an excluded directory, every value a sha256, and both wheel
hashes present in [`../oracles/sglang.md`](../oracles/sglang.md). That closes
the `## Owed` item which recorded the manifest as landing unreached.

## Gates

```sh
scripts/agent-preflight.sh --fail-on-skip
```

The full preflight is this row's gate, and since W2 it runs
`tests/scripts/test_sglang_lease_identity.py` inside that preflight and on the
`agent-record` CI lane. W1 added no behavior and no test. W2 adds the driver,
the identity gate and that suite, so the preflight is no longer a records-only
gate for this row.

## Evidence

| Claim | How it was checked, 2026-08-19 |
|---|---|
| wheel hashes | Both aarch64 wheels downloaded and hashed; both match the PyPI index and #1265. |
| `sm_121a` coverage | `.nv_fatbin` parsed from `sgl_kernel/sm100/common_ops.abi3.so`: 56 containers, 336 entries, all kind 2 (cubin), architecture set `{90,100,103,110,120,121}` in every container, 0 containers without 121. |
| the 121 cubins are `sm_121a` | `cuobjdump -lelf` (13.0, V13.0.85) names 56 `sm_121a` against 56 plain `sm_90` among the 336, and `sgl-kernel/CMakeLists.txt:221` builds `-gencode=arch=compute_121a,code=sm_121a` on aarch64 at CUDA >= 13.0 while `:127` builds plain `sm_90`. Six gencode targets, six cubins per container. |
| declared architecture is real | 336 payloads zstd-decompressed; `(e_flags >> 8) & 0xFF` agrees with the declared architecture 336 of 336 times. That agreement is the whole of what `e_flags` establishes here. |
| `e_flags` cannot answer the variant question | `EF_CUDA_ACCELERATORS = 0x8` is clear in all 336 values, and `EF_CUDA_ACCELERATORS_V1 = 0x800` is the old ABI's bit, which under `EI_ABIVERSION = 8` sits inside `EF_CUDA_SM_MASK` and so reads set on `sm_121a` and on plain `sm_90` alike. |
| no PTX | `cuobjdump -lptx` prints `No PTX file found to extract`, and the container walk finds 336 entries of kind 2 and none of kind 1. |
| `sm100/` is selected on GB10 | `sgl_kernel/load_utils.py::_load_architecture_specific_ops` takes `sm90/` only at compute capability 90. |
| FA3 is absent and unreachable | 80 wheel entries versus 81 on x86_64, both counts read from the zip central directory; `flash_ops.abi3.so` is the missing one, 578,045,190 bytes compressed and 1,758,623,416 uncompressed; `sgl-kernel/CMakeLists.txt:108-114` defaults `SGL_KERNEL_ENABLE_FA3` off on aarch64, which drops the whole target at `:390-476`; and `flash_attn.py:25-28` needs capability major 8 or 9. |
| every `sgl_kernel.flash_attn` reference | `grep -rn "sgl_kernel.flash_attn" python/sglang/` at the pin returns 8 lines, all eight tabulated above. None is on the GB10 dense text path. |
| the image's FlashInfer JIT cache is unverified | `docker/Dockerfile:21` is `ARG INSTALL_FLASHINFER_JIT_CACHE=0`, so `:329-349` is a no-op unless the build passed `1` and `:462` copies whatever `/flashinfer_jit_output/` holds. The wheel that would close it exists: `flashinfer_jit_cache-0.6.12+cu130-cp39-abi3-manylinux_2_28_aarch64.whl`, sha256 `097ee785b4886719761cda20d8363957d4b158e937c6dcde3ba84ea774065048`, 1,750,271,858 bytes by `Content-Length`. |
| identity | 3335 of 3335 common files byte-identical against the source tarball at the pin, 0 differing, 0 only-in-source; the one repository symlink resolves to matching content, giving 3336 of 3336. |
| no runtime commit assertion | `sglang/_version.py` sets `__commit_id__ = commit_id = None`. |
| no dependency wall | All 70 hard requirements have a prebuilt aarch64 or `py3-none-any` wheel for cp312. |
| upstream installs the same way | `docker/Dockerfile:210,242` at the pin. |
| the old driver is unrunnable | `scripts/dgx-sglang-low-concurrency.sh:5-7,10-11,55-57`. |

| Claim | How it was checked, 2026-08-23, in a lease on `dgx:gpu0` |
|---|---|
| both wheels installed, hashed IN the job | job `86282a1a`: `pip download` from PyPI, then `sha256sum` on the bytes that landed. `12716006`/`1c2d2602…` and `34243333`/`727e4bc5…`, both `WHEEL_SHA_OK=1`, `SGLANG_INSTALL_RC=0`, `KERNEL_INSTALL_RC=0`. |
| the identity of the installed tree | `IDENTITY_RC=0` from `cd /`: 3338 derived against 3338 in the manifest, 0 missing, 0 extra, 0 differing, at pin `f63458b5…`. |
| the identity gate detects what it claims | `tests/scripts/test_sglang_lease_identity.py`: five mutations each red, each asserting it applied, and the fixture green after each revert. |
| the FlashInfer JIT cache difference is CLOSED, not accepted | `JITCACHE_STATE=INSTALLED`, `flashinfer-jit-cache==0.6.12+cu130` from `https://flashinfer.ai/whl/cu130`. |
| the device the predicates read | `NVIDIA GB10`, capability `(12, 1)`, `torch 2.11.0+cu130`, `torch.version.cuda 13.0`; `is_sm100_supported=False`, `is_sm120_supported=True`, `is_flashinfer_available=True`. |
| clocks cannot be pinned | `LGC_RC=4`, "The current user does not have permission to change clocks", reproducing #1354 on a fourth job. |
| no compiler was needed | 199 resolved packages, all from wheels; the CUDA toolkit is installed for FlashInfer's run-time JIT and not for a build. |
| the server reached readiness on the gate model | job `b9e7709d`: `READY=1`, `SECONDS_TO_READY=454`, polled on `/health_generate`, which generates rather than answering a liveness probe. |
| the resolved backends, as the log names them | `attention_backend='flashinfer'`, `moe_runner_backend='auto'`, `mamba_backend='triton'`, `linear_attn_backend='triton'`, plus the server's own "Use flashinfer backend by default" and "Using hybrid linear attention backend for hybrid GDN models". |
| decode CUDA graphs are captured | "Capture target decode CUDA graph end. elapsed=30.54 s", `bs=[1,2,4,8,12,16,24,32]`. Prefill graphs are disabled by upstream's own multimodal rule. |
| one leg, zero errors, exact token count | timed c1: 6 of 6 completed, 0 failed, 6144 input and **768 output** tokens = 6 x 128, 173.23 s, `sglang-oai` on `/v1/completions`. |
| the warm JIT cache actually mattered | the discarded warmup and the timed c1 differ by 0.05% (4.431 against 4.433 output tok/s). An unwarmed first leg would not resemble the second. |
| the clock could not be pinned, and drifted | 214 samples over 441 s derived from the RAW file: median 2489 MHz, min 2346, max 2535, spread **7.59%** against a 5% ceiling, 84 C peak, SW thermal slowdown active in 4 of 214. |
| the host-memory floor this configuration leaves | 435 samples: `MemAvailable` 117,688 MB to a minimum of **14,935 MB**. The 5,000 MB watchdog never fired. |
| the resource came back | job `0f84b66d` at 23:10:48Z: `COMPUTE_APPS=0`, `SGENV_PROCS=0`, `/tmp/sgenv` gone. It also read the POST-reboot `boot_id`, a third job confirming the reboot. Source `rc logs 0f84b66d-1c30-4de5-bdb8-ee7b058f284a`, copied to `/mnt/nas_share/rc/sglang-w2/out/reap-20260823T231048Z/job.log` after the fact, because the reap script wrote no `$OUT`. |
| the c8 leg is VOID because the BOX REBOOTED DURING it, in its warmup | everything stops at 22:27:00Z with `sglang-c8.log` at 0 bytes and one request in flight, and no `TEARDOWN_VERDICT` or `DONE_MARKER` is printed; the next job on `dgx:gpu0` at 22:32:47Z reads `boot_id=26394f62…` against this row's `02d5a76f…`, with the kernel PID counter down from 3510 to 594. |
| the `MemAvailable` watchdog cannot see this collapse | the last eight samples read 17074, 17251, 17269, 17296, 17323, 17338, 14935 and 15449 MB against a 5,000 MB floor. The box went down with 15 GB available. |

## Stop conditions

- Stop if a record edit would state the static reading as a run result. Return
  `NEEDS_DECISION`. That distinction is the point of this row.
- Stop if a change would set `gateable = yes` before all five conditions in
  `## The exit criterion` hold.
- Stop if a change would edit the `pin` key or add a second oracle id.
- Stop if a change would edit an existing `../issue-index.md` row.
- Stop if the staging work needs a container image, `ssh` to a fleet device, or
  a file mutex taken outside a lease. Those are the walls this row exists to
  route around, not obstacles to push through.

## Owed

- [#1265](https://github.com/mudler/vllm.cpp/issues/1265) **stays open.** Its
  headline debt -- a demonstrated lease-compliant route -- is discharged by W3
  and W4, and this change deliberately does NOT close it. What it still owes is
  below: a clean teardown assertion, the c8 point, and every arm
  `docs/benchmarks/open-gaps.md` lists.
- ~~The teardown assertion of job `b9e7709d` was never printed.~~ **Closed, and
  here is the artifact.** The box rebooted before the job reached its own
  teardown, and
  [`../../scripts/rc-sglang-lease-reap.sh`](../../scripts/rc-sglang-lease-reap.sh)
  made the assertion separately in job `0f84b66d` at 23:10:48Z:
  `COMPUTE_APPS=0`, `SGENV_PROCS=0`, and `/tmp/sgenv` gone with the reboot.
  Re-derive it with `rc logs 0f84b66d-1c30-4de5-bdb8-ee7b058f284a`; a copy sits
  at `/mnt/nas_share/rc/sglang-w2/out/reap-20260823T231048Z/job.log`. The reap
  script as it ran archived nothing itself, which is why that copy is post-hoc;
  the script now tees to `$OUT` like `install` and `serve`.
- **The c8 leg is VOID, and it is not a concurrency-8 datapoint of any kind.**
  It had issued only its `warmup_requests=1` request when the box died, so the
  48-prompt 8-concurrency main loop never ran. `sglang-c8.log` is 0 bytes. Only
  c1 is a recorded leg.
- **A ratio.** The clock spread was 7.59% against a 5% ceiling, on a GB10 at
  84 C with software thermal slowdown active. Two clean absolutes are a result;
  a ratio is not available from this run and none is offered.
- **A MoE arm.** `moe_runner_backend='auto'` is what the log names on a DENSE
  model. Nothing here measures SGLang's MoE runner.
- **Why the box went down.** The reboot is established from the `boot_id` and
  PID-counter change; that this row's load CAUSED it is an attribution, not a
  proof. Nothing in the job's own output names a cause, and **the load at the
  moment of collapse was one in-flight request**, not the 8 the leg's name
  suggests -- so concurrency is not the identified variable and lowering it is
  not an identified fix. What differed between that request and the eleven
  minutes of identical single-request decode before it is the open question.
  Reproducing the c8 point needs a lower `--mem-fraction-static`, or an
  instrument that can see the collapse coming, or both; anything else is a
  guess dressed as a plan.
- **A watchdog that works on this box.** The `MemAvailable` floor in
  `scripts/rc-sglang-oracle-lease.sh` is now known not to be one: the machine
  rebooted with 15,449 MB available against a 5,000 MB floor. The floor is left
  in place because it costs nothing, and it is NOT protection.
- ~~[`sglang-wheel-in-lease.json`](sglang-wheel-in-lease.json) lands
  unreached.~~ **Closed by W2.** `scripts/sglang_lease_identity.py` reads it in
  the lease and `tests/scripts/test_sglang_lease_identity.py` reads it in CI,
  where it is the only executing code that opens the file.
- ~~A replacement driver for `scripts/dgx-sglang-low-concurrency.sh`.~~
  **Closed by W2**, as `scripts/rc-sglang-oracle-lease.sh`. The old script stays
  unrunnable and was not patched.
- The `flashinfer_cubin-0.6.12` architecture-token reading is inherited from an
  earlier investigation, was not re-derived in W1, and **was not re-derived in
  W2 either**. Nothing in `## W2, and what the lease measured on 2026-08-23`
  rests on it, and it stays fenced rather than being quoted as measured. The
  artifact is 447,533,460 bytes; re-deriving it needs its own job.
- A c1 pairing verdict for any SGLang-versus-ours ratio, which
  [#1354](https://github.com/mudler/vllm.cpp/issues/1354) can refuse for the
  same reason it refused every c1 pairing of the vLLM campaign.
- **The manifest count `3338` is gated only against itself**
  ([#1832](https://github.com/mudler/vllm.cpp/issues/1832)). It is quoted as
  measured in `../environment.md`, `../oracles/sglang.md` and
  `../sglang-matrix.md`, and it appears in no executing code:
  `grep -rn '3338' scripts/ tests/scripts/ .github/` is `rc=1`.
  `test_file_count_agrees_with_the_file_table` compares
  `manifest["file_count"]` with `len(manifest["files"])`, so dropping a real
  file and decrementing the header leaves the suite at `rc=0` on a manifest
  asserting a DIFFERENT tree. Since `__commit_id__` is `None`, the manifest is
  the only identity assertion this oracle has, so this makes `IDENTITY_RC=0` a
  tautology one level up. The repair is a re-derivation on a second independent
  install, not another checker reading the committed JSON. Raised by the fresh
  review of PR #1831, PRE-EXISTING from W1 (`727efb39c`), and deliberately not
  repaired in W2 because it needs a job on `dgx:gpu0`.
- **Neither registration of the identity suite is protected**
  ([#1833](https://github.com/mudler/vllm.cpp/issues/1833)). Deleting
  `test_sglang_lease_identity` from `scripts/agent-preflight.sh:176` leaves
  `check-test-registration.py` at `rc=0`, and so does deleting the whole
  `.github/workflows/ci.yml` step. The CONTROL is what makes this general:
  deleting the unrelated `test_tower_skip_rss_report` from the same array
  behaves identically, so the `SUITES` array is a list with no guard and the
  "registered in TWO places, deliberately" pattern buys nothing in either
  direction. Raised by the fresh review of PR #1831, not repaired in W2 because
  a population rule in `check-test-registration.py` is a semantic checker change
  owing its own row, spec and red-first evidence, and it will red on the twelve
  never-executed suites its own neighbour already lists. That neighbour is named
  in the filed issue rather than here, because a number written inside this
  section claims ownership of it and this row owns neither.

## Now

**The pinned SGLang oracle runs the gate model inside an `rc` lease, and
`gateable` is `yes` on that run.** Two jobs on `dgx:gpu0`, no `ssh`, no container
image, no file mutex outside the lease. The first installed both PyPI wheels,
hashed them from inside the job against the values this spec committed, installed
`flashinfer-jit-cache` so the first leg would not measure the JIT compiler, and
asserted the installed tree at `IDENTITY_RC=0` against a 3338-file manifest from
`cd /`. The second re-asserted that identity, served
`Qwen3_5ForConditionalGeneration` in 454 s with decode CUDA graphs captured,
named `flashinfer` as its resolved attention backend in its own log, and
completed a c1 leg with 6 of 6 requests, zero errors and exactly 768 output
tokens. The warm cache is measurable: the discarded warmup and the timed leg
differ by 0.05%.

**The box went down during the c8 leg, in its single warmup request.**
Everything stopped at 22:27:00Z, and the
next job on `dgx:gpu0` five minutes later read a different `boot_id` with the
kernel PID counter reset -- a reboot, which is the recorded GB10 unified-memory
signature. The 5,000 MB `MemAvailable` watchdog never fired and could not have:
the machine died with 15,449 MB available. c8 is VOID, the teardown assertion is
owed to a separate job, and the clock spread 7.59% against a 5% ceiling. No ratio
is available and none is offered. **Concurrency 8 was never issued**, so nothing
here is a concurrency result and why the box went down stays open.

#1265 stays open for the teardown assertion, the c8 point, the MoE arm, and every
floor arm `docs/benchmarks/open-gaps.md` still lists as unreached.

## Outcome

**What was measured.** The route works, end to end, by a permitted path. The
numbers that matter are not the throughput: they are `WHEEL_SHA_OK=1` twice,
`IDENTITY_RC=0` at 3338 of 3338 on two separate jobs, `READY=1` at 454 s, and
6 of 6 requests with 768 output tokens and zero errors.

**What was rejected, and why.**

- **Patching `scripts/dgx-sglang-low-concurrency.sh`.** Its missing half IS the
  forbidden path. A new driver was cheaper than a repair that would have had to
  delete the only thing the old script does.
- **Accepting the FlashInfer JIT-cache difference as a stated caveat.** W1
  marked it conditional and unverified. Installing the matching aarch64 wheel
  closed it instead, and the two c1 legs at 4.431 and 4.433 output tok/s are the
  evidence that closing it mattered.
- **`--backend sglang` (native `/generate`).** `sglang-oai` posts to
  `/v1/completions`, the path the 2026-07-28 image run drove and the path a vLLM
  arm uses. A different client path would not have been comparable to anything
  already recorded.
- **A second `docs/bench-evidence/` file for `evidence`.** It would be a copy of
  this record that nothing keeps in sync.
- **Re-deriving the `flashinfer_cubin-0.6.12` architecture tokens.** 447 MB for a
  claim nothing here rests on. It stays fenced.
- **A ratio against any vllm.cpp arm.** The clock rule refuses it.

**Why each default has its value.**

- `--mem-fraction-static 0.80` rather than the 0.85 of the recorded vLLM
  denominator: this run had to COMPLETE, and this box reboots rather than
  OOM-killing. **It was not conservative enough.** c1 was clean at a 14,935 MB
  `MemAvailable` floor and the box went down inside c8. The right conclusion is
  not "set a watchdog at 14,935": it is that a `MemAvailable` watchdog does not
  see this failure at all, because the machine rebooted with 15,449 MB
  available. Do not trust the counter, and do not read this as a concurrency
  result: concurrency 8 was never issued, and the box went down with one request
  in flight. Lowering the fraction is the lever this run supports. Lowering the
  concurrency is a guess, because the load at the moment of collapse was c1.
- Two `rc` submissions rather than one: a failed identity then costs no model
  time, and the serve phase re-asserts identity because the worker container is
  reused and `/tmp` can be gone.
- The serve script is staged under a SECOND filename. bash reads a script
  lazily, so overwriting the file a running job is executing corrupts that job.
- 6 prompts per concurrency point rather than the campaign's larger corpus: this
  is a reachability demonstration, not a floor, and a short leg that completes
  says more than a long one that a lease ends.
