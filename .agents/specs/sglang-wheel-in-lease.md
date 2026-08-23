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

**`gateable` stays `no`, and it stays `no` until a job serves a model and
completes a leg.** W2 met the first two conditions and did not reach the rest;
the verdict table is at the end of this section.

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
| 1 | an `rc` job on `dgx:gpu0` installed both wheels and reported their sha256 from inside the job | **MET.** Job `86282a1a`, both `WHEEL_SHA_OK=1`. |
| 2 | `IDENTITY_RC=0` against the committed manifest, asserted from `cd /` | **MET.** 3338 of 3338, 0 missing, 0 extra, 0 differing. |
| 3 | the server reached readiness and the log names the resolved attention backend and MoE runner | **UNMET.** Job `b9e7709d` is queued. |
| 4 | one leg completed with zero errors and the recorded output-token count | **UNMET.** Same job. |
| 5 | `evidence` moved off `#1265` to a path that exists in this tree | **NOT TAKEN.** It would be false while 3 and 4 are unmet. |

Two of five. `gateable` stays `no` and #1265 stays open.

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

**Not yet reported here.** The serving job is
`b9e7709d-cc96-4247-9d01-c611bce707ac`, submitted to `dgx:gpu0` at
2026-08-23T21:07Z and queued behind two other sessions' work, which is the lease
working rather than failing. Queueing is the correct behaviour and the reason
this section says nothing rather than something provisional.

Until it reports, exit conditions 3, 4 and 5 are UNMET and `gateable` stays
`no`. `AGENTS.md` requires all five in one recorded run, and a partial result is
recorded as a partial result.

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
| W4 | this row | One `rc` job that serves the gate model and completes a leg, with the resolved backends read from the server log. **Submitted 2026-08-23**, job `b9e7709d`, queued behind other sessions. |
| W5 | this row | The `gateable` flip, if and only if all five conditions in `## The exit criterion` hold. **Not taken.** Conditions 3, 4 and 5 are unmet, so `gateable` stays `no`. |

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

- [#1265](https://github.com/mudler/vllm.cpp/issues/1265) stays open. It owes a
  demonstrated lease-compliant route, which is W3 and W4.
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

## Now

The route is specified, the driver is written, and the pin is reachable inside a
lease. One `rc` job on `dgx:gpu0` installed both PyPI wheels, hashed them from
inside the job against the values this spec committed, and asserted the
installed tree against the 3338-file manifest at `IDENTITY_RC=0` from `cd /`.
`flashinfer-jit-cache==0.6.12+cu130` is installed, so the warm-cache difference
is closed rather than accepted. No compiler ran.

**No model has been served yet.** That job is submitted and queued behind other
sessions' work on the same device. `gateable` stays `no`, #1265 stays open, and
the next step is the serving leg and the resolved backends read from the server
log.
