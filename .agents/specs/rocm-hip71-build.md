# Spec: ROCm build on HIP 7.1, gated on gfx1102 and gfx1103

Row: `BACKEND-ROCM`
Issue: `ISSUE-LOCAL-01M2T6Q2GSRRV0G6PTFJSJNSP4`
State: DONE (2026-09-18)
Git integration: one PR for spec and implementation (repository default; no
preference recorded).
Base: `origin/main` at `984f7265c` (#2787).

## Scope

`-DVLLM_CPP_HIP=ON` does not compile on HIP 7.1.52802 (Fedora 44, ROCm clang
20). Two sites fail for every target architecture:

- `src/vt/rocm/rocm_skinny_gemm.hip`: `on_gfx1151()` reads
  `hipDeviceAttributeGcnArch`, which the HIP 7.1 headers no longer declare,
  and calls `hipDeviceGetAttribute` from a `__device__` function. Since
  #2787 replaced the tile selection with `kYtile`/`kUnrl`, its only caller
  `SelectYtileUnrl` has no caller either, and `-Werror=unused-function`
  rejects it.
- `src/vt/rocm/rocm_paged_attn.hip`: `PagedAttnDecodeSplitKvReduce` (#845)
  declares `g` and `m_split` and never reads them; `-Werror` rejects both.

Out of scope: the reduce kernel's per-split max tracking that its own comment
names as missing (the kernel's numerics are unchanged), allocator policy for
`PageableMemoryAccess = 0` boards (#2511 stands), dual-GPU execution (#480),
and any throughput claim.

## Attribution (measured 2026-09-18)

- Red: `cmake --build build-hip` on `58c986eb9` stops at
  `error: use of undeclared identifier 'hipDeviceAttributeGcnArch'` and at
  `error: unused variable 'm_split' [-Werror,-Wunused-variable]`. On
  `984f7265c` a host-side rewrite of `on_gfx1151()` still stops at
  `error: unused function 'SelectYtileUnrl' [-Werror,-Wunused-function]`.
- Green: after the change the Release build for
  `VLLM_CPP_HIP_ARCHITECTURES="gfx1102;gfx1103"` completes and
  `ctest -R 'rocm|cross_device'` passes 31/31 with `HIP_VISIBLE_DEVICES=0`
  (RX 7700S) and with `HIP_VISIBLE_DEVICES=1` (Radeon 780M).

## Design

`on_gfx1151()`, `YtileUnrl`, and `SelectYtileUnrl` are removed. The include
list is unchanged: `<string>` stays for `std::to_string`. Nothing calls the
three after #2787, and AGENTS.md forbids keeping dead code for a possible later caller. A future
gfx1151 tile table can port vLLM's `skinny_gemms.cu:1240-1258` again from git
history (`git log -S'SelectYtileUnrl' -- src/vt/rocm/rocm_skinny_gemm.hip`)
with a host-side `gcnArchName` probe, since the device-side attribute is gone
from HIP 7.1. The reduce kernel drops the two dead locals and leaves
`num_kv_heads` unnamed, because the head-group index it computed was never
consumed.

Rejected: keeping `SelectYtileUnrl` alive with a host-side `on_gfx1151()`
(the first version of this row). It compiled on `58c986eb9` but is unreachable
on `984f7265c`, so it would only carry a tuning table no dispatch reads.

## Tests

- The existing ROCm gate covers both files:
  `ctest --test-dir build-hip -R 'rocm|cross_device'` (31 tests on 984f7265c).
- No new unit test: the change deletes unreachable code and dead locals, so
  no behaviour exists for a test to pin. The red-first test is the HIP 7.1
  compile itself: on `984f7265c` the two objects fail with
  `use of undeclared identifier 'hipDeviceAttributeGcnArch'` and
  `unused variable 'g'` / `'m_split'` (`-Werror`).
- Fresh mutation review (2026-09-18, a session that did not author the edit):
  each file restored from `origin/main` in turn and its object rebuilt. Both
  mutations go red with the errors above; the head goes green, the full build
  completes, and `ctest -R 'rocm|cross_device'` passes 31/31 on both GPUs.
  `PagedAttnDecodeSplitKvReduce` is still launched from
  `src/vt/rocm/rocm_paged_attn.hip:2912-2918`, so the reduce kernel stays
  reached from the ROCm attention path.
- Upstream review of #3220 found that the same kernel computes `m_global` and
  never rescales the partials by it (`rocm_paged_attn.hip:1318-1338`). That is
  pre-existing since #845 and out of scope here; it is tracked as
  `ISSUE-LOCAL-01M31EXFQ7NCN3EA6290SH8K2N`.

## Gates

- `scripts/agent-preflight.sh`: all record gates `ok`; `role-undeclared` and
  `tools suites` fail, the latter in `test_drop_file_cache` identically on a
  clean `origin/main` on the same host.
- `scripts/check-commit-style.py --range upstream/main..HEAD`: OK.
- `scripts/check-agent-record.py`: OK.
- Fork CI on KhazAkar/vllm.cpp#1: 16 jobs green; `agent-record` fails only
  because the fork carries no `row/*` refs.

## Outcome

Measured on a Framework 16 (RX 7700S gfx1102, 8 GiB; Radeon 780M gfx1103,
Integrated=1, 27.4 GiB shared; HIP 7.1.52802; kernel 7.2.5-200.fc44):

- Both GPUs report `ManagedMemory=1 ConcurrentManagedAccess=1
  PageableMemoryAccess=0`, so the default allocator is plain `hipMalloc` on
  both, as `docs/ROCM.md` states.
- Qwen3-0.6B greedy (`--temperature 0`, 64 tokens) matches the CPU tokens on
  both GPUs with zero reference-tier hits: 61 tok/s on the 7700S, 34 tok/s
  on the 780M. These are allocator A/B readings, not benchmark entries.
- `VT_ROCM_MANAGED_ALLOC=1` on the 780M: 7 of 7 legs exit 0 with matching
  tokens and no `page fault` line in `journalctl -k`, unlike gfx1151 in
  #2511, but decode falls to 12-21 tok/s. Rejected as a default: slower and
  without a correctness gain. On the 7700S the knob changes nothing
  measurable (60.4-61.2 vs 61.0-61.6 tok/s, 3 legs each).
- `docs/ROCM.md` gains the two measured hardware rows.
