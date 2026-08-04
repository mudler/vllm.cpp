# Hardening adoption review — external C++/Rust porting study, 2026-07-27

Row: `HARDEN-DETECTOR-LANES` (engine matrix, cross-cutting quality).
Scope: read an external, independently written hardening guide distilled from
defects found in two unrelated tensor-engine ports, decide item by item what
transfers to vllm.cpp, and land the transferable prevention.

Source (read-only, outside this repository):
`../float-harness/public-export/float-porting-analysis/` — `HARDENING-GUIDE.md`
(the prevention rules), `LANGUAGE-ANALYSIS.md` §6 confirmed action items and §7
auditor blind-spot guide, `REPORT.md` (the experiment the defects came from).
The study's subject engines are audio-driven portrait models, not inference
servers, so the transfer is by DEFECT CLASS, never by copying a fix.

## Verdict per guide section

| § | Guide rule | State here | Action |
|---|---|---|---|
| 1 | Validate once, preserve the proof in types | Partial. `vt::Tensor` carries dtype/shape/device and `VT_CHECK` validates at op entry, but token/slot/block counts are bare `int32_t`/`int64_t` throughout the runner. | NO CODE CHANGE. A units/newtype pass over the runner is a large refactor with a real regression surface and no observed defect behind it. Recorded as a standing preference for new interfaces, not a retrofit. |
| 2 | Model metadata is a schema, validated at load | ALREADY STRONGER. The safetensors/GGUF loaders validate the whole manifest (dtype, rank, extents, byte size) and the model gates are token-exact against the vLLM oracle. | None. |
| 3 | Encode pipeline state and policy | Partial by construction: the engine mirrors vLLM's staged pipeline, and policies (quantization scheme, attention backend, cache mode) are already closed enums resolved once. | None. |
| 4 | Confine unsafe memory; scoped arenas | `DevicePool` is a per-block cache, not a suballocating arena, so the guide's worst case (many logical tensors inside ONE allocation) does not exist. Two weaker forms of the same blind spot DO: size-class rounding hands back a block up to 6.25% larger than the tensor, and blocks are never returned to the driver. | **LANDED** — `VT_POOL_BYPASS=1` (`include/vllm/model_executor/models/device_pool.h`). |
| 5 | Harden parsers and FFI boundaries | ALREADY STRONGER on parsers (bounded readers, fuzz-covered GGUF/safetensors). The C ABI checks handles and translates failures. | None. |
| 6 | Separate semantic correctness from memory safety | ALREADY THE HOUSE RULE: every kernel keeps a CPU reference and the parity tiers are differential tests against it, plus the vLLM oracle above that. Fixed reduction order is a live concern, not a gap: the recorded 119/128 benchmark-token identity under Triton AOT is exactly the completion-order effect the guide warns about, and it is already attributed to reduction order with operator coverage supplying the equivalence proof. | None. |
| 7 | Make state ownership and concurrency visible | Partial. Profiling counters are per-object, and the aux-stream pool split (`AuxPool()`) is precisely the guide's "hand out disjoint capabilities" rule applied to CUDA streams. Untested: concurrent independent engines, callback re-entry. | Covered by the TSan lane below; a dedicated concurrency matrix is left as a follow-up row. |
| 8 | Make the build observable; use a build MATRIX | **THE REAL GAP.** Build observability is already strong (the bench harness records commit, dirty state, CMake cache, `ldd`, dataset/binary SHA-256, GPU/thermal state). But the test matrix was ONE default build: no ASan, no UBSan, no TSan, anywhere in CI. | **LANDED** — `VLLM_CPP_SANITIZE` + the `sanitize-cpu` CI matrix. |
| 9 | Build tests around a defect matrix | ALREADY STRONGER in the behavioural tier (parity taps per stage, ported upstream tests as the spec). Weakest in the DYNAMIC tier, which is the same gap as §8. | Same action as §8. |
| 10 | Treat performance and process defects as defects | ALREADY STRONGER than the guide asks: `.agents/benchmark-protocol.md` requires an idle box, one exclusion lock per series, >=3 repetitions, same-binary A/B, a recorded repro recipe, and it VOIDS contended or non-reproducing numbers. | None. |

## What was landed

### 1. Dynamic detector lanes (§8, §9 — the highest-priority gap)

`VLLM_CPP_SANITIZE` selects a HOST instrumentation lane
(`address`, `undefined`, `address,undefined`, `thread`), and CI job
`sanitize-cpu` runs the full CPU suite under ASan+UBSan and under TSan as two
separate jobs. Design points, all from the guide:

- ASan/UBSan and TSan are separate builds; their runtimes are mutually exclusive.
- The lane REFUSES to configure with the CUDA backend on: the host runtime does
  not instrument nvcc device TUs and false-positives against the CUDA driver.
  The CUDA tier's equivalent detector is `compute-sanitizer`.
- `-fno-sanitize-recover=all`, so a finding ABORTS and ctest fails the lane. The
  guide's specific warning is that a print-and-continue sanitizer produces a
  green run containing a real report.
- The lane keeps `-ffp-contract=off`, so a bit-identity assertion still means in
  the lane what it means in production.
- `-Werror` is dropped IN THIS LANE ONLY, in `cmake/CompilerWarnings.cmake`
  where the per-target warning flags are set (a global `-Wno-error` is overridden
  by those PRIVATE target options and does nothing). Sanitizer instrumentation
  changes inlining enough that GCC's range and initialization analyses fire
  inside LIBSTDC++ on correct code: a one-element
  `std::vector<int32_t> v = {x}` in `voxtral.cpp` draws "forming offset 4 is out
  of the bounds [0, 4]" for a 4-byte read of a 4-byte array, and `<regex>` draws
  26 `-Wmaybe-uninitialized` reports from `std::function` internals. None is
  project code, and the plain build — the one that DOES enforce `-Werror` — stays
  clean. Letting a false positive in a system header stop the runtime detectors
  would defeat the point.
- `continue-on-error: true` remains until the repaired PR validates the two
  lanes on the hosted runner. The first complete local survey is now triaged:
  both full suites pass, and removing the flag after remote confirmation is the
  closing step of this row.

**The lane paid for itself before it was even committed.** Its first real build
failed on `-Werror=unused-function`: `AsyncDeviceMirrorEnvDefault()`, added by the
W4 work in the same session, is referenced only inside `#ifdef VLLM_CPP_CUDA` and
is therefore dead on a CPU build. That would have broken the EXISTING
`build-test-cpu` CI job too — the sanitizer lane just happened to compile the
CPU configuration first. Fixed by guarding the definition with the same `#ifdef`
as its use. This is precisely the guide's §8 point: a single default build never
compiles the other `cfg` branches.

**Verified end to end**, not merely configured: `address,undefined` and `thread`
both configure; the two guards fire (a lane value outside the allowlist, and a
lane with the CUDA backend on, are both `FATAL_ERROR`). On GCC 15.2.0 the full
ASan+UBSan suite passes 331/331 under
`ASAN_OPTIONS=detect_leaks=1:strict_string_checks=1`,
`UBSAN_OPTIONS=print_stacktrace=1`, and `VT_POOL_BYPASS=1`; the full TSan suite
also passes 331/331 with the bypass. The detector-specific shared test image
reduces the ASan+UBSan build from 93 GiB to 5.6 GiB, while keeping file/line
traces with `-g1`; TSan is 1.9 GiB. The survey also removed the minja macro
context ownership cycle that leak detection exposed. The plain GCC 15 `-Werror`
build and the affected `test_serve_low_tools`, `test_none_hash_determinism`, and
`test_chat_template` cases pass 3/3.

### 2. Allocator-blind-spot bypass (§4)

`VT_POOL_BYPASS=1` makes every `DevicePool` Get an exact-size driver allocation
and every Put a real Free. Default OFF and production-inert. It exists because a
caching, size-class-rounding, never-returning pool defeats the detector twice:
an overrun past the last tensor row stays inside the block, and a use-after-free
reads memory that is still mapped (and may already belong to another op). This
is the guide's single highest-priority blind spot ("the most likely place a real
bug is hiding undetected"), in the milder form our design permits. It is a
DEBUGGING lane, never a timing configuration — it reinstates the per-op
cudaMalloc/cudaFree device-sync storm the pool exists to remove.

Intended use: `VT_POOL_BYPASS=1 compute-sanitizer --tool memcheck <gate test>`.

## Surveyed and found already prevented

The guide's greps were run against this tree; two classes it flags do not
reproduce here, which is worth recording so a later reader does not re-audit:

- **Fixed-size stack scratch sized independently of the input shape** (the
  study's `srow[4096]` stack smash): no instances in the host layer. Scratch is
  pooled device memory sized from the actual shape.
- **Unvalidated `atoi` on environment variables**: 8 sites, and 5 are already
  guarded downstream by an allowlist (`VT_GDN_DECODE_NW` accepts only 1/2/4/8),
  a range check (`VT_GDN_OCC_BLOCK` 32..1024), or a clamp
  (`VLLM_CPP_CPU_THREADS` -> `std::clamp(n, 1, kMaxThreads)`). Three are
  unvalidated and remain open, all low severity because none is a memory-safety
  input: `VT_MOE_SHARED_AUX_THRESHOLD` (`qwen3_5.cpp`, garbage -> 0 silently
  changes an overlap threshold), `VT_VULKAN_DEVICE` (`vulkan_context.cpp`,
  garbage -> 0 silently forces device 0 instead of failing), and the LMCache
  remote port (`remote_client.cpp`, garbage -> port 0). Tracked as follow-up.

## Not adopted, with reasons

- **Rust/Miri items** — no Rust in this project.
- **`panic = "abort"` / `catch_unwind` boundary policy** — Rust-specific; our C
  ABI's exception policy is already explicit.
- **FIL-C lane** — its separate ABI and rebuild cost do not fit a CUDA project
  whose hot path is device code the tool cannot see. ASan/UBSan plus
  compute-sanitizer cover the same classes for us.
- **Units/newtype retrofit (§1)** — see the table; deliberate, not an oversight.

## Follow-up rows

1. Confirm both repaired `sanitize-cpu` jobs on the hosted PR runner, then
   remove `continue-on-error` (closes this row).
2. Validate the three residual `atoi` sites.
3. A concurrency matrix per §7 (independent concurrent engines, callback
   re-entry, forced worker failure).
4. Run a gate test under `VT_POOL_BYPASS=1` + `compute-sanitizer` and record the
   result; that is the evidence the bypass lane is worth keeping.
