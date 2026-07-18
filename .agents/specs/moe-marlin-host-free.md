# MoE Marlin resident — free the host fp4 expert double-store (35B peak-PSS lever)

Row: `ENG-MOE-HOSTFREE` (engine-matrix, KV cache and memory).
Owner claim: `CLAIM-MEM35-HOSTFREE`.

## Scope

In: after `BuildMoeMarlinResident` builds the device-resident Marlin repacked
routed experts (and `Synchronize`s), FREE the per-expert **host** fp4 byte
buffers (`expert_gate_fp4[e]`/`expert_up_fp4[e]`/`expert_down_fp4[e]` `.packed`
and `.scale` `OwnedTensor::bytes`) so the ~16.9 GiB host mirror of the routed
MoE experts is returned to the OS. This is the dominant contributor to the 35B
peak-PSS divergence (ours 21.2 GB vs vLLM 13.3 GB, 0.63× — see state.md
2026-07-18 binding).

Out (deliberately not in this row):
- The shared-expert / `lm_head` dense fp4 host bytes (small; `BuildMarlinDense*`
  residents) — retained, not freed here.
- The generic safetensors source-mmap host mirror (`LOAD-SAFETENSORS` windowed
  `madvise(DONTNEED)`; a distinct, already-partial residency reduction).
- Flipping `CudaPlatform::residency_policy().release_host_weights_after_upload`
  to `true` and routing ALL weight-residency through it — that is item 2 under
  `BACKEND-PLATFORM` (`CLAIM-BACKEND-PLATFORM-1`). This row REALIZES that policy
  behavior for the single largest host consumer; the platform-flag wiring stays
  with its owning claim. See "Item-2 link" below.

## Upstream chain

vLLM never keeps a host fp4 mirror of the MoE experts once they are on device:
`base_loader.py:43-82` constructs weights streamed onto the target device, and
the Marlin MoE `process_weights_after_loading` repacks in place on-device and
drops the source tensor. That is why vLLM's 35B peak PSS (13.3 GB) reflects the
device-resident weights with no host double-store. Our loader keeps a host
`OwnedTensor` per expert (`LoadNvfp4Raw` `MakeOwned`+memcpy,
`qwen3_5_weights.cpp:222-231`; retained in `LoadMoe:294-302`) for the CPU
reference / wmma-fallback path; on the production CUDA+Marlin path that host copy
is dead weight after the repack. The residency capability this maps to is
`vllm/platforms/interface.py` residency/memory-model (folded locally as
`ResidencyPolicy.release_host_weights_after_upload`).

## Our baseline

- `BuildMoeMarlinResident` (`src/vllm/model_executor/models/qwen3_5.cpp`) already
  frees the per-expert **device** transients (`d_packed`/`d_scale.reset()`) after
  the repack + `Synchronize`, but never the host `.bytes`.
- `OwnedTensor` (`include/vllm/model_executor/models/qwen3_5_weights.h`) holds the
  host bytes in a non-mutable `std::vector<uint8_t> bytes`; device residency
  (`d_dev`/`d_packed`) is `mutable` and populated lazily on a `const` weight.
- Gap: the ~16.9 GiB host expert mirror is never returned → 35B peak PSS 0.63×.

## Port map (deviations)

- `qwen3_5_weights.h`: add `void OwnedTensor::ReleaseHost() const;` — a logically
  const residency op (only the dead host buffer is reclaimed), mirroring the
  existing mutable lazy-device-upload design.
- `qwen3_5_weights.cpp`: implement `OwnedTensor::ReleaseHost()` via
  `std::vector<uint8_t>().swap(...)` (swap-with-empty forces capacity dealloc)
  through a narrow `const_cast`. **CRITICAL — page return:** the logical swap
  alone did NOT drop RSS/PSS (measured: 35B serving PSS stayed ~20 GiB). glibc
  raises its dynamic mmap threshold as large blocks are freed, so the ~0.5 MB
  per-expert allocations are served from the sbrk arena where `free()` only
  returns them to the free-list. `ReleaseHost` therefore `madvise(MADV_DONTNEED)`s
  the buffer's interior whole pages BEFORE the swap, dropping the resident
  anonymous pages immediately (dead bytes; a stray later read faults fresh zero
  pages) — the identical idiom as the `LOAD-SAFETENSORS` windowed release
  (`safetensors_reader.cpp:317`). With this, 35B serving PSS drops 20.17 → 3.53
  GiB.
- `qwen3_5.cpp` (`BuildMoeMarlinResident` host-free region only): after the
  `Synchronize`, gate `release_host = MarlinMoeEnabled()` and call `ReleaseHost()`
  on each routed expert's `.packed`/`.scale`.

**wmma-fallback GUARD.** The `VT_NVFP4_MARLIN=0` fallback (`MoeBlockFusedCuda`)
re-reads these host bytes via `ResidentNvfp4` on every first touch. Freeing is
safe only when the Marlin resident is the committed compute path.
`MarlinMoeEnabled()` is a process-static const evaluated once from the launch
env; `BuildMoeMarlinResident` is reached only from `MoeBlockFusedMarlinCuda`
(itself gated on `MarlinMoeEnabled()`) and from `PrepareMarlinResident` (same
gate), so when we are here the gate is TRUE by construction and the wmma path can
never be selected in this process. The explicit `release_host = MarlinMoeEnabled()`
re-check enforces that invariant and gates the free.

## Item-2 link (BACKEND-PLATFORM residency_policy)

This is exactly the `release_host_weights_after_upload` behavior that
`CudaPlatform` advertises today as `false` (`src/vllm/platforms/cuda.cpp:37`;
"Nothing branches on it yet"). Wiring the free THROUGH the platform flag (flip it
`true` on CUDA and gate on `CurrentPlatform().residency_policy()`) is item 2,
owned by `CLAIM-BACKEND-PLATFORM-1` (that row also owns the flag + the diagnostic
retention rationale). To respect ownership and stay minimal, this row does the
direct, correctness-gated free and records the link; item 2 can later replace the
`MarlinMoeEnabled()` gate with the platform-policy read without changing behavior.

## Tests to port

No direct upstream test (vLLM has no host-mirror to free). Local:
- CPU unit: `tests/vllm/test_qwen36_weights.cpp` — `OwnedTensor::ReleaseHost`
  frees the buffer AND its capacity, retains shape/dtype.
- DGX integration: `tests/vllm/test_qwen36_weights.cpp` — through the public
  `Qwen3_5Model::PrepareMarlinResident`, assert the routed expert host bytes are
  freed under the default Marlin gate and RETAINED under `VT_NVFP4_MARLIN=0`
  (keyed on the launch env because the runtime gate is process-static). Skips
  without the real 35B shard 1 / CUDA / `VT_MARLIN_NVFP4`.

## Measured result (steady vs load-transient peak)

Same-binary A/B via `VT_MOE_HOST_FREE` (server + grid sampler `sample_process_memory.py`):

| Metric (35B) | arm0 retain | arm1 free (default) |
|---|---|---|
| STEADY serving PSS (post-ready) | ~20 GiB | **3.53 GiB** |
| whole-window PEAK PSS (incl. load) | ~20 GiB | ~19.82 GiB |

The **steady serving footprint** drops ~16.6 GiB (matches the ~16.9 GiB expert
mirror; hits the 4–5 GB target; beats vLLM's 13.3 GiB). This is the "kept
resident forever" fix — the double-store is gone during serving.

The **whole-window peak is unchanged** because ALL routed-expert host copies are
allocated during `LoadQwen3_5Moe` and coexist until `PrepareMarlinResident`
(load-prepare) frees them, so the sampler — which starts at server spawn —
records the ~20 GiB load-phase coexistence. Moving the PEAK below vLLM requires
freeing per-expert DURING the load (upload+repack+free as each expert loads),
i.e. the direct-to-device streaming load (`ENG-EXPERT-STREAM` / the streaming
follow-on), which is out of this row's scope. This row is the necessary,
validated half: the serving resident is now minimal.

## Gates

1. CPU: clean `-Werror` build, full ctest, tools unittest, checkers.
2. Memory: 35B/27B STEADY serving PSS before vs after (A/B via `VT_MOE_HOST_FREE`).
3. DGX token-exact: `test_qwen36_paged_engine` 315/315 + `test_qwen27_paged_engine`
   235/235 (device resident is the compute path → token-neutral); serving smoke
   (c2) for no use-after-free; compute-sanitizer clean.
4. Load-path test asserting release + wmma-retention (above).

## Dependencies

- `BuildMoeMarlinResident` / `PrepareMarlinResident` (already present) — the device
  Marlin repack this hooks the free onto; no new dependency.
- `LOAD-SAFETENSORS` windowed release (`safetensors_reader.cpp:317`) — the page-return
  idiom (`madvise(MADV_DONTNEED)`) reused here; independent (releases the mmap source,
  not the heap copy).
- `BACKEND-PLATFORM` item 2 (`CLAIM-BACKEND-PLATFORM-1`) — owns the
  `residency_policy().release_host_weights_after_upload` platform flag this behavior
  realizes; this row does the direct, gated free and does NOT touch the flag.
- No dependency on the sibling MoE-memset / `cuda_moe.cu` / `cuda_marlin_repack.cu` work
  (edit region is ~70 lines away; staged explicit paths).

## Work breakdown

- W1 — `OwnedTensor::ReleaseHost` (madvise interior pages + swap-free) in
  `qwen3_5_weights.{h,cpp}`.
- W2 — gated free loop in the `BuildMoeMarlinResident` host-free region
  (`MarlinMoeEnabled()` guard + `VT_MOE_HOST_FREE` rollback), `qwen3_5.cpp:3743-3781`.
- W3 — tests: CPU `ReleaseHost` mechanism + DGX `PrepareMarlinResident`
  release/retention (`test_qwen36_weights.cpp`).
- W4 — DGX gates: memory A/B, token-exact 315/315+235/235, c2 smoke, memcheck.
- W5 — records (this spec, engine-matrix row, ledger, state, README, BENCHMARKS,
  coordination, backend-matrix item-2 note).
- Follow-on (NOT in this row) — load-time streaming interleave to move the
  whole-window peak (`ENG-EXPERT-STREAM`).

## Risks/decisions

- DECISION: direct free gated on `MarlinMoeEnabled()` rather than flipping the
  `CudaPlatform` residency flag — respects `CLAIM-BACKEND-PLATFORM-1` ownership and
  stays minimal; item-2 link recorded. RISK if a future non-Marlin CUDA path is added
  that also reaches `BuildMoeMarlinResident`: it would need the host bytes → the
  `MarlinMoeEnabled()` guard already fails closed (frees only on the committed Marlin
  path).
- RISK: `madvise(MADV_DONTNEED)` on heap pages zeroes them; a stray later read of a
  released host weight returns zeros (not a fault). MITIGATION: nothing reads a routed
  expert host weight after the Marlin resident is built (device resident is the compute
  path); the wmma fallback that would re-read is guarded out; c2 smoke + memcheck +
  token-exact gates confirm no runtime read.
- DECISION/RISK: the post-build free reduces STEADY serving PSS (the "resident forever"
  mirror) but NOT the whole-window load-phase peak; accepted as scoped, peak deferred to
  the streaming follow-on. Recorded honestly in every surface.
