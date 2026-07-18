# MoE expert load streaming — bound the 35B load-phase peak (whole-window PSS)

Row: `ENG-MOE-LOADSTREAM` (engine-matrix, KV cache and memory). Follow-up to
`ENG-MOE-HOSTFREE` (the STEADY host-free, `ac77bec`).
Owner claim: `CLAIM-MEM35-LOADSTREAM`.

## Scope

In: interleave the routed-MoE expert LOAD with the device Marlin build + host
free, per LAYER, so at most one layer's ~256 expert host copies coexist during
load — bounding the whole-window `peak_pss`/`peak_rss` (the binding memory axes)
from all N layers toward one. The device Marlin residents stay byte-identical;
only the host-copy LIFETIME changes.

Out (deliberately not in this row):
- The steady host-mirror free itself (`ENG-MOE-HOSTFREE`, DONE) — this row REUSES
  its per-layer `OwnedTensor::ReleaseHost` inside `BuildMoeMarlinResident`.
- The 22.92 GiB safetensors source steady mirror (`LOAD-SAFETENSORS` windowed
  release; separate).
- Inference-time disk expert paging / bank remap (`ENG-EXPERT-STREAM`, a distinct
  surpass-track capacity mode).
- Flipping `CudaPlatform::residency_policy().release_host_weights_after_upload`
  to `true` and routing residency through it — item 2 under `BACKEND-PLATFORM`
  (`CLAIM-BACKEND-PLATFORM-1`). This row realizes the LOAD-TIME streaming shape of
  that policy; the platform-flag wiring stays with its owning claim (see Item-2).

## Upstream chain

vLLM never holds a host fp4 mirror of the MoE experts: `base_loader.py:43-82`
constructs weights streamed onto the target device, and the Marlin MoE
`process_weights_after_loading` repacks in place on-device and drops the source
(`marlin_utils_fp4.py:375-434`), so its 35B peak PSS (13.3 GiB) reflects
device-resident weights with no host double-store. The residency capability maps
to `vllm/platforms/interface.py:134-229`
(`ResidencyPolicy.release_host_weights_after_upload`). Pin `e24d1b24`.

## Our baseline

- `LoadQwen3_5Moe` (`qwen3_5_weights.cpp`) loaded ALL layers' routed experts to
  host (`LoadMoe`→`LoadNvfp4Raw` `MakeOwned`+memcpy per expert), THEN a separate
  whole-model pass `Qwen3_5Model::PrepareMarlinResident` (`qwen3_5.cpp`) built the
  device residents and freed each layer's host copies (`ENG-MOE-HOSTFREE`).
- Peak = every layer's ~256 experts host-resident simultaneously at the Load→
  Prepare boundary (~19.8 GiB), so the binding `peak_pss`/`peak_rss` axes FAIL
  2/4 even with the steady free landed.
- Structural constraint: the mmap'd shards and the device queue live in DIFFERENT
  phases — Load has the shards but no queue; Prepare (from the runner) has the
  queue, and `LoadFromDir` freed the shards (`shards.clear()`) in between. A true
  interleave needs both together.

## Port map (deviations)

- `ModelSource::FromSafetensorsOwned(shared_ptr)` shares ownership of the mmap'd
  shards past the load; `LoadFromDir` uses it and no longer clears them
  (`model_registry.{h,cpp}`, `model_loader.cpp`). Deviation: vLLM has no such
  seam because it streams to device at load; our Load/Prepare phase split
  requires retaining the shards for the deferred materialization.
- `LoadQwen3_5Moe(shards, config, shards_owner)`: with an owner, load each layer
  WITHOUT its experts (`LoadLayerImpl(with_experts=false)`) and install
  `Qwen3_5MoeWeights::load_layer_experts` — a `mutable std::function` capturing
  the shared name-map + shards owner (keepalive) that materializes ONE layer's
  experts (`LoadMoeExpertsInto`) into a by-ref `MoeBlockWeights`. It does NOT
  capture the movable weights struct, so it survives the model's move into the
  `LoadedModel` (`qwen3_5_weights.{h,cpp}`).
- `PrepareMarlinResident` per-layer loop: when `load_layer_experts` is set, call
  it for layer `li` immediately BEFORE `BuildMoeMarlinResident` (whose existing
  `ReleaseHost` frees that layer's host bytes after the repack `Synchronize`);
  reset the closure after the last layer (returns the shards). Non-CUDA /
  `VT_NVFP4_MARLIN=0` / no-`VT_MARLIN_NVFP4` → `MaterializeAllDeferredExperts`
  bulk-loads to host (those forwards read the host bytes; not the production
  path) (`qwen3_5.cpp`).
- Granularity is per-LAYER because `MarlinNvfp4CombinedScaleFactor`
  (`qwen3_5.cpp:3665`) spans all E experts of a layer — all E must coexist for
  that layer's build, which is unchanged and byte-identical.

## Tests to port

No direct upstream test (vLLM has no host-mirror to stream/free). Local:
- CPU unit (`tests/vllm/test_qwen36_weights.cpp`, "deferred routed-expert load:
  move-safe closure + bounded coexistence"): the closure survives a
  `Qwen3_5MoeWeights` move, and driving the per-layer materialize→ReleaseHost loop
  keeps at most ONE layer's routed experts host-resident (peak == 1). Real-NVFP4/
  GPU-free.
- DGX end-to-end: the full model loads via the deferred disk path
  (`FromModelDir`); the token gate (315/315 + 235/235) proves byte-identity, the
  peak-PSS A/B proves the drop, `compute-sanitizer memcheck` proves no
  use-after-free of the freed host bytes.

## Dependencies

- `ENG-MOE-HOSTFREE` (`ac77bec`, DONE): reuses its `OwnedTensor::ReleaseHost`
  per-layer free inside `BuildMoeMarlinResident`.
- Shares the `qwen3_5.cpp` / `model_registry.cpp` files (disjoint functions) with
  `CLAIM-BACKEND-PLATFORM-1` and `CLAIM-MOE-DECODE-PARALLEL-1` — no line overlap.
- Item-2 (`CLAIM-BACKEND-PLATFORM-1`) owns the `residency_policy()` flag; this row
  can later gate the deferral on `CurrentPlatform().residency_policy()` without
  behavior change.

## Work breakdown

- W1 — deferred load plumbing: shared shards owner + loader defer + streaming
  closure (`model_registry`, `model_loader`, `qwen3_5_weights`). DONE (CPU-gated).
- W2 — per-layer interleave in `PrepareMarlinResident` + non-Marlin/CPU fallback.
  DONE (CPU-gated).
- W3 — CPU coexistence-bound contract test. DONE.
- W4 — DGX gates: token 27B 235/235 + 35B 315/315, load-to-ready peak PSS/RSS A/B
  (~19.8→~4-5 GiB, below vLLM 13.3), c2 smoke, memcheck. PENDING (orchestrator
  re-grids the binding memory axes to confirm the flip).

## Gates

1. CPU: clean `-Werror` build, full ctest, tools unittest 164, checkers.
2. Memory (the win): 35B load-to-ready PEAK PSS/RSS ~19.8 → ~4-5 GiB (below vLLM
   13.3); 27B unaffected (different loader).
3. DGX token-exact 27B 235/235 + 35B 315/315; c2 serving smoke clean; memcheck
   clean (no use-after-free of the freed host bytes).
4. Load-path coexistence bound test (the CPU contract above).

## Risks/decisions

- Byte-identity: the device build reads identical source bytes in the identical
  per-layer order — tokens must be bit-identical; the DGX 315/315 + 235/235 gate
  and memcheck are the confirmation. RISK: a use-after-free if a path re-reads the
  freed host bytes — mitigated by the `MaterializeAllDeferredExperts` fallback for
  every non-Marlin/CPU path and by memcheck.
- Shard lifetime: the deferred closure holds the last `shared_ptr` to the mmap'd
  shards and drops it when `PrepareMarlinResident` resets `load_layer_experts`.
  DECISION: keep the shards alive via the closure rather than reintroduce a
  Load-time device queue (which the Load/Prepare phase split does not provide).
- Granularity: per-LAYER (not per-expert) is the finest lifetime unit that keeps
  the combined-scale-factor build byte-identical. DECISION: accept the one-layer
  peak floor (~16.9/N GiB) — enough to clear vLLM's 13.3 GiB.
- Scope of paths: 27B (`LoadQwen3_5Dense`, true-W4A4) is a different loader →
  unaffected; GGUF/synthetic/borrowed pass no owner → eager (unchanged).
