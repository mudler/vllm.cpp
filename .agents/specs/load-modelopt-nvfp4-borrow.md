# LOAD-MODELOPT-NVFP4-BORROW — the ModelOpt NVFP4 arm never tried to borrow, so 21 GiB of packed weights landed as anonymous heap

Issue: [#1647](https://github.com/mudler/vllm.cpp/issues/1647).
Owning row: `ENG-LOAD-DIRECT-UPLOAD` ([engine-matrix.md](../engine-matrix.md)),
spec [load-direct-upload.md](load-direct-upload.md).

## Scope

Two edits in `src/vllm/model_executor/models/qwen3_5_dense_weights.cpp`:

1. `LoadNvfp4AnyNaming`'s **ModelOpt** branch gets the borrow-then-fallback
   shape its compressed-tensors sibling `LoadCtNvfp4Raw` has carried since
   `ENG-LOAD-DIRECT-UPLOAD`.
2. `DirectDeviceLoadEligible` stops asking `!platform.is_unified_memory()`,
   a predicate `include/vllm/platforms/interface.h:294-298` documents as
   answering the opposite question.

Out of scope, and recorded under `## Owed`: rewriting the loader to upstream's
allocate-device-first polarity, and everything else #1647 owes.

## The bug

`r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121` (21 GiB on disk, `Qwen3_5ForConditionalGeneration`,
ModelOpt `MIXED_PRECISION`: 193 `W4A16_NVFP4` group-16 + 208 static per-tensor
FP8) exhausts a GB10 box (`dgx.casa`, 119 GiB **unified** memory — a device
allocation IS host RAM) and takes the machine's system services with it. It has
done so five times. Measured on the committed tree at `27d8bfa70`, 2026-08-23:

| Run | Result |
|---|---|
| `VT_LOAD_STATS=1 --device cpu` | `host_copy=16.394 GiB  borrowed=2.373 GiB  device_upload=0.000 GiB` |
| `--device cuda --kv-cache-memory 512MiB --max-num-seqs 1 --max-model-len 1024` | loads, 85 GiB available at load-end |
| `--device cuda --kv-cache-memory 8GiB --max-num-seqs 4 --max-model-len 4096 --kv-cache-dtype fp8` | `avail 37 GiB -> 28 GiB in ONE second`, killed at a 28 GiB floor; unguarded it reaches 0 and the box reboots with NVRM `NV_ERR_NO_MEMORY ... _memdescAllocInternal` |
| vLLM, same checkpoint, same box, `--gpu-memory-utilization 0.70` | steady at 71-83 GiB available, serves fine |

The first row **falsifies the obvious hypothesis**: the weights reach the host
PACKED, not dequantized to bf16. Nothing in this row chases a bf16
materialization. What the first row *does* say is that 16.394 GiB arrived as a
**host copy** where 2.373 GiB was borrowed, and a host copy of a safetensors
range is anonymous heap where upstream's is reclaimable page cache.

## Upstream, read at the pin `5559679229bc961848b121ccdeaa8fa5d79bec98`

vLLM allocates the **final packed quantized parameter on the device before it
opens any file**, then streams the checkpoint into those pre-existing buffers
one tensor at a time.

| Anchor | What it says |
|---|---|
| `vllm/model_executor/model_loader/base_loader.py:52-58` | `target_device = torch.device(load_device)`, then `with target_device:` around `initialize_model(...)`. Every parameter a quant method registers is allocated **inside** that context, before `self.load_weights(...)` at `:64`. |
| `vllm/model_executor/layers/quantization/modelopt.py:1313-1345` | The NVFP4 method registers `weight` as `torch.empty(out, in // 2, dtype=torch.uint8)`, `weight_scale_2` as f32, `weight_scale` as `torch.empty(out, in // group_size, dtype=torch.float8_e4m3fn)`. **No `device=` on any of them**, so all three inherit the target device, at their PACKED sizes. |
| `vllm/model_executor/layers/quantization/modelopt.py:483-491` | The FP8 half does the same: `torch.empty(out, in, dtype=torch.float8_e4m3fn)`, no `device=`. |
| `vllm/model_executor/model_loader/weight_utils.py:969-974` | `with safe_open(st_file, framework="pt") as f: for name in f.keys(): param = f.get_tensor(name); yield name, param` — a generator over a **file-backed mmap**, one tensor at a time. |
| `vllm/model_executor/model_loader/weight_utils.py:1247` | `param.data.copy_(loaded_weight)` — the yielded value is consumed immediately and dropped. |

So upstream's transient host residency is **reclaimable page cache** whose
lifetime is one loop iteration. It never holds an anonymous copy of the model.

Upstream also handles this hardware explicitly, which is why the oracle survives
on the same box:

| Anchor | What it says |
|---|---|
| `vllm/utils/mem_utils.py:148-155` | `if current_platform.is_integrated_gpu(device.index): self.free_memory = psutil.virtual_memory().available` — because `cudaMemGetInfo` underreports free memory on UMA. The comment names DGX Spark. |
| `vllm/utils/mem_utils.py:54-83` | `release_device_memory_under_pressure` empties the caching allocator once `psutil` available drops below 20% of total. |

This tree has neither. `--kv-cache-memory` bounds only the KV pool, and
[#83](https://github.com/mudler/vllm.cpp/issues/83) records that
`--gpu-memory-utilization` is accepted and ignored, so nothing bounds the rest.
That is why the oracle degrades on this box where we take it down.

## Mechanism, derived from this tree before the fix was written

`LoadCtNvfp4Raw` (`qwen3_5_dense_weights.cpp:197-215`) is the shape the row
already ships:

```cpp
  if (!BorrowStTensorBytes(r.packed, packed, vt::DType::kI8, {out_dim, in_dim / 2})) {
    r.packed = MakeOwned(vt::DType::kI8, {out_dim, in_dim / 2});
    VT_CHECK(packed.nbytes == r.packed.bytes.size(), "...");
    std::memcpy(r.packed.bytes.data(), packed.data, packed.nbytes);
    MaybeReleaseSourcePages(packed.data, packed.nbytes);
  }
```

`LoadNvfp4AnyNaming`'s ModelOpt branch (`:398-407`) goes straight to
`MakeOwned` + `std::memcpy` for **both** `r.packed` and `r.scale`, with no
borrow attempt at all. Our checkpoint declares `quant_method: "modelopt"`, so
`IsNvfp4Projection` (`:356`) sends its 193 NVFP4 modules — the whole 21 GiB —
down that branch. Two consequences follow, and the second is the expensive one:

1. ~21 GiB of **anonymous** heap where upstream has page cache. On unified
   memory that competes with the device allocation for the same 119 GiB.
2. `AdoptDeviceBytesAsHost` (`qwen3_5_weights.cpp:311-320`) becomes a **silent
   no-op**, because its whole body is gated on
   `w.mmap_src != nullptr && w.bytes.borrowed()`. `ResidentNvfp4`
   (`qwen3_5.cpp:1373-1400`) calls it after publishing `d_dev` for each of
   `packed` and `scale`, expecting exactly the ENG-LOAD-DIRECT-UPLOAD contract:
   release the consumed source pages, and where device memory is
   host-addressable, re-point `bytes` at the device allocation so the process
   stops holding a second copy of the model. A non-borrowed weight gets neither.
   `ResidentNvfp4` has **no aliasing branch** — unlike `ResidentWeight`
   (`qwen3_5.cpp:1240-1274`), it always `Alloc`s and `Copy`s — so the device
   copy is genuinely a second allocation, and after the fix the adoption is what
   collapses the pair back to one.

That is the double residency: packed host copy plus packed device copy, ~42 GiB
of a 119 GiB pool before a single KV byte is reserved, which is the right order
for `avail 37 GiB -> 28 GiB in ONE second`.

## The second predicate, and which of the two is right

`DirectDeviceLoadEligible` (`qwen3_5_dense_weights.cpp:126-138`) ends:

```cpp
  return !platform.is_unified_memory() &&
         platform.residency_policy().release_host_weights_after_upload;
```

`is_unified_memory()` is true on GB10, so `StageAndReleaseLoadedDense` — the
only caller of `ReleaseResidentQwen3_5DenseHostWeights` — never runs there.

Two predicates disagree about one device, and **`needs_weight_staging()` is the
right one**. `src/vllm/platforms/cuda.cpp:85` returns `true` unconditionally on
CUDA and its own comment says why: "the CUDA path stages host tensors into
distinct device-resident buffers ... **regardless of GB10 being physically
unified**". `include/vllm/platforms/interface.h:294-298` says the same from the
other side: `is_unified_memory()` "answers the OPPOSITE question: CUDA on GB10
reports unified because host and device address one physical pool, while a
`cudaMalloc` pointer is still not host-dereferenceable". `ResidentNvfp4` is the
proof by construction — it allocates and copies with no aliasing branch, so on
GB10 there really are two buffers, and a predicate that says otherwise is
describing physics rather than this allocator. `needs_weight_staging()` is
already the first thing `DirectDeviceLoadEligible` asks, so the correct body is
the residency policy alone.

**Say plainly what this second edit does and does not buy.** It does *not* fix
the reported OOM. `StageAndReleaseLoadedDense` is reached only while
`IsPlainBf16Qwen3_5Dense(w)` holds (`:938`), which is false for this checkpoint
at its first NVFP4 projection, and
`ReleaseResidentQwen3_5DenseHostWeights` releases only `OwnedTensor`s — it never
touches an `Nvfp4Weight`'s `packed`/`scale`. Fix 1 is the one that addresses the OOM. Fix 2
reconciles a predicate that is wrong on its own terms, and its measurable effect
is on a **plain bf16 dense checkpoint on unified memory**: those weights are now
staged and their redundant host mirrors released, where before the whole lane
was switched off. Landing it with the correct scope stated is better than
leaving a predicate in the tree that contradicts the two comments beside it.

**It is not a use-after-free.** The long comment at
`qwen3_5_dense_weights.cpp:1039-1060` records that this exact `!is_unified_memory()`
test is what kept an aliasing use-after-free unreached, and records the repair:
the release site asks the invariant `HostMirrorIsRedundant(tensor)` —
`w.d_dev != nullptr` (`qwen3_5_weights.h:445-447`) — instead of the
`d_dev || d_dev_f32` proxy. On the aliasing arm `ResidentWeight` leaves `d_dev`
null (`qwen3_5.cpp:1258-1272`), so an aliased weight is never released. The
comment's own conclusion is that the disjunct is redundant on the staging arm;
this row removes the accident it was relying on and leaves the invariant doing
the work.

## Design

Fix 1 — give the ModelOpt branch the identical borrow-then-fallback shape,
preserving both byte-size `VT_CHECK`s and both fallbacks exactly. This is a
**widening of an already-gated path to a second checkpoint naming**, not a new
policy: the same `BorrowStTensorBytes`, the same fail-closed size identity, the
same `VT_LOAD_DIRECT_UPLOAD=0` rollback, the same `MaybeReleaseSourcePages` on
the copy arm. Both encodings are byte-identical between the two spellings
(E2M1 nibbles + fp8-e4m3 group-16 scale); only the global-scale convention
differs, and that is arithmetic on a scalar, not on the borrowed range.

Fix 2 — `return platform.residency_policy().release_host_weights_after_upload;`.

## Tests

`tests/vllm/models/test_qwen3_5_dense_load_residency.cpp`, a new binary. Every
case enters through the production loader `vllm::LoadQwen3_5Dense(shards, config)`
over a **real safetensors file on disk**, because `BorrowStTensorBytes` fails
closed on a synthetic `StTensor` with no `mapping` and an in-memory fake would
therefore report the defect as fixed.

| Case | Claim |
|---|---|
| ModelOpt borrow | Every `Nvfp4Weight` the loader produces from ModelOpt-named tensors borrows: `bytes.borrowed()`, `bytes.data()` inside the mapping, `mmap_src` set, and the ranges counted under `borrowed` rather than `host_copy`. **Red before Fix 1.** |
| compressed-tensors borrow | The sibling spelling still borrows. Guards against a repair that moves the defect rather than removing it. |
| rollback | With the direct-upload decision forced off, the ModelOpt arm copies and loads the **same bytes**, so the lever fails closed. |
| unified-memory staging | A fake platform with `needs_weight_staging()`, a unified backend and `release_host_weights_after_upload` reaches `StageAndReleaseLoadedDense` through `LoadQwen3_5Dense(..., &queue)` on a plain bf16 checkpoint. **Red before Fix 2.** |

## Reachability

Both changes are reached from `LoadQwen3_5Dense`, which is the loader — a
production entry point under `AGENTS.md` `## Nothing lands dead`. The production
call sites to delete for the mutation are `LoadDenseMlp`'s
`m.gate_proj_fp4 = LoadNvfp4AnyNaming(...)` (`:599`) for Fix 1 and
`if (direct_device) StageAndReleaseLoadedDense(w, *load_queue);` (`:939`) for
Fix 2.

## Risks

- A borrowed range keeps its mapping alive through `StTensor::mapping`, so the
  shard stays mapped past `~SafetensorsFile`. That is the design
  `ENG-LOAD-DIRECT-UPLOAD` already ships for the CT spelling and the whole
  reason the keep-alive exists; this row adds no new lifetime.
- `MaybeReleaseSourcePages` moves from the load to `AdoptDeviceBytesAsHost` on
  the borrow arm. `test_load_direct_upload.cpp` already pins that ordering.
- Nothing here is measurable without a GB10, and this session has no GPU and no
  fleet access. The load-time byte accounting is the CPU-observable half; the
  `avail` curve is owed.

## Gates

```sh
ctest --test-dir build -R 'test_qwen3_5_dense_load_residency|test_load_direct_upload|test_qwen35_plain_weights|test_resident_weight_host_addressable|test_qwen38_27b_modelopt_mtp_arm' --output-on-failure
scripts/agent-preflight.sh --staged
```

`test_cpu_x86_llamacpp_floor` ([#618](https://github.com/mudler/vllm.cpp/issues/618))
is the one known-flaky exception.

## Evidence

Recorded in the pull request body and in `## Outcome` at `DONE`: the red
capture for each new case before its fix, the green after, each mutation with
the tree hash taken before and re-verified after, and the two reachability
mutations.

## Owed

- **The faithful long-term target: upstream's polarity.** Allocate the final
  packed device parameter first and stream the shard into it, as
  `base_loader.py:52-58` + `weight_utils.py:969-974,1247` do. This row is the
  minimal faithful mirror of the *residency*, not of the *order*. Needs its own
  row, its own spec and a device to measure on.
- **Upstream's integrated-GPU memory accounting**, `mem_utils.py:148-155` and
  `:54-83`: read available memory from the OS rather than from
  `cudaMemGetInfo` on an integrated device, and release under pressure. Nothing
  in this tree does either, and it is why the oracle degrades where we reboot
  the box.
- **A bound that refuses before the allocator takes the box's system services.**
  `--kv-cache-memory` bounds only the KV pool, and the older
  `--gpu-memory-utilization` gap named in the section above leaves nothing
  bounding the rest. This bullet does not claim that gap's issue: it is a
  separate, unowned filing, and naming it here would make this spec its owner
  under `scripts/check-agent-record.py`. Named in #1647's own `## Owed`.
- **The GB10 re-measurement of #1647's own numbers** at this head: the `avail`
  curve, `RssAnon` after target load, and the DFlash2 double-model configuration
  the issue was filed against. This session has no GPU.
- **`ReleaseResidentQwen3_5DenseHostWeights` ignores every `Nvfp4Weight`.** It
  walks only `OwnedTensor` fields, so no quantized dense checkpoint can ever
  have a host mirror released through it. Fix 1 makes this moot for the borrow
  arm (adoption handles it) and leaves it true for the copy fallback.

## Stop conditions

- `NEEDS_DECISION` if either fix turns out to change loaded bytes rather than
  where they live.
- `BLOCKED` on anything that requires a GPU. This session has none, by
  instruction.

## Now

`ACTIVE` — both fixes and the red-first gate are in
[PR #1820](https://github.com/mudler/vllm.cpp/pull/1820), awaiting a fresh
review. The row does not reach `DONE` until the GB10 re-measurement under
`## Owed` is paid, because every gate on this branch is CPU-only.
