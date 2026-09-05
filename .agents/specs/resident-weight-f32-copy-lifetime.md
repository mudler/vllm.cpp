# `ResidentWeightF32` frees its copy source before the copy retires — #2711

`ResidentWeightF32` builds an f32 upcast of a bf16 weight in a **function-local
`std::vector<float>`**, hands that vector's `data()` to `Backend::Copy`, and lets
the vector die at the closing brace. `Backend::Copy` is asynchronous on both
device backends, so the driver may still be reading a buffer the allocator has
already reclaimed.

Issue: [#2711](https://github.com/mudler/vllm.cpp/issues/2711). Owning row:
`ENG-EXPERT-STREAM-DEVICE` ([engine-matrix.md](../engine-matrix.md)), which is
the row the issue names and the row whose W0f work created the current shape of
these two helpers.

## The defect, grounded

| Where (line anchors read at this row's base, `8e582a5f9`) | What |
|---|---|
| `include/vllm/model_executor/models/dense_attn_block.h:342-360` | The wide copy. 49 translation units under `src/vllm/model_executor/models/` include this header. |
| `src/vllm/model_executor/models/qwen3_5.cpp:1359-1379` | A private twin with the same body. The header flags its existence at `:222-223`; `qwen3_5.cpp` explains why it stays private at `:790`. |
| `src/vt/cuda/cuda_backend.cu:116-118` | `cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, AsStream(q))` — `Copy` is genuinely asynchronous. |
| `src/vt/rocm/rocm_backend.hip:269-271` | `hipMemcpyAsync` — likewise. |
| `src/vllm/model_executor/models/glm5_next_kv.cpp:143-150` | The in-tree precedent. It synchronises on **every** span and names this exact hazard: a deferred wait "would hand the driver a pageable source that the next iteration has already overwritten." |

The sibling `ResidentWeight` (`dense_attn_block.h:249-268`) issues the same
asynchronous `Copy` and does **not** synchronise, and that is correct there: its
source is `w.bytes`, an owned buffer or a file mapping that outlives the call.
The asymmetry is the whole defect. `ResidentWeightF32`'s source is a temporary.

`StageAndReleaseLoadedDense` (`qwen3_5_dense_weights.cpp:181-186`) already calls
`Backend::Synchronize` once, after `PrepareBf16Resident` returns. That drain is
**not** a repair: by the time it runs, every `std::vector<float>` the loop
created has been destroyed. It is the deferred-wait shape the `glm5_next_kv.cpp`
comment rejects, written out in full.

## The design call, and what it costs

Two repairs are available.

1. **Synchronise before the source leaves scope.** Correct, and it adds one
   stream drain per f32-upcast weight.
2. **Keep the source alive until the copy retires.** With no completion callback
   in `vt::Backend` and no polling of `Event`, "until it retires" degenerates to
   "for the weight's lifetime" — a permanent host allocation the size of the
   upcast, on every f32 weight, on every model.

**Chosen: (1), synchronise.** The reasons, in order:

- Option (2) permanently doubles the residency of exactly the weights this
  subsystem spends its effort *not* holding twice. `AdoptDeviceBytesAsHost` and
  `ReleaseResidentQwen3_5DenseHostWeights` exist to drop host mirrors after
  upload; option (2) would reinstate one for every upcast weight and make the
  #1299 memory work argue with itself.
- The tree already answered this question once, in `glm5_next_kv.cpp:143-150`,
  and answered it with a synchronise per span.
- The population is bounded and small. `ResidentWeightF32` is memoised on
  `OwnedTensor::d_dev_f32`, so the synchronise runs **once per distinct weight
  per process**, not per token and not per forward. The f32 upcast serves the
  per-head q/k norms, the GDN `conv1d_weight` and the GDN `norm_weight`:
  `PrepareBf16Resident` passes exactly two of them per layer, so the added drains are `2 * n_layers` for the whole process.

**What is NOT claimed: the wall-clock cost was not measured.** No device
measurement was taken for this change. This box is CPU-only, where `Copy` is a
`memcpy` and `Synchronize` is the `vt::Backend` default no-op, so a timing run
here would measure zero by construction and would be worthless. The choice is
therefore conservative and argued rather than measured: it is bounded above by
`2 * n_layers` drains of an otherwise-idle queue, each waiting on one small H2D
transfer that the model must complete before its first use in any case, against
a load that moves tens of GiB from disk. If that bound is ever found to matter,
the follow-on is a batched variant that keeps the sources alive in a caller-owned
vector and drains once — which is option (2) with a bounded lifetime, and needs
every caller to participate. It is not taken here.

## Design

The two bodies are unified into **one** shared implementation rather than
repaired twice, because #2711 is precisely the failure of having two.

`dense_device_glue.h` gains `dense_attn::InstallResidentF32(Dev, const
OwnedTensor&, std::vector<float>)`: the CPU alias arm, the device staging arm,
and the new `Synchronize`. It is the right home and not a new one — that header
already carries `Dev`, `DBuf` and `MakeTensor`, was created for exactly this
"both sides need it and the include graph must not cycle" problem, and is
already included by `dense_attn_block.h` and by `qwen3_5.cpp` (`:24`).

Both `ResidentWeightF32` bodies become two lines: install if absent, then return
the view. **This is a three-file change, not a 49-file refactor.** The signature,
the namespace and the memoisation field are all unchanged, so no consumer of
either helper is touched.

## Tests

`tests/vllm/model_executor/test_resident_weight_f32_copy_retires.cpp`, its own
binary, for the same reason as its two siblings: it registers a fake backend and
a fake platform in the process-global `kXPU` slot.

The fake backend **defers**. `Copy` records `{dst, src, bytes}` and returns
without moving anything; `Synchronize` performs the recorded `memcpy`s and clears
the queue. `Alloc` poisons its block. That is the shape of the contract
`cudaMemcpyAsync` actually offers, and it makes the defect deterministic instead
of dependent on a driver's mood: with no `Synchronize` the destination still
holds the poison when the function returns.

Cases:

1. **Through a production entry point.** `Qwen3_5DenseModel::PrepareBf16Resident`
   is a real load-time hook (`StageAndReleaseLoadedDense` calls it) and it
   passes `attn.q_norm` and `attn.k_norm` to the qwen3_5.cpp helper. After it
   returns, `d_dev_f32` must hold the f32 upcast of the weight and must not be in
   the backend's pending queue.
2. **The header copy**, `dense_attn::ResidentWeightF32`, driven directly. Its own
   production callers are `DenseAttnBlock`'s four `attn_f32 ? ResidentWeightF32`
   sites, on every attention layer of the 49 models that include the header, and
   driving one of those needs full attention metadata and a KV cache. This
   case therefore calls the inline helper directly and is labelled as doing so.
   It shares one body with case 1 after this change, which is what makes case 1's
   production evidence carry.
3. **The CPU arm is unchanged**: still aliases, still allocates nothing,
   still never synchronises.

### What the gate proves, and what it does not

It proves that `ResidentWeightF32` does not return while its copy is
outstanding — that the source outlives the transfer under a backend that defers.

It does **not** prove that a real CUDA or ROCm driver defers this particular
transfer, and it cannot: on the CPU backend `Copy` is a `memcpy` and the race is
not expressible. A device-observed reproduction would need a GPU lease and a
driver that defers a small pageable H2D copy, which the issue itself records as
usually staged eagerly. The gate is a **structural and ordering** gate over a
simulated asynchronous backend, not an observation of the race.

## Gates

- `tests/vllm/model_executor/test_resident_weight_f32_copy_retires.cpp` red
  before the change with both counts non-zero, green after.
- `ctest --test-dir <build>` for the affected suites, then the full gate.
- `scripts/agent-preflight.sh`, read by grepping for `gate(s) failed` and
  `NOT a green` rather than by its exit code.
- `python3 scripts/check-pr-size.py --base origin/main --head HEAD` by hand,
  because that checker is CI-only.

## Mutations

Each rebuilt, each restored and the restoration verified by `sha256sum`.

- **M1** — delete `d.b.Synchronize(d.q)` from `InstallResidentF32`. Cases 1 and 2
  must fail. This is the defect itself.
- **M2** — move the `Synchronize` after the `shared_ptr` install but still inside
  the function. Must stay green: the source is still alive, and a gate that
  reddens here would be pinning a line number rather than a guarantee.
- **M3** — delete the `f32(attn.q_norm)` / `f32(attn.k_norm)` calls from
  `PrepareBf16Resident`. Case 1 must fail. This is the reachability mutation:
  it removes the production call site, not the fix.
- **M4** — make the fake backend's `Copy` eager (perform the `memcpy`
  immediately). Cases 1 and 2 must stay green **on the unfixed tree**, which is
  the control: it shows the red in M1 comes from the deferral and not from the
  fixture.

## Risks

- `dense_device_glue.h` is included by both consumers; a mistake there breaks 49
  translation units at once. Mitigated by the body being moved verbatim with one
  statement added.
- The fake platform occupies the process-global `kXPU` slot. Mitigated by giving
  the suite its own binary, as `test_resident_weight_host_addressable.cpp` and
  `test_expert_stream_device_slot.cpp` already do for the same reason.

## Owed

Nothing. The one adjacent question this change raised was grounded and closed
rather than left as a suspicion, and it is recorded here so the next reader does
not re-find it:

**`ResidentWeight` calls `AdoptDeviceBytesAsHost` immediately after its own
asynchronous `Copy` out of `w.bytes`, and that function can release the copy
source.** It is the same shape of question #2711 asks, on a different helper,
and it is **not** a live defect. Both halves were checked:

- The half that can `munmap` synchronously (the `self.bytes = OwnedBytes::Borrow(...)`
  reassignment, whose own comment records that it can drop a shard's last
  mapping reference inside the assignment) is gated on
  `backend.DeviceMemoryIsHostAddressable()`. Exactly one backend in this tree
  answers that true — Vulkan, `src/vt/vulkan/vulkan_backend.cpp:135` — and
  Vulkan's `Copy` is a synchronous `std::memcpy` (`:101-106`). Where `Copy` is
  asynchronous the branch returns before reaching the reassignment.
- The half that always runs, `ReleaseDirectUploadSource`, is
  `ReleaseSourcePages`, which is a `madvise(MADV_DONTNEED)` over whole interior
  pages of a clean `PROT_READ MAP_PRIVATE` file mapping
  (`safetensors_reader.cpp:314-337`). Those pages re-fault from the file to
  identical bytes, so a driver reading them later reads the same weight.

This becomes real the day a backend that answers `DeviceMemoryIsHostAddressable()`
true gains an asynchronous `Copy`. It has no issue because there is no defect to
track, and filing one would put an unowned false positive into an intake that
already carries six issues the tree had answered.

## Outcome

Landed as one shared body plus one `Synchronize`. What was measured, and what was
rejected:

**The gate.** RED on the pre-fix tree at both counts -- 3 cases with 2 failed,
17 assertions with 6 failed. GREEN after: 3 of 3 and 17 of 17. The 30 test
targets whose sources reach `dense_attn_block.h`, `dense_device_glue.h`,
`qwen3_5_internal.h`, `ResidentWeightF32`, `StageWeightForTest` or
`PrepareBf16Resident` were built and run: 0 failing binaries.

**The mutations**, each rebuilt before it was run and each restored with the
restoration verified by `sha256sum -c` against a baseline taken at the fix
commit:

| # | Mutation | Build | Result |
|---|---|---|---|
| M1 | delete `d.b.Synchronize(d.q)` from `InstallResidentF32` | rc 0 | 2 of 3 cases, 6 of 17 assertions FAIL |
| M2 | move the `Synchronize` after the `shared_ptr` install | rc 0 | GREEN 3/3, 17/17 |
| M3 | delete `f32(attn.q_norm)` / `f32(attn.k_norm)` from `PrepareBf16Resident` | rc 0 | case 1 FAILS, 1 of 12 assertions (a `REQUIRE` ends the case) |
| M4 | revert the fix AND make the fixture's `Copy` eager | rc 0 | GREEN 3/3, 17/17 |

**M4 is the finding worth keeping.** With an eager `Copy` -- which is exactly
what the CPU backend does -- the UNFIXED tree is fully green. A CPU-only host
cannot see this defect at all, and that is a property of the box rather than a
weakness of the case. It is why the fixture had to simulate deferral to gate
anything, and why the gate is stated as structural and ordering rather than as
an observation of the race.

**M2 is the second one.** Moving the drain one statement later leaves the gate
green, and it should: the source is still alive there. A gate that reddened on
M2 would be pinning a line rather than the guarantee.

**Rejected: the batched drain.** `StageAndReleaseLoadedDense` already
synchronises once after `PrepareBf16Resident` returns, and it was tempting to
call that sufficient. It is not, and the reason is the whole issue: by the time
it runs, every `std::vector<float>` the loop created has been destroyed. A
deferred drain is the shape `glm5_next_kv.cpp` rejects, and this one is that
shape spread over an entire model.

**Rejected: owning the staging buffer.** See `## The design call` above. The
short form is that `vt::Backend` gives no way to know when a copy retired, so
ownership has no release point short of the model's lifetime.

**Not measured: the wall-clock cost of the drain.** Stated as unmeasured rather
than estimated, because a number produced on a host where `Synchronize` is a
no-op would be a fabrication wearing a measurement's clothes.

**Not run: the full `ctest`.** Each test binary is 26.6 MB and the box had 14 GB
free at 97% full; 661 of them need about 17.6 GB, and an ENOSPC makes checkers
emit false policy refusals rather than verdicts. The 30 reachable suites were run
instead. This is a narrower gate than a full `ctest` and is recorded as such.

## Stop conditions

- Stop and report `NEEDS_DECISION` if closing the header copy turns out to need
  changes in more than the three files named above.
- Stop and report if the red case cannot be made to fail for the stated reason —
  a case that fails by throwing is a different result and is not the red this
  change needs.
