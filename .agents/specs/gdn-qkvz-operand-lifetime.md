# GDN `in_proj_qkvz` operand lifetime (issue #2476)

Row: `MODEL-MM-QWEN4-EXP`
Issue: https://github.com/mudler/vllm.cpp/issues/2476

## Now

`ACTIVE` — the fix for the illegal memory access that blocks the production
CUDA arm of `qwen4_exp`.

## Scope

One defect: the buffer the GDN `in_proj_qkv` / `in_proj_z` GEMM is pointed at
is owned by a per-step temporary and is freed while the GEMM is still queued.

Out of scope: #2496 (the garbage-token sequence), the QSA block, the packed
GDN decode ceiling, and the per-step MoE adapter rebuild that
`qwen4_exp_forward.cpp` already records as owed.

## The measurement this starts from

`compute-sanitizer memcheck` on `thor:gpu0` (sm_110, CUDA 13.0.88), released
`unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ1_S, binary
`70e522df28d7748d8925ce9fc54dffd4909b3e8fd53d72fafd87bb13bed62056` from
`e934fb0020ba099125994e9a44e93e8c89d87977`. Evidence on the share at
`/mnt/nas_share/rc/q4exp-sanitize/out-san/`.

Facts that constrain the diagnosis, read off `server.log`:

- The prefill (`prompt_tokens=5`) and **five** decode steps complete. The first
  sanitizer report is at `server.log:58`, inside the sixth decode step.
- The faulting kernel is `nvjet_sm110_tst_512x8_64x3_2x1_v_bz_TNT`, reached
  through `cublasLtMatmul` <- `MatmulBTKernelCuda` <- `MatmulBf16D` <-
  `ProjectGdnQkvz` <- `GdnBlockPaged` <- `Qwen4ExpTextModelForward` <-
  `ModelRegistry::Forward`.
- The reports are **block 17 threads 128..159, then block 18 threads 128..135**
  (`--print-limit 40` truncates there). No lower block index appears before
  them. At the released geometry `conv_dim = 2*16*128 + 48*128 = 10240`, so the
  col-major `M` axis of this GEMM is 10240 and a 512-wide tile ladder has 20
  blocks.

  READ THAT AS A SHAPE, NOT AS AN ADDRESS. It says the fault is PARTIAL — some
  of the operand is addressable and some of it is not — which is what a
  contiguous freed-and-partly-remapped range looks like and what a wholly
  invalid base pointer does not. It does NOT establish which rows are missing:
  the block-index-to-operand-row mapping of a closed-source `nvjet` kernel is
  not published, and `--print-limit` truncated the report. Nothing below depends
  on the row number.
- `[vt load] w0f-alias per-call totals` grows `rehomed` by about 1 GiB on
  **every** step (3.381 -> 3.673 -> ... -> 6.079 -> 6.406 GiB). A weight that
  has been re-homed once is 256-byte aligned and takes the
  `kAliasedInPlace` branch forever after, so a `rehomed` counter that keeps
  climbing proves that **fresh owned host buffers appear each step**.

## The extents, at both token counts

Read off the code for the released geometry (`hidden_size` 2560, 16 key heads
and 48 value heads at `head_dim` 128, so `key_dim` 2048, `value_dim` 6144,
`conv_dim` 10240). `MatmulBf16D` -> `vt::MatmulBT` -> `MatmulBTKernelCuda` sets
`m = a.shape[0]`, `k = a.shape[1]`, `n = b.shape[0]` and builds column-major
layouts `A[k,n] ld=k`, `B[k,m] ld=a.stride[0]`, `C[n,m] ld=n`.

| operand | told (elements) | allocated | prefill `T=5` | decode `T=1` |
|---|---|---|---|---|
| `in_proj_qkv` (A) | `k*n` = 2560 x 10240 | `[10240,2560]` bf16 | 52,428,800 B | 52,428,800 B |
| `in_proj_z` (A) | 2560 x 6144 | `[6144,2560]` bf16 | 31,457,280 B | 31,457,280 B |
| `mixed` (B) | `k*m` = 2560 x `T`, ld 2560 | `DBuf [T,2560]` bf16 | 25,600 B | 5,120 B |
| `mixed_qkv` (C/D) | `n*m` = 10240 x `T`, ld 10240 | `DBuf [T,10240]` bf16 | 102,400 B | 20,480 B |

**Every told extent equals its allocation at both token counts, and no extent
is derived from a token count other than the one the operand was allocated
with.** `T` reaches the GEMM only as `m`, and `m` is read from the activation's
own `shape[0]`; the activation is the `DBuf(d, dt, {T, H})` the same layer
allocated. `vt::MatmulBT` re-checks `a.shape[1] == b.shape[1]`,
`out.shape == {a.shape[0], b.shape[0]}` and contiguity before dispatch, so a
disagreement would refuse by name rather than fault.

So the token-count hypothesis is REFUTED for this GEMM: there is no dimension
that is right at `T=5` and wrong at `T=1`. The one `T`-dependent behaviour worth
naming is `DevicePool`'s size-class rounding, which serves a `T=1` request from
a block that may have held a `T=5` activation. That gives a block LARGER than
the request, never shorter, and `Qwen4ExpGatedResidual` writes every logical row
before the GEMM reads it, so no stale row is reachable.

## The defect

`qwen4_exp_forward.h` states the invariant this code breaks:

> nothing is transposed, reordered or reallocated here — the returned
> `OwnedTensor`s are COPIES OF THE HANDLES and share the loader's bytes.

`Qwen4ExpGdnBlockWeights` implements that with plain assignment
(`w.in_proj_qkv = g.in_proj_qkv;`). `OwnedTensor`'s implicit copy constructor
deep-copies `OwnedBytes`, which for the GGUF arm is an **owned**
`std::vector<uint8_t>` (`reorder` is on at the released 16-vs-48 head ratio, so
`LoadGdn` expands every GDN projection to bf16 through `Bf16From`). So the
adapter reallocates ~115 MiB of GDN weights per linear layer per step, and:

1. the copy's `d_dev` / alias memo is written to a temporary and thrown away,
   so residency is re-established from scratch every step; and
2. `ResidentWeight`'s host-alias arm — the arm this box takes, because
   `StagingFitsModel` is false for a 65 GiB checkpoint on a 122 GiB box —
   returns `MakeTensor(w.bytes.data(), ...)`, i.e. a **host** pointer into the
   temporary's buffer, and hands it to `cublasLtMatmul`.

`const GdnLayerWeights gw` is scoped to the `if (linear)` arm of the layer
loop. It is destroyed at the end of that arm, which runs `::operator delete` on
the aligned block (or `~vector` on the copy). Every GDN kernel of that layer is
still only *queued*. `free()` of a 52 MiB block `munmap`s it, so the GEMM then
reads unmapped host virtual addresses.

This is an **extent-correct, lifetime-wrong** operand: `M`, `N`, `K` and every
declared byte length agree with each other and with the allocation at launch
time. What disagrees is *when* the allocation stops existing. It is a fixed
index, not a race: `CUDA_LAUNCH_BLOCKING=1` completes the kernel inside the
launch, before the scope exits, which is exactly why it suppresses the fault.
The partial validity (blocks 0..16 fine, 17+ faulting) is the freed range being
partly re-mapped by the next layer's copies before the kernel runs.

## Design

Two edits, one invariant: *the operand a queued kernel reads is owned by the
model, never by a per-step temporary.*

1. **Share the bytes.** `Qwen4ExpGdnBlockWeights` builds each field as a
   zero-copy view over the loader's buffer (`OwnedBytes::KeepAlive()` +
   `OwnedBytes::Borrow`), which is what the header already claims and what
   `Qwen4ExpMoeBlockWeights` already does for the shared expert's projections.
   The helper is lifted to `qwen3_5_weights.{h,cpp}` as
   `BorrowWholeOwnedTensor` so there is one implementation, and
   `qwen4_exp_moe.cpp`'s `BorrowWhole` delegates to it.
2. **Build it once.** The adapter is cached on `Qwen4ExpLayerWeights`, so the
   residency memo lives as long as the model and no temporary ever owns a
   device allocation either.

Edit 1 alone fixes the reported fault on the aligned arm. Edit 2 is what makes
the fix independent of `malloc` alignment and of `cudaFree`'s implicit
synchronisation, and it removes the per-step re-upload the same lines cause.

## Risks

- Sharing turns the loader's owned buffer into a shared read-only one
  (`OwnedBytes::Share()`), so nothing may write through it afterwards. Nothing
  does: the GDN weights are read-only from the end of the load.
- Caching pins the adapter for the model's life. It holds no bytes of its own,
  so the residency cost is zero.

## Tests

- `tests/vllm/models/test_qwen4_exp_layer_loop.cpp`: the adapter's every field
  points at the loader's own buffer (pointer identity), and after a forward
  through `Qwen4ExpTextModelForward` the layer's GDN buffers are shared rather
  than duplicated. Red before, green after.

## Gates

```sh
ctest --test-dir build -R qwen4_exp --output-on-failure
```

## Stop conditions

A GPU rerun on `thor:gpu0` is required to claim the production arm completes a
forward. Without it this row claims the defect and the fix, not the arm.

## Owed

- The GPU rerun of `compute-sanitizer` on the fixed binary, and the token
  sequence with no `CUDA_LAUNCH_BLOCKING`.
- Whether #2496's garbage tokens share this root cause. The evidence available
  here says NO, on two independent grounds. First, #2496 was measured UNDER
  `CUDA_LAUNCH_BLOCKING=1`, and blocking launches complete this GEMM before the
  scope that owned its operand closes, so this defect is inert on the arm that
  produced it. Second, the sibling wave measured #2496's wrong decode as
  bit-stable across builds, and this defect's wrong read is whatever the
  allocator last put at a freed address, which is not stable. The extent table
  above adds a third: no operand of this GEMM has a token-count-dependent
  extent, so the "correct prefill, wrong decode" shape does not come from here.
  The fixed binary should still be re-measured, because this defect is live on
  the production (non-blocking) arm and can corrupt what decode carries forward
  there.
