# `vt::AttentionCross` gets the decomposition FlashAttention-2 actually has

Row `KERNEL-ATTENTION-CROSS-CUDA`, issue
[#1555](https://github.com/mudler/vllm.cpp/issues/1555), found by
[#1542](https://github.com/mudler/vllm.cpp/issues/1542) and listed under
`## Owed` in [`minimax-music3.md`](minimax-music3.md) §21.7. The DiT half is
owned by
[#672](https://github.com/mudler/vllm.cpp/issues/672).

This is a `vt` kernel row with two model consumers, so it has its own file.
`minimax-music3.md` §15-§21 are taken and none of them is the owner of a shared
op.

## Now

`ACTIVE`. The mechanism is measured, the oracle is determined, the kernel is
written and the unit gate is written. Speed is a same-box A/B on `thor:gpu0`
and is recorded in §6 with its state.

## 1. Scope

`vt::AttentionCross`'s **CUDA** kernel, and nothing else.

**In scope.** A second CUDA kernel for the op, registered above the existing one
through the `vt` provider seam and declining per call on a shape it has no
tiling for. The two tilings that fit the 48 KiB of static shared memory a launch
gets without opting in: head_dim 64 and head_dim 128.

**Out of scope, and each for a reason.**

- **The CPU kernel.** It is the correctness reference for this op and the
  backbone every other backend ports from (`.agents/backends.md`). It does not
  move.
- **`vt::Attention` and `vt::AttentionDenseFlash`.** Different ops with
  different contracts. `vt::Attention` is deliberately frozen on the naive
  kernel so text decode stays byte-identical, and
  [#1549](https://github.com/mudler/vllm.cpp/issues/1549) is separately moving
  LTX-2.5's *self*-attention off it. Nothing here touches either.
- **The dtype.** f32 is upstream's resolved choice for the MiniMax-Music3 DiT,
  established against the pinned `diffusers` oracle `c6da9936` in §21.2 with
  anchors. A bf16 or TF32 attention would be a divergence from the oracle, not a
  repair of one. This row changes the DECOMPOSITION and not the arithmetic type.
- **A tensor-core arm.** There are no f32 tensor cores, and the bf16 arms are
  owned by [#1551](https://github.com/mudler/vllm.cpp/issues/1551).

## 2. The gap, as it was measured

`.agents/specs/minimax-music3.md` §21.9, `rc` job
`0f95377f-70dd-4bf8-93b5-8e44fd762713` on an idle `thor:gpu0`, checkpoint staged
to local disk, intra-DiT spans partitioning `denoise.dit_device` to 99.98 %:

| group | s | % of the DiT forward | TFLOP/fwd | TFLOP/s |
|---|---:|---:|---:|---:|
| **`vt::AttentionCross`** | **11.010** | **43.9** | 0.140 | **0.204** |
| the four `vt::MatmulBT` GEMMs | 13.390 | 53.3 | 3.334 | **3.98** |
| norms, rope, SiLU, pack, readback | 0.700 | 2.8 | — | — |

**19.5x slower per flop than the GEMMs in the same forward on the same device**,
against a Thor fp32 CUDA-core peak of 5.369 TFLOP/s that the GEMMs reach 71.5-79 %
of (§21.10). At the shipped 20 s / 30 steps that is **165.1 s of the 370.5 s
`denoise.dit_device` bucket and 27.6 % of the whole run**.

The geometry is measured, not inferred: `dit.seq_sum / 16 = 690`, window 689
latent frames, 36 blocks, `num_attention_heads` 32 and `attention_head_dim` 64
(`minimax_music3_loader.h:206-207`), f32, no bias.

## 3. THE MECHANISM — measured, and it is NOT what #1555 guessed

#1555 named a per-key `__shfl_xor_sync` butterfly and "about 8 warps per
scheduler" and labelled both a hypothesis, because **no `ncu` counter had been
read**. This section reads what could be read and ablates what could not.

### 3.1 `ncu` counters are UNAVAILABLE on this device, and that is a finding

`cuda-nsight-compute-13-0` installs inside the lease and `ncu` 2025.3.1.0 runs,
and every profiled launch returns:

```text
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA
GPU Performance Counters on the target device 0.
```

The `rc` worker is **root** and it still cannot read them: the restriction is
`NVreg_RestrictProfilingToAdminUsers` on the driver module on the HOST, which is
not a lease's to set and not this row's to change. So **there is no hardware
counter in this section, no occupancy counter, no stall-reason histogram and no
memory-throughput percentage**, and none is quoted. The attribution below rests
on ablation and on the CUDA occupancy API, both of which are stated as what they
are.

Two figures printed by the profiled runs are **VOID and are not quoted
anywhere**: under `ncu` the shipped kernel read 17.25 ms and the candidate
7.48 ms against 16.59 and 2.18 unprofiled. Serialised replay is not a timing.

### 3.2 The instrument that WAS available: one ablation per hypothesis

A standalone `nvcc` probe transcribing `AttentionCrossFlashKernel` verbatim, at
the DiT's own geometry, built and run inside an `rc` lease on `thor:gpu0`
(`rc` job `90c42f65-10f7-4040-8dbe-7ecd192105e1`, worker `rc-worker-m4d7t`,
boot id `c99b7805-6e26-47a7-bc9d-93d592d676a6`, `uptime` **3.39-3.41 with 0
logins** across all three rounds, which is this box's idle floor; nvcc 13.0.88,
`-O3 -std=c++17 -arch=sm_110`, probe source sha256
`9db64d3309bbe925ba71a6cd343b1330bfc02e13736b7b6dca00408970f838f6`). Each
variant is the SAME kernel with EXACTLY ONE mechanism removed, so each delta
attributes time to that mechanism. Every variant writes `out` and the harness
checksums it, so an accidentally-emptied kernel cannot read as fast.

`nvidia-smi` reports `clocks.sm` as `[N/A]` on this device, so every figure here
is a WITHIN-RUN split on one binary in one process and none is quotable as a
cross-box number. Three rounds; the medians agree to 0.03 %.

**Tq = S = 690, Hq = Hkv = 32, head_dim 64, f32. `grid(44, 32)`, 512 threads,
`bc = 64`, 32 KiB dynamic shared memory. 3.9002e9 flop.**

| variant | med ms | TFLOP/s | vs V0 |
|---|---:|---:|---:|
| **V0 shipped** | **16.591** | 0.235 | — |
| V1 the `__shfl_xor_sync` butterfly DELETED | 13.588 | 0.287 | **−18.1 %** |
| V2 the online softmax DELETED | 13.303 | 0.293 | **−19.8 %** |
| V3 the per-tile K/V GLOBAL read DELETED | 17.979 | 0.217 | **+8.4 % SLOWER** |
| V0 at 8 query-warps per CTA | 16.980 | 0.230 | **+2.3 % SLOWER** |
| V0 at 4 query-warps per CTA | 19.915 | 0.196 | +20.0 % slower |
| **VB the blocked candidate** | **2.176** | **1.792** | **7.62x FASTER** |

Occupancy, from `cudaOccupancyMaxActiveBlocksPerMultiprocessor` and
`cudaFuncGetAttributes` — an API, not a counter:

| variant | regs | shared | blocks/SM | warps/SM | occupancy |
|---|---:|---:|---:|---:|---:|
| V0 shipped (512 thr) | 58 | 32 KiB dyn | 2 | 32 | 0.67 |
| V0 at 256 thr | 58 | 32 KiB dyn | 4 | 32 | 0.67 |
| V0 at 128 thr | 56 | 32 KiB dyn | 6 | 24 | 0.50 |
| VB blocked (128 thr) | 96 | 42 KiB static | 5 | 20 | 0.42 |

Device: `NVIDIA Thor cc=11.0`, 20 SMs, 233472 B shared per SM, 232448 B shared
per block with opt-in, **32 MiB L2**, 65536 registers per SM, 1536 threads per
SM.

### 3.3 What that refutes, and what it leaves

**OCCUPANCY IS REFUTED.** The kernel already achieves 0.67, and halving the CTA
to 8 query-warps — which DOUBLES blocks per SM at the SAME 32 warps per SM —
measures 2.3 % **slower**, not faster. Adding resident CTAs at constant warp
count buys nothing, which is what "not occupancy-bound" looks like. The 128-thread
variant is slower because it drops to 24 warps per SM, which is a different
statement and not evidence for the hypothesis.

**BANDWIDTH IS REFUTED.** #1555 priced the K/V re-read at 17.9 GB per forward and
~10 % of the cost. Deleting that re-read ENTIRELY measures 8.4 % **slower**. The
device has 32 MiB of L2 and the whole of K plus V at this geometry is 11.3 MiB,
so the re-reads never leave L2 and were never DRAM traffic. The sign is what
matters and it is unambiguous; the reason the ablation is slower rather than
neutral is not established here and is not needed for the attribution.

**THE SHUFFLE IS REAL BUT SMALL.** 18.1 %, against a headline claim that named
it as the mechanism. Deleting it and the online softmax TOGETHER would leave
roughly 62 % of the cost standing.

**WHAT IS LEFT is the DECOMPOSITION.** One warp owns one query and walks the keys
**one at a time**, so a whole loop body — two shared loads per lane, the
butterfly, two exponentials, the rescale of the accumulator, the loop — is paid
per key for `head_dim / 32 = 2` useful multiply-adds per lane. The kernel issues
roughly one useful FMA per five to eight issued instructions, and no amount of
parallelism repairs an instruction mix. That is why every ablation of a single
instruction returns a fifth and the restructure returns 7.6x.

### 3.4 The same reading at LTX-2.5's video geometry

Tq = S = 2352, Hq = Hkv = 32, head_dim 128, f32, 9.0635e10 flop:

| variant | med ms | TFLOP/s | vs V0 |
|---|---:|---:|---:|
| V0 shipped | 206.454 | 0.439 | — |
| V1 no butterfly | 172.949 | 0.524 | −16.2 % |
| V2 no online softmax | 169.221 | 0.536 | −18.0 % |
| V3 no K/V global read | 208.321 | 0.435 | +0.9 % slower |
| **VB blocked** | **93.527** | **0.969** | **2.21x** |

The same shape of answer, and a smaller win, for a reason named in §4: the
head_dim 128 tiling is the one the 48 KiB shared budget forces, not the one the
arithmetic wants.

## 4. THE ORACLE — vLLM owns this, and the kernel is a 1:1 port of its structure

This question was answered before the design and not after, because a row was
already spent this session finding a "divergence" against a secondary oracle for
an op vLLM implements.

### 4.1 vLLM implements a comparable primitive, and fp32 is a FIRST-CLASS path in it

At the parity pin `555967922`:

- `vllm/v1/attention/ops/triton_prefill_attention.py::_fwd_kernel` — a dense,
  non-paged, GQA-capable attention kernel, entered non-causally by
  `::context_attention_fwd`.
- `vllm/v1/attention/backends/triton_attn.py::TritonAttentionBackend`
  declares `supported_dtypes = [float16, bfloat16, float32]`, while
  `vllm/v1/attention/backends/flash_attn.py::FlashAttentionBackend` declares
  `[float16, bfloat16]`. `vllm/v1/attention/backend.py::AttentionBackend::supports_dtype`
  is the symbol that rejects f32 for FlashAttention.
- `vllm/platforms/cuda.py::CudaPlatform::get_vit_attn_backend` walks
  `get_supported_vit_attn_backends`' list in order and takes the first backend
  that supports the head size AND the dtype.

**So for f32 at head_dim 64 on CUDA, the pinned vLLM selects `TRITON_ATTN` and
runs `_fwd_kernel`.** f32 is not an accident of that kernel: its own
`::get_block_size` special-cases `torch.float32` to a 32-wide tile.

Note for a later reader: **`vllm/attention/` does not exist at this pin.** The
encoder path moved to `vllm/v1/attention/` and `MultiHeadAttention` /
`maybe_get_vit_flash_attn_backend` are gone, replaced by
`vllm/model_executor/layers/attention/mm_encoder_attention.py::MMEncoderAttention`.
A search of the old paths finds nothing, and finding nothing there is not
absence.

### 4.2 The structure this row ports, side by side

| `_fwd_kernel` | this kernel |
|---|---|
| one CTA per (query tile x **query head**) | same: `grid((Tq+BR-1)/BR, Hq)` |
| `BLOCK_M = BLOCK_N = 32` for f32 | `BC = 32` at head_dim 64 — the same key tile |
| whole head dim in registers, masked | whole head dim swept in registers, tile-masked |
| K loaded **pre-transposed** so `tl.dot` needs no transpose | K staged **transposed** in shared, `[d][key]` |
| `acc`, `m_i`, `l_i` all `tl.float32` | `acc`, `m`, `l` all `float` |
| rescale ONCE per key tile: `m_ij`, `alpha`, `l_i*alpha`, `acc*alpha` | rescale ONCE per key tile: `mt`, `corr`, `l*corr`, `acc*corr` |
| `cur_kv_head = cur_head // (Hq//Hk)` | `g = h / (hq / hk)` |
| ONE final `acc / l_i` | ONE final `acc * (1/l)` |

**The decomposition is the port.** `_fwd_kernel` realizes its two GEMMs with
`tl.dot` on tensor cores; f32 has none, so the same outer-product tiling is
realized on CUDA cores. The STRUCTURE is mirrored; the instruction cannot be, and
that is stated rather than glossed.

**Two deliberate divergences, both recorded rather than chased:**

1. **Base.** `_fwd_kernel` pre-multiplies the scale by `1/ln 2` on the host and
   uses `tl.math.exp2`. This kernel uses `__expf`, which compiles to the same
   `ex2.approx` after one multiply. Mirroring the pre-multiply would save one FMA
   per exponential and would require the additive bias to be pre-multiplied too;
   the exponentials are a small part of this kernel's instruction stream and the
   change is not measured here. Owed, §7.
2. **The masking sentinel.** `_fwd_kernel` masks with a finite `-1.0e8`. This op's
   contract (`include/vt/ops.h`) is torch's: a fully masked key arrives as a large
   negative FINITE number from the caller's bias and an all-masked row degenerates
   to a uniform average. That contract is UNCHANGED by this row and is gated.

### 4.3 The bias arm has NO vLLM equivalent, and its contract does not move

`_fwd_kernel` carries no additive bias, and a content search of
`vllm/v1/attention/` and `vllm/model_executor/layers/attention/` for
`attn_bias`/`attention_bias` at this pin returns exactly one hit and it is a
comment (`vllm/v1/attention/ops/prefix_prefill.py:415`).

So the bias arm's reference is what it already was: torch's mem-efficient kernel,
`ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h::AttentionKernel`,
which applies `scale` first and then adds the bias INTO the f32 accumulator. That
is the order `src/vt/cpu/cpu_ops.cpp::AttentionCrossKernel` and the existing CUDA
kernel already form it in, and it is preserved here unchanged.

### 4.4 What the `diffusers` oracle actually executes, which sharpens §21.2

`diffusers` at the pin `c6da9936`:
`src/diffusers/models/transformers/transformer_minimax_music3.py::MiniMaxMusic3AttnProcessor`
calls `dispatch_attention_fn` with no mask, no `is_causal` and no `scale`;
`src/diffusers/models/attention_dispatch.py::_native_attention` is the default
backend (`DIFFUSERS_ATTN_BACKEND` defaults to `native`,
`src/diffusers/utils/constants.py`), and it is a bare
`torch.nn.functional.scaled_dot_product_attention`. Every flash / cuDNN / sage /
xformers backend in that file is guarded by
`::_check_qkv_dtype_bf16_or_fp16`, so the f32 checkpoint cannot reach one.

For f32 on CUDA that SDPA call resolves to `SDPBackend::EFFICIENT_ATTENTION`, not
to `math`: `sdp::can_use_mem_efficient_attention` accepts `at::kFloat` while
`sdp::check_dtypes_flash_attention` and the cuDNN gate do not, and
`ATen/Context.h::Context::sdp_priority_order` puts `efficient` ahead of `math`.

**One consequence is worth recording because it inverts an intuition.** That
kernel's f32 instantiation uses
`mem_eff_attention/gemm_kernel_utils.h::DefaultGemmType<ArchTag, float, ...>` with
`Operator = cutlass::arch::OpMultiplyAddFastF32` on sm80 and above — **3xTF32
emulation, not true f32 FMA**. So the f32 oracle for this model is itself less
precise than our kernel, and a bit-exact gate against it is not available at all.
Ours is the more faithful arithmetic, and §21.2's conclusion (f32 is the resolved
dtype) is unaffected and now better anchored.

That determination was made by reading the pinned checkouts and, for torch, the
shipped headers plus the compiled `sdp::` predicates in `libtorch_cuda.so`; **no
PyTorch source tree exists on this host and the selection was therefore
established STATICALLY rather than by running
`torch.backends.cuda.can_use_efficient_attention` on a device.** Stated as
unverified-at-runtime.

## 5. Design

### 5.1 The seam

The blocked kernel is registered as a SECOND provider for
`OpId::kAttentionCross` on CUDA through `include/vt/op_provider.h`, at priority
10 above `vt-native`'s 0, under the name `vt-cross-blocked`. It inspects the
shape per call and, when it has no tiling for it, counts a decline with
`NoteOpDecline` and forwards to a once-resolved `GetOpFallback` pointer — the
`DECLINE-AND-FALL-BACK` axis that header describes and that
`src/vt/metal/metal_mlx_provider.mm::MlxMatmulKernel` already uses, including its
measured reason for caching the fallback.

Three things follow, and they are why this shape was chosen over a private env
flag:

- **No call site changes.** All five `vt::AttentionCross` callers are untouched.
- **Selection is OBSERVABLE.** `GetOpProviderStats(kAttentionCross, kCUDA)`
  reports `declines`, which is the discriminator: the blocked provider is always
  the SELECTED one, and what separates a served call from a forwarded one is
  whether it declined. Every gate below asserts it two-sidedly.
- **The A/B lever already exists.** `VT_OP_PROVIDER_DISABLE=vt-cross-blocked`,
  or `DisableOpProvider` from a test, restores the shipped kernel **from one
  binary**. No second build, and therefore no #1516 hash problem.

### 5.2 The kernel

`AttentionCrossBlockedKernel<Tin, Tout, D, BR, BC>`, 128 threads as 16 query
groups x 8 key groups. A thread owns a `QT x KT` tile of the score block
`S[BR, BC]` and a `QT x DT` tile of the output block `O[BR, D]`, with
`QT = BR/16`, `KT = BC/8`, `DT = D/8`.

Per key tile: stage `K^T` and `V`; accumulate `S = Q.K^T` over the whole head dim
with the reduction **sequential in each thread's own registers**; scale, add the
bias, send the tile tail to `-inf`; fold each thread's own `KT` scores to a local
max and a local sum in registers and reduce those across the 8 key groups through
a 2 KiB staging array; advance `m`, `l` and rescale `acc` **once**; then
`O += P.V`.

There is no cross-lane shuffle anywhere, and the recurrence advances once per
`BC` keys instead of once per key. Those are the two things §3.3 says the cost
actually is.

### 5.3 The two tilings, and why exactly two

Both are chosen by the 48 KiB of static shared memory a launch gets without
`cudaFuncSetAttribute`:

| head_dim | BR | BC | QT | KT | DT | shared | regs | measured |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 64 | 64 | 32 | 4 | 4 | 8 | 43 008 B | 96 | 7.62x |
| 128 | 32 | 16 | 2 | 2 | 16 | 35 840 B | 80 | 2.21x |

`BC = 32` at head_dim 64 is vLLM's own f32 block size. `BR = 64` is double
vLLM's `BLOCK_M`, taken because the budget allows it and it halves the number of
times the key stream is walked; the `BR = 32` variant is not measured and is
owed (§7).

### 5.4 The shape gate, stated as a decision

```text
hq % hk == 0            (the op's own contract)
out dtype in {f32, bf16}
head_dim 64  and Tq >= 64
head_dim 128 and Tq >= 32
```

Everything else declines and keeps the kernel it has today, byte for byte.

The `Tq` floor is not a guess dressed as a capability. At `Tq = 1` — which is
exactly what MiniMax-Music3's RVQ depth decoder calls this op with, one query row
against its cached history
(`src/vllm/model_executor/models/minimax_music3_depth_device.cpp`) — 63 of 64
query rows in a tile would be padding, and the warp-per-query kernel is the right
shape for that regime. The gate is on head_dim and query count and **not on
dtype**: a bf16 caller with a long query tile gets the blocked kernel too, and
the staging tiles widen the operand once rather than on every read.

## 6. Correctness, and the two consumers

`vt::AttentionCross` has two model consumers and five call sites. Each is named
with how it is gated, because "the callers pass" is not a statement about an op.

| consumer | call site | shape | takes the blocked path? | gated by |
|---|---|---|---|---|
| MiniMax-Music3 DiT | `minimax_music3_device.cpp::DitForwardDevice` | f32, Tq=S=690, H=32, D=64, no bias | **YES** | `test_ops_attention_cross`, `test_minimax_music3_*` |
| MiniMax-Music3 RVQ depth decoder | `minimax_music3_depth_device.cpp` | bf16, **Tq=1** | **no — declines** | a BYTE-FOR-BYTE case, below |
| LTX-2.5 DiT cross/biased attention | `ltx2_device.cpp::AttentionDev` | D 64 and 128, optional dense bias | yes when Tq clears the floor | `test_ops_attention_cross` (see §6.4) |
| LTX-2.5 host arm | `ltx2.cpp` | CPU only | n/a — CPU kernel untouched | `test_ltx2` |
| LTX-2.5 duration head | `ltx2_duration_head.cpp` | CPU only (`std::vector<float>`) | n/a | `test_ltx2` |

The op's own unit suite `tests/vt/test_ops_attention_cross.cpp` holds **both**
backends to an INDEPENDENT f64 reference computed from the layout contract, not
to each other — the "gate on a shared helper proves consistency, not
correctness" trap this tree has already been bitten by. This row adds cases that
reach the new path and asserts, for each, which arm ran.

### 6.1 The numerics claim, made before the code and not after

**NOT bit-identical to the shipped kernel, and this row does not pretend
otherwise.** The head-dim sum moves from a 32-lane PAIRWISE butterfly to a
SEQUENTIAL f32 accumulation over `head_dim` terms, and the softmax denominator
sums a tile's keys before the running `l` absorbs it. Both are max-subtracted f32
online softmax; both are held to the SAME independent f64 reference at the SAME
committed tolerances. That is exactly the relationship
`src/vt/cuda/cuda_attention_cross.cu`'s header already declares between this op's
CUDA and CPU kernels.

**A wide accumulator makes a reordering unobservable on benign data**, so a suite
built only on uniform random inputs would report a change it cannot see. The
suite therefore carries a CATASTROPHIC-CANCELLATION case in which K alternates
sign along the head dim while Q does not, so consecutive products cancel and the
partial sums reach ~2.6e5 while the true dot product is O(1e2). It asserts three
things and the first is the anti-mute-switch: the EXISTING kernel must itself
miss the f64 reference by more than the benign 2e-5 band, or the bound is
measuring nothing; both arms stay inside one committed absolute bound; and the
new summation is within a committed RATIO of the old one, which is the actual
question a reordering raises and the one a single absolute bound cannot answer.

Both constants are MEASURED on `thor:gpu0` and recorded in §6.3, not chosen.

### 6.2 The cases this row adds

All in `tests/vt/test_ops_attention_cross.cpp`:

1. the provider is registered ABOVE `vt-native`, asserted through
   `OpProviderNameAt` — if that order inverts, every case below still passes
   numerically while measuring the wrong kernel;
2. head_dim 64 with a RAGGED query tile (Tq 80 = 64 + 16) and a ragged key tile
   (S 200 = 6x32 + 8), two-sided on `declines`;
3. a DENSE `[Tq, S]` bias across tiles on the blocked path — the arm with no
   vLLM equivalent and therefore the one most likely to be dropped;
4. `Hq > Hkv` on the blocked path;
5. the bf16 stream at head_dim 64, proving the gate is on shape and not dtype;
6. a fully masked key, re-asserted on the path where the max and the sum travel
   through shared memory across 8 key groups;
7. **the depth-decoder shape declines, BYTE FOR BYTE** — one call, one decline,
   and a bit-level comparison against the same run with the provider disabled;
8. an unhandled head_dim declines once per call, and still answers correctly;
9. the catastrophic-cancellation case of §6.1;
10. the same-binary A/B lever agrees with itself.

### 6.4 LTX-2.5's own suites are BYTE-IDENTICAL, and that is a coverage finding

Every geometry in `tests/vllm/models/ltx2_goldens.inc` is REDUCED-DIMENSION:
`kLtx2Arch_attention_head_dim` is **8**, `kLtx2Arch_audio_attention_head_dim` is
**4**, and `kLtx2VideoTokens` is **8**. All of them fail the shape gate, so every
LTX-2.5 call declines and `test_ltx2` and `test_ltx2_device` are byte-identical
under this change.

**Their green therefore says nothing about the path LTX-2.5 actually runs**,
which in production is head_dim 128 over thousands of tokens and DOES take the
blocked kernel. That is the same defect
`tests/vt/test_ops_attention_cross.cpp`'s own header was written for -- "a
fixture that cannot reach the regime which discriminates is the same defect as a
missing test, wearing different clothes" -- and it is why the head_dim 128
coverage is a case in the op's own suite rather than an inherited one. The
32x16 tiling is a DIFFERENT instantiation from the 64x32 one: `QT` and `KT` fall
to 2 and `DT` rises to 16, so the register tile, the cross-group reduction and
the output mapping are all different code.

The LTX-2.5 suites are still run and reported, because byte-identical is a claim
that has to be measured rather than asserted.

### 6.3 The tolerances, MEASURED — and the reordering went the other way

`rc` job `0fc0bd6e-754e-43d6-9210-a9dfb4075c41` on `thor:gpu0`, tree
`0ecff9e9b`, `Release`, sm_110, nvcc 13.0.88.

`tests/vt/test_ops_attention_cross.cpp` **19 test cases, 19 passed, 0 failed,
142 assertions, 142 passed, `Status: SUCCESS!`**.

The benign cases sit where they always did — `cuda max|diff|` against the f64
reference is **6.6e-07 to 1.3e-06** across the blocked geometries, inside the
op's committed `2e-5` f32 band with more than an order of magnitude to spare, and
the two arms agree with each other to **2.15e-06**.

The cancellation case is the one that had to be measured, and its answer is the
opposite of what a reordering usually gives:

| seed | blocked max\|diff\| | shipped max\|diff\| | ratio |
|---:|---:|---:|---:|
| 251 | 8.61287e-05 | 8.50797e-04 | 0.1012 |
| 263 | 7.06390e-05 | 5.17908e-04 | 0.1364 |
| 271 | 9.10163e-05 | 6.51047e-04 | 0.1398 |
| 281 | 6.71158e-05 | 7.78854e-04 | 0.0862 |
| **worst** | **9.10163e-05** | **8.50797e-04** | **0.1398** |

**The blocked kernel is 7.2x MORE accurate than the one it replaces** on
ill-conditioned data, and all four seeds agree to within a factor of 1.6, so this
is not one sample.

**Why, and it is worth stating because it inverts the intuition.** The head-dim
sum did get worse: 64 sequential f32 adds against a 32-lane pairwise butterfly.
But that is not where this kernel's error lives. The shipped kernel advances the
online-softmax recurrence **once per key** — 96 dependent rescales of `acc` and
`l` at this geometry — and each one rounds. The blocked kernel advances it once
per 32-key tile, which is **three**. Trading 96 rescales for 3 buys far more than
the pairwise-to-sequential change costs. The same property is what makes it fast,
so speed and accuracy move together here rather than against each other.

**The bounds follow from that.** The absolute bound `kCancellationTol = 2e-3` is
set on the BINDING arm, which is the shipped kernel at 8.51e-04 with 2.35x of
margin, not on the better one. The ratio bound is **2.0** rather than the
measured 0.14 plus a margin, because what it exists to catch is a DEGRADATION —
the blocked kernel must not become less accurate than the kernel it replaces —
and 2.0 gives that claim 14x of headroom while still failing on any real
inversion.

**The case is not a mute switch, and that is asserted rather than argued.** It
requires the SHIPPED kernel to miss the f64 reference by more than the benign
`2e-5` band; it misses by 8.51e-04, which is **42x** above it. A first draft of
this case used the natural `1/sqrt(D)` scale, and at that scale the softmax is a
one-hot, both kernels return one row of V exactly, and BOTH reported
`max|diff| = 0` — a case that passed while measuring nothing. Its own
ill-conditioning assertion is what caught that.

**MiniMax-Music3's acoustic suite** ran in the same job: **36 test cases, 36
passed, 353 assertions, 353 passed, `Status: SUCCESS!`**.


## 6.5 The RENDER is not bit-identical, and no gate in this tree bounds that

Said here rather than left to be noticed in a diff. The two arms produce audio
of identical length and different bytes -- `576cd9fd77e7f8ed14` against
`61a8989763bba749ed`, both 3 530 796 bytes -- because the kernel is not
bit-identical and never claimed to be.

**What IS gated** is the DiT forward, at MiniMax-Music3's committed
`kDitRelTol` / `kDitAbsFloor` / `kDitMeanAbsTol` (§14.4), unwidened, and the op
itself against an independent f64 reference at `2e-5` f32. Both pass.

**What is NOT gated, by anything, and was not before this row either**, is the
ACCUMULATED difference over a 30-step denoise. A per-forward tolerance does not
bound a fixed point reached by iterating that forward, and this tree has no
render-level oracle for MiniMax-Music3 to bound it against -- vLLM and vLLM-Omni
register nothing for this architecture (§21.9).

One thing keeps that from being a hidden risk rather than a stated one: the
direction is MEASURED, and it is TOWARD the reference rather than away. §6.3
shows the blocked kernel 7.2x closer to f64 on the ill-conditioned case, because
it trades 96 per-key softmax rescales for 3 per-tile ones.

**What was NOT checked, and is not claimed:** nothing in this row inspected the
SAMPLES. Both renders are 3 530 796 bytes and both processes exited 0, and that
is the whole of it -- a byte count and an exit code are not evidence that audio
was produced, which is the mistake the LTX-2.5 render row records. The follow-up
job computes a peak and an RMS per arm so the next revision of this section can
say something about the waveform instead of about its length.

A render-level bound is a real gap and it is NOT this row's to close; it predates
this row and applies equally to the kernel being replaced.

## 7. Owed

- **[#1551](https://github.com/mudler/vllm.cpp/issues/1551)** already owns the
  bf16 tensor-core arm. Nothing here reaches tensor cores and nothing here
  claims to.
- **The head_dim 128 tiling is the one the shared budget forces, not the one the
  arithmetic wants.** 2.21x against 7.62x. Thor reports 232 448 B of shared
  memory per block WITH opt-in against the 48 KiB used here, and
  `src/vt/cuda/cuda_paged_attn.cu` already has a cached
  `cudaDevAttrMaxSharedMemoryPerBlockOptin` seam. Not taken here because it
  widens the change and because the row's measured target is head_dim 64.
- **`BR = 32` at head_dim 64 — vLLM's own `BLOCK_M` — is UNMEASURED.** `BR = 64`
  was taken on a shared-budget argument and it wins 7.62x, but the deviation from
  the oracle's tile is not justified by a measurement.
- **The base-2 softmax of §4.2 is not mirrored.** One FMA per exponential, and it
  would require the bias to be pre-multiplied too.
- **`Tq < 64` at head_dim 64 keeps the warp-per-query kernel** and therefore
  keeps its cost. That regime is not measured by this row.
- **[#1584](https://github.com/mudler/vllm.cpp/issues/1584) — the provider seam
  double-counts the FIRST decline of every process.** `GetOpFallback` increments
  `OpProviderStats::declines` itself (`op_provider.cpp:709`) and the caching
  pattern `op_provider.h` prescribes calls `NoteOpDecline` beside it, so the
  header's "stays exact" is one too high, once. Found by this row's own audit and
  **worked around here rather than fixed**: the suite warms the fallback static
  outside every counted window, so its routing assertions are order-independent
  instead of depending on which case declined first. The seam fix changes
  semantics for four backends and for the callers that use `GetOpFallback`
  WITHOUT the caching pattern, so it needs its own row.
- **The op does not check that `key.dtype` and `value.dtype` equal
  `query.dtype`.** `Tensor::Ptr<T>` is an unchecked `static_cast`, so an f32
  query against a bf16 key reads twice the buffer. This is a contract-level
  exposure of the op that `LaunchAttentionCross` has identically, so it predates
  this row and is not introduced by it — recorded because the audit surfaced it
  and nobody had written it down.
- **[#1131](https://github.com/mudler/vllm.cpp/issues/1131) is NOT closed by this
  row.** It remains open on the DiT device arm.

## 7a. Reachability — a POSITIVE production observation, not a mutation

`.agents/reachability.md` asks two questions and they are not the same question.

**Does a production entry point reach this?** Yes, and it was observed rather
than traced. `examples/minimax-music3-gen` is an ABI client; it loads the real
28 517 617 303-byte checkpoint from local disk and runs `--device 1` at its
DEFAULT configuration, and with `VT_OP_PROVIDER_STATS=1` the engine itself
printed, on `thor:gpu0`, `rc` job `0fc0bd6e-754e-43d6-9210-a9dfb4075c41`:

```text
[vt op-provider] op=19 device=1 selected=vt-cross-blocked priority=10 registered=2
```

`OpId` 19 **is** `kAttentionCross` — the enum position, counted, not assumed.
`registered=2` is both providers; every other op in the same run reads
`registered=1`. The same binary with `VT_OP_PROVIDER_DISABLE=vt-cross-blocked`
printed `selected=vt-native priority=0 registered=2`, so the control is
two-sided and neither half is an inference.

**This row's shape is not the `tp`-handle shape, and the difference matters.**
The op already had five production call sites before this change; what is new is
a provider UNDER an existing call site. So the question "is the call site
reached" was already answered, and the question that could still fail — "does a
production run SELECT the new provider" — is the one the line above answers, in
the affirmative, on the real model.

**Does a test enter through it?** The op's own suite asserts the routing
two-sidedly through `GetOpProviderStats(...).declines` on every geometry, and
the mutation suite (§8) removes the routing and shows the gate go red. The
mutation is what proves the gate has teeth; the announce line above is what
proves a user arrives there.

**#1131 is NOT closed by this row.** It stays open on the DiT device arm, and
this row's evidence is about which kernel serves an op, not about that.

## 8. Gates, and their state

### 8.1 The shipped configuration, MEASURED

`rc` job on `thor:gpu0`, worker `rc-worker-m4d7t`, boot id
**`fabedc13-97a1-4cb9-909f-217a425d3f70`** — a DIFFERENT boot from §6.3's
`c99b7805`, so nothing here is divided by anything there. Tree `a04a4d2b1`,
asserted equal to the requested sha before anything was built. Checkpoint staged:
`STAGE_SECONDS=996`, `SRC_BYTES = DST_BYTES = 28517617303`, `findmnt -T /tmp/ckpt`
-> `overlay overlay /`. Correctness ran FIRST and green
(`BASELINE_TEST_RC=0`) before any figure below was taken.

**`--duration 20 --steps 30 --device 1 --seed 7`, spans OFF, one alternated pair
— the developer's own configuration:**

| | arm A `vt-cross-blocked` | arm B `vt-native` | ratio |
|---|---:|---:|---:|
| `denoise.dit_device` (120 calls) | **225.352 s** | **370.955 s** | **1.6461x** |
| DiT share of the run | 50.20 % | 62.37 % | — |
| **wall** | **449.969 s** | **595.496 s** | **1.3234x** |

**The DiT-bucket ratio is 1.6461x at 30 steps and 1.6461x at 2 steps** (§6.3's
three pairs), on different boots and over a fifteen-fold longer run. §21.10's
bound was 1.71x on the DiT from a flop ratio; the delivered 1.6461x is 96 % of
it.

**Two independent cross-checks the row did not arrange.** Arm B's 370.955 s
against §20.5's separately measured **370.556 s** is **0.11 %** apart, and the
`dit.attn` span below reproduces §21.9's 11.010 s to **0.22 %** — both on a
different boot and a different tree. Read as evidence that the quantity is
stable, not as a licence to divide across jobs.

**The projection §6.3 carried was right and is now retired.** It predicted a
449-453 s wall; the measurement is 449.969 s, **0.52 %** off. The 62 %-of-run
figure is no longer a projection.

### 8.2 The attribution — `dit.attn` itself

One 2-step pair with `VLLM_CPP_MUSIC3_DIT_SPANS=1`. **NOT a headline**: §21
measures the spans at 1.49 % perturbation, and `denoise.dit_device` reads 15.392
and 25.167 here against 15.046 and 24.770 without them.

| span | arm A | arm B | ratio |
|---|---:|---:|---:|
| **`dit.attn`** (576 calls) | **1.282 s** | **11.034 s** | **8.607x** |
| `dit.attn_out` | 0.918 | 0.931 | 1.014 |

`vt::AttentionCross` falls from **43.8 % of the DiT forward to 8.3 %**, and from
4.44 % of the whole run to 0.54 %. The standalone probe predicted 7.62x on
synthetic uniform data; in situ it is **8.607x**, so the probe understated it.
`dit.attn_out` moving 1.4 % is the control: the GEMM beside it did not change.

### 8.3 The audio, checked as SAMPLES rather than as a byte count

§6.5 withdrew a claim about audio nobody had inspected. This is the measurement
that replaces it — peak, RMS, non-zero fraction and clipped count over all
1 765 376 samples of each 30-step render:

| arm | samples | peak | RMS | non-zero | clipped | sha256 |
|---|---:|---:|---:|---:|---:|---|
| A blocked | 1 765 376 | 20 748 | 2 344.3 | 0.9998 | 0 | `572f8880d2b0c30a` |
| B shipped | 1 765 376 | 20 748 | 2 344.3 | 0.9998 | 0 | `55856deb3b5b727a` |

**Every waveform statistic is identical and the hashes differ.** So the render is
real audio rather than silence, nothing clips, and the two arms differ only where
a non-bit-identical kernel must make them differ. §6.5's gap — that nothing
bounds the ACCUMULATED difference over 30 steps — is unchanged by this; four
summary statistics are not such a bound.

### 8.4 The mutation suite — four clean reds, and TWO INSTRUMENT FAILURES

Every mutation is one line, rebuilt incrementally and re-gated, with the compile
return code printed BEFORE any test result and the tree restored byte for byte
afterwards.

| mutation | hunks | compile rc | test rc | verdict |
|---|---:|---:|---:|---|
| M1 the shape gate routes nothing | **0** | — | — | **INSTRUMENT FAILED — never applied** |
| M2 `vt-native` outranks the provider | 1 | **1** | — | **INSTRUMENT FAILED — did not build** |
| M3 the ragged key-tile tail enters the softmax | 1 | 0 | **1** | **RED** |
| M4 every query reads bias row 0 | 1 | 0 | **1** | **RED** |
| M5 the kv-head is the query head | 1 | 0 | **1** | **RED** |
| M6 no rescale at a tile boundary | 1 | 0 | **1** | **RED** |

Each red carries a DISTINCT binary sha256, so the rebuild demonstrably took, and
each reports `RESTORED_BYTE_FOR_BYTE`. `TREE_CLEAN` and `FINAL_TEST_RC=0`
afterwards.

**The two failures are the part worth keeping.** Neither produced a verdict about
the code and neither could be mistaken for one, because the harness counts diff
hunks and prints the compile return code first. M1 was the only mutation whose
replacement text contained an unmatched `{`, and perl's `s{...}{...}` counts
nested braces, so it consumed the closing delimiter and substituted nothing — a
pattern that silently matches nothing is EXACTLY the shape that reads as a
passing test. M2 replaced `kBlockedPriority` with a literal, which orphaned that
`constexpr` in an anonymous namespace, and `-Wunused-const-variable` is an error
here. Both are re-run in §8.5, where both are RED. M1 needed a third form as well,
for a third distinct instrument reason.

### 8.5 M1 and M2, re-run — both RED, and M1 took three attempts

Two further leases, tree `22042caba` (byte-identical to the head under review in
`cuda_attention_cross.cu` and `test_ops_attention_cross.cpp`, so the whole
mutation suite sits at one tree). The third used the cached build,
`REUSE_CACHED_BUILD=yes`, and both opened green: `BASELINE_TEST_RC=0`,
**20 cases / 156 assertions / `Status: SUCCESS!`**.

| mutation | hunks | compile rc | test rc | cases red | assertions red |
|---|---:|---:|---:|---:|---:|
| **M1** the shape gate routes nothing | 1 | 0 | **1** | **8 of 20** | 8 of 137 |
| **M2** `vt-native` outranks the provider | 1 | 0 | **1** | **6 of 20** | 7 of 156 |

Both restored `RESTORED_BYTE_FOR_BYTE`, `TREE_CLEAN`, and the tree re-gated
`FINAL_TEST_RC=0` at **20 / 156 / `SUCCESS!`** afterwards. Each carries a
distinct binary sha256 (`21987a98…`, `7b216199…`), so the rebuild demonstrably
took.

**M1 IS THE REACHABILITY MUTATION and this is its evidence.** The diff is two
lines, printed by the harness and read rather than assumed:

```text
-  if (d == 64) return tq >= 64;
-  if (d == 128) return tq >= 32;
+  if (d == 64) return tq < 0;
+  if (d == 128) return tq < 0;
```

The blocked kernel is still registered, still linked and still selected — and it
now serves nothing. Every routing assertion in the suite inverts, in the same
direction, with the same value: `declines` reads **1 where 0 is required**, at
eight separate sites including a `REQUIRE` that aborts its case. A gate that
stayed green here would be measuring a class rather than a capability, which is
`.agents/reachability.md`'s whole subject.

**M2 fails the complementary way.** With the priority inverted the provider is
registered but not selected, so `OpProviderNameAt(…, 0)` reads `vt-native`,
`last_selected` reads `vt-native` at four sites, and the two decline counts that
should read 1 read 0 instead. The pair therefore brackets the routing claim from
both sides: M1 breaks selection-does-something, M2 breaks selection-happens.

**M1 FAILED TWICE AS AN INSTRUMENT BEFORE IT FAILED AS A TEST, and never once
produced a wrong verdict.** Pass 1: `MUTATION_HUNKS=0`, never applied — perl's
`s{...}{...}` counts nested braces and M1's replacement was the only one carrying
an unmatched `{`, so it consumed the closing delimiter and substituted nothing.
Pass 2: `MUTATION_COMPILE_RC=1` — `return false` orphaned `tq`, and this build
runs `-Werror=all-warnings`, so `variable "tq" was declared but never referenced`
failed the compile. Pass 3 answers false through `tq < 0`, which is always false
for a shape and keeps the variable read.

Both failures are the exact shapes this repository records as reading like a
passing test, and neither could here, because the harness counts diff hunks and
prints the compile return code **before** any test output. That ordering is not
decoration: with it, a broken mutation is visibly broken; without it, M1 would
have been reported twice as "the gate detects this" while nothing had been
mutated at all.


### 8.6 The consumer suites, on the device

All six built and run in the same lease, at the tree under review:

| suite | rc |
|---|---:|
| `test_ops_attention_cross` | 0 |
| `test_ltx2_device` | 0 |
| `test_ltx2` | 0 |
| `test_ops_attention` | 0 |
| `test_minimax_music3_acoustic` | 0 |
| `test_minimax_music3_ar` | 0 |
| `test_music3_profile` | 0 |

`test_ltx2` and `test_ltx2_device` are green and BYTE-IDENTICAL for the reason
§6.4 gives: their fixtures are reduced-dimension and decline. That is reported as
a coverage finding, not as coverage.


## 9. Stop conditions

- **Report `NEEDS_DECISION` if the mechanism is not what #1555 guessed.** It is
  not, and §3 records it. The direction #1555 proposed — "amortize the reduction
  … or give a warp a query TILE the way FlashAttention-2 does" — is nevertheless
  the correct one, and the restructure is what the measurement endorses, so the
  row continues rather than stopping.
- **Report `NEEDS_DECISION` if mirroring vLLM would mean a larger restructure
  than one row can carry.** It does not: §4.2 is a table of correspondences and
  the port fits in one kernel in the op's existing TU.
- **STOP and report if the blocked kernel cannot hold both consumers' existing
  tolerances with their existing margins.** #1555 makes that admissibility
  explicit and this row does not widen a band to fit.
- **STOP if a speed number cannot be taken on `thor:gpu0`.** Every MiniMax-Music3
  baseline is Thor sm_110 and another box is not a denominator for it.
