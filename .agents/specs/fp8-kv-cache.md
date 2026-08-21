# fp8 KV cache (`cache_dtype=fp8*`) — spike + W1 + W2 (`KV-FP8`, `QUANT-KV-FP8`)

Rows: `KV-FP8` (engine-matrix, KV cache and memory) and `QUANT-KV-FP8`
(quantization-matrix). HIGH-priority feature gap #5
([vllm-feature-gap-analysis.md](vllm-feature-gap-analysis.md)): the standard
memory/throughput lever that halves the KV footprint by storing K/V as fp8 with
a per-tensor dequant scale. Claim `CLAIM-KV-FP8`.

Pinned oracle vLLM `555967922` (0.26.0.dev0) at `/home/mudler/_git/vllm`; every
anchor below is `file:line` in that tree. REUSE the landed fp8-e4m3 codecs (no
re-port).

## Scope

- **In (this spike + W1):** the vLLM fp8-KV surface (the `cache_dtype`
  config, `BaseKVCacheMethod` k/v scale handling, the fp8 store in
  `reshape_and_cache*`, the fp8 dequant on the attention read, the calibrated
  vs checkpoint scale, and the halved-block memory accounting); and the first
  CPU-buildable brick: an fp8-e4m3 K/V **store** (`vt::ReshapeAndCacheFp8`) +
  the fp8 **read** dequant in CPU paged attention + the `cache_dtype` config
  parse (`vllm::v1::ParseCacheDType`), all unit-gated RED-first.
- **In (W2, `## W2 — the CUDA arm` below):** the CUDA fp8-e4m3 K/V store kernel
  and the fp8 dequant on the CUDA paged-attention read, gated for parity against
  the W1 CPU reference.
- **Out (named later bricks):** fp8_e5m2 compute on either backend,
  per-attention-head scales, the Metal and ROCm fp8-KV arms (both refuse by name
  — see `## W2` below), the full engine-runner integration
  (half-sized KV blocks in the real runner + checkpoint `k_scale`/`v_scale`
  threading + `--kv-cache-dtype`/`--calculate-kv-scales` CLI), and the vendor
  KV dtypes (`fp8_inc`, `fp8_ds_mla` — `QUANT-KV-FP8-VENDOR`) and turboquant /
  nvfp4 / per-token-head KV (`KV-NVFP4-TURBO`).

## Upstream chain

The vLLM fp8-KV path is NOT delegated to a dependency (unlike the GEMMs) — the
store/read kernels are vLLM's own csrc:

- **Config.** `CacheDType` Literal (`vllm/config/cache.py:19-36`) —
  `auto`/`fp8`/`fp8_e4m3`/`fp8_e5m2` (+ `fp8_inc`/`fp8_ds_mla`/`turboquant_*`/
  `int*_per_token_head`/`nvfp4`); `cache_dtype: CacheDType = "auto"`
  (`:76`, "if auto, use model dtype; fp8 == fp8_e4m3"); `calculate_kv_scales`
  (`:99-104`, dynamic on-the-fly k/v scale when the checkpoint has none).
  `is_quantized_kv_cache` (`vllm/utils/torch_utils.py`) = `cache_dtype != "auto"`.
- **Scale handling.** `BaseKVCacheMethod` (`kv_cache.py:42`) adds `_k_scale`/
  `_v_scale`/`_q_scale`/`_prob_scale` to the Attention layer;
  `process_weights_after_loading` (`:108-191`) loads per-tensor scales from the
  checkpoint (`k_scale`/`v_scale`), enforces **per-tensor only** ("Only support
  per-tensor scaling factor for fp8 KV cache", `:145-148`), defaults to 1.0 when
  absent (with the uncalibrated warning `:169-174`), and duplicates a single
  `kv_scale` to both. `KVCacheScaleParameter` (`:17-38`) is the scalar `()`/`(1,)`
  loader.
- **Store.** `reshape_and_cache_flash` (`csrc/libtorch_stable/cache_kernels.cu:
  746`) → `reshape_and_cache_flash_kernel` (`:314-401`); the fp8 branch is
  `CopyWithScaleOp` (`:241-252`), `dst = fp8::scaled_convert<cache_t, scalar_t,
  kv_dt>(src, scale)`. The scale convention is documented at
  `csrc/quantization/w8a8/fp8/nvidia/quant_utils.cuh:296-300`:
  **`FP8 = Quantize(HP / scale)`; `Dequant(FP8) * scale = HP`.** `k_scale`/
  `v_scale` are `[1]` (per-tensor) or `[num_heads]` (per-head, `kv_scale_stride`,
  `:365-401`). Store dtype `cache_t = uint8_t`; the fp8 *interpretation* is the
  `Fp8KVCacheDataType` template param (`csrc/attention/dtype_fp8.cuh:15-19`).
- **Read (dequant).** The fp8 attention read multiplies back by the scale:
  `scaled_vec_conversion<float, uint8_t>` (`quant_utils.cuh:419-429`) =
  `half_to_float(fp8_to_half(byte)) * scale`. Consumed by the FA/flashinfer fp8
  paths and the reference `test_cache.py`'s `convert_fp8`.
- **Memory accounting.** An fp8 KV element is 1 byte vs bf16's 2 → the KV block
  is half-sized; `AttentionSpec.page_size_bytes` (`kv_cache_interface.py:380-398`)
  derives from the storage dtype, so `num_gpu_blocks` (profiled) roughly doubles.

## Our baseline

The vt runtime already has: `vt::ReshapeAndCache` + the CPU paged-attention read
(`src/vt/cpu/cpu_cache.cpp`, `src/vt/cpu/cpu_paged_attn.cpp`) over the NHD
unbind-slice cache; the fp8-e4m3 codecs, landed twice and bit-identical —
`vllm::F8E4M3ToF32`/`F32ToF8E4M3`
(`src/vllm/model_executor/model_loader/nvfp4_dequant.cpp:11`,
`src/vllm/model_executor/layers/quantization/compressed_tensors/nvfp4_emulation.cpp:14`)
and the vt-layer in-file copies `Fp8ToF32`/`F32ToFp8` (`src/vt/cpu/cpu_ops.cpp:
418-460`, written because "vt does not depend on vllm"); and the storage-dtype
resolver seam `ResolveKvCacheDType` (`include/vllm/v1/kv_cache_dtype.h`). W1
adds the canonical vt-layer fp8-KV codec home (`include/vt/fp8_kv.h`, reusing —
not re-porting — that e4m3 math, and the future consolidation target for the two
existing copies), the fp8 store op, the read dequant, and the config parse.

## Port map

W1 (this change; CPU-only, `-Werror`):
- `include/vt/fp8_kv.h` (NEW) — `Fp8KVCacheDataType` enum (mirror
  `dtype_fp8.cuh:15-19`) + `F8E4M3ToF32`/`F32ToF8E4M3`/`StoreKvFp8E4M3`/
  `LoadKvFp8E4M3` (bit-match the landed codecs; the store/load scale convention
  from `quant_utils.cuh:296-300`).
- `include/vt/ops.h` — `OpId::kReshapeAndCacheFp8`, `ReshapeAndCacheFp8Fn`,
  `vt::ReshapeAndCacheFp8` decl, and the additive `PagedAttentionArgs`
  `kv_cache_dtype`/`k_scale`/`v_scale` fields (default kAuto/1.0 → every existing
  caller byte-identical).
- `src/vt/ops.cpp` — the `ReshapeAndCacheFp8` wrapper (validate + dispatch) and
  the `PagedAttention` fp8 dtype guard (kI8 cache + scales when != kAuto).
- `src/vt/cpu/cpu_cache.cpp` — `ReshapeAndCacheFp8Kernel` (store =
  `Quantize(hp / scale)`; mirror `reshape_and_cache_flash_kernel` fp8 branch).
- `src/vt/cpu/cpu_paged_attn.cpp` — the read-side dequant
  (`Dequant(fp8) * k_scale|v_scale`) when `args.kv_cache_dtype != kAuto`.
- `include/vllm/v1/kv_cache_dtype.h` — `ParseCacheDType` + `IsQuantizedKvCache`
  (mirror `CacheDType` + `is_quantized_kv_cache`).
- `tests/vt/test_ops_fp8_kv_cache.cpp` (NEW) + its `tests/CMakeLists.txt` line.

W2 (`## W2 — the CUDA arm` below): `src/vt/cuda/cuda_cache.cu` (the fp8 store
kernel + its kCUDA registration), `src/vt/cuda/cuda_paged_attn.cu` (`LoadKv`,
the two scale parameters on `PagedAttentionKernel`/`PagedFlashKernel`,
`LaunchPagedBlock`, `LaunchPagedFp8`), `src/vt/ops.cpp` (the device-class guards
replaced by provider routing plus a named Metal/ROCm refusal), and
`tests/vt/test_cuda_fp8_kv_cache.cpp` (NEW) + its `tests/CMakeLists.txt` line.

Later bricks: the runner/spec integration (half-sized blocks + checkpoint scale
threading + CLI); fp8_e5m2 compute; per-head scales; the Metal and ROCm arms.

## Tests to port

- `vllm/tests/kernels/attention/test_cache.py::test_reshape_and_cache` (the
  `kv_cache_dtype == "fp8"` branch, `:97-165`) → the store→dequant round-trip
  within `atol=0.001, rtol=0.1` — LANDED as `test_ops_fp8_kv_cache.cpp`
  (round-trip + fp8-vs-bf16 baseline + the paged-attention e2e). The `"fp8"`
  parametrization also covers e5m2 in `KV_CACHE_DTYPE`; the e5m2 half is
  SKIPPED-with-reason (refused as a later brick) by the `refuses e5m2` case.
- `vllm/tests/kernels/attention/test_cache.py::test_reshape_and_cache_flash`
  (the flash-layout fp8 + per-head/nvfp4 cases, `:180-260`) — the per-head and
  nvfp4 scale-type arms are later bricks (per-tensor is W1); named here.

## Gates

- **Correctness (W1, CPU):** `test_ops_fp8_kv_cache` — 8 cases / 511 assertions
  GREEN: store→read within the e4m3 band, fp8 tracks the bf16 KV baseline
  (NMSE < 1%), paged-attention over the fp8 cache matches the bf16-cache output
  within 5%, `ParseCacheDType` mirrors the CacheDType surface. RED-first proven:
  a wrong store direction (`hp * scale`) fails 3 cases / 480 assertions; a wrong
  read `v_scale` diverges > 0.05 from the baseline; an auto (no-dequant) read of
  an fp8 cache is refused. No sibling regressions (reshape 12/12, paged 14/14).
- **Correctness (W2, provider routing — the CPU leg):** `test_cuda_fp8_kv_cache`
  7 cases / 10 assertions GREEN on a CPU-only build. Only G1 and G1b assert
  there; the four device cases skip with a MESSAGE naming what did not run.
  RED-first proven: with the W1 device-class guards in place the suite reports
  10 assertions / 6 failed for the store guard plus the read guard, naming both
  refusal strings.
- **Correctness (W2, device — UNEXECUTED, see `## Owed`):** the store byte gate
  (zero tolerance; f32, bf16 and f16 sources; a padded slot), the paged-read
  parity gate in both the f32 and the bf16 query/output instantiation (decode +
  prefill), the CUDA-kernel e5m2 refusal and the registration gate all need a
  device. **The CUDA translation units DO compile** — CI job `cuda-fat-build`
  builds them for ten architectures under `-Werror=all-warnings` — but that job
  configures `-DVLLM_CPP_BUILD_TESTS=OFF`, so nothing in this file has ever
  been executed on a device.
- **Later:** the real memory-halving e2e (KV blocks ~2× on a gate model at token
  parity) is the binding gate and is DGX-blocked (docs/BENCHMARKS PENDING).

## Dependencies

None to land W1 (reuses the vt-layer fp8-e4m3 math + the existing NHD paged
cache). Downstream: the CUDA kernels need the GB10 lane; the memory-halving e2e
needs the runner/spec integration + a gate model; per-head scales and the
vendor/turbo/nvfp4 KV dtypes are separate rows.

## Work breakdown

| Brick | Content | State |
|---|---|---|
| W0 | this spike | DONE (this commit) |
| W1 | CPU fp8-e4m3 store + read dequant + config parse + unit gate | DONE (this commit) |
| W2 | CUDA fp8-e4m3 store + fp8 paged-attention read (parity vs W1) | DONE (code + gate landed; the DEVICE cases are UNEXECUTED — see `## Owed`) |
| W3 | runner/spec integration: half-sized KV blocks + checkpoint k/v_scale threading + `--kv-cache-dtype`/`--calculate-kv-scales` | later |
| W4 | memory-halving e2e on a gate model (the binding gate, DGX) | later |
| W5 | fp8_e5m2 CPU+CUDA compute; per-attention-head scales | later |

## W2 — the CUDA arm (#1593)

Issue: [#1593](https://github.com/mudler/vllm.cpp/issues/1593). W1 is the
ORACLE: every W2 gate compares CUDA to the landed CPU kernels, never to a fresh
reference.

**Store** (`src/vt/cuda/cuda_cache.cu`, `ReshapeAndCacheFp8KernelCuda` +
`ReshapeAndCacheFp8Kernel<Tin>`, registered for `DeviceType::kCUDA`). An
ELEMENTWISE-IDENTICAL port of the fp8 branch of `reshape_and_cache_flash_kernel`
(`cache_kernels.cu:314-401`) + `CopyWithScaleOp` (`:241-252`), restricted to
upstream's `is_contiguous_heads && kv_scale_stride == 0` arm (`:352-366`) —
which is the only arm the op's wrapper admits, because the vt cache is the NHD
unbind slice and `ReshapeAndCacheFp8` takes two scalar scales. It is NOT
instruction-identical, and calling it a 1:1 port would overstate it: upstream's
contiguous-heads arm moves the row through `vectorize_with_alignment<VEC_SIZE>`
(`:360-363`, `VEC_SIZE` 8 for a 2-byte source and 4 for f32), and ours is a
scalar strided loop over the same elements in the same order. Same bytes out,
fewer bytes per instruction; W4 owns closing the bandwidth gap. The converter is
upstream's own `__nv_cvt_float_to_fp8(hp / scale, __NV_SATFINITE, __NV_E4M3)`
(`quant_utils.cuh:497-503`) — a true DIVIDE, not the activation path's hoisted
reciprocal multiply — and its byte-for-byte equality to the CPU software codec
`vt::F32ToF8E4M3` is already MEASURED at zero tolerance on sm_110 and sm_121a
([vt-fp8-quant-arch-gate.md](vt-fp8-quant-arch-gate.md) G2). Source dtypes
f32/f16/bf16, the same set the CPU `LoadSrcF32` serves.

**Read** (`src/vt/cuda/cuda_paged_attn.cu`). `LoadKv(ptr, i, scale)` joins
`Load`: inert on the f32/bf16 arms (they forward to `Load` unchanged, so every
existing caller reads the same bytes in the same order), and on `uint8_t` it is
`Fp8E4M3ToF32Dev(byte) * scale` — upstream's `scaled_vec_conversion<float,
uint8_t>` (`quant_utils.cuh:419-429`), written as the SAME ARITHMETIC as
`vt::F8E4M3ToF32` so CUDA==CPU on the read is a property of the source rather
than of a measurement this session could not take. `PagedAttentionKernel` and
`PagedFlashKernel` gain `k_scale`/`v_scale`; `LaunchPagedByKv` keys on
`args.kv_cache_dtype` (never on the storage dtype, which is a bare `kI8` byte)
and routes to `LaunchPagedFp8`.

Same-arithmetic holds for all 254 FINITE e4m3 codes and NOT for the two NaN
ones. On `0x7F`/`0xFF` the CPU returns `std::numeric_limits<float>::quiet_NaN()`
(`0x7FC00000`) and the device returns `CUDART_NAN_F` (`0x7FFFFFFF`): both quiet,
both propagating, different payload. **No gate in this file can see that**,
because a NaN compares unequal to itself, so a byte or NMSE comparison fails
on any payload rather than on the wrong one. It is recorded rather than
measured, and reaching it needs a non-finite input in the first place — `__NV_SATFINITE`
clamps an out-of-range magnitude to `0x7E`/`0xFE`, so a store of a finite
`hp / scale` never writes a NaN code.

`LaunchPagedFp8`/`LaunchPagedFp8Out` are templated on `TQ, Tout` over
`{float, __nv_bfloat16}`, so the fp8 read has FOUR instantiations and the gate
exercises two of them: `<float,float>` (G4) and `<__nv_bfloat16,__nv_bfloat16>`
(G4b), the one a served model takes. The two mixed-dtype instantiations are
compiled and ungated.

**Scope of the read, argued.** Only the two correctness-grade kernels serve fp8:
the tiled flash prefill and the block decode. That is what the existing ladder
already implies — the WMMA prefill kernels stage `__nv_bfloat16` fragments, the
vendored FA-2 launchers take bf16 pointers, and the vectorized decode-opt/GQA
kernels read through `LoadRowN`/`LoadRow8`, 128-bit `uint4` loads specialized
for bf16 and f32 only. Upstream draws the same line from the other side:
FlashAttention serves a quantized KV cache only where
`flash_attn_supports_kv_cache_dtype` says so (`flash_attn.py:181-187,796-805`).
A tensor-core fp8 read is a PERFORMANCE brick; W2's gate is parity, and W4 owns
the memory/throughput measurement.

**The device-class guards are gone, but not the refusal.** W1 hard-refused every
non-CPU queue inside both op wrappers, before provider lookup — that is what kept
the CUDA arm unreachable. The STORE now resolves through the provider table like
every other op, because `kReshapeAndCacheFp8` is its own `OpId` that only CPU and
CUDA register, so an unimplemented backend refuses BY NAME inside `GetOp`. The
READ cannot: it rides ADDITIVE fields on `PagedAttentionArgs` of an op `kMETAL`
and `kROCM` already register for the FLOAT path, and nothing in the provider
table can tell the two arms apart, so an fp8 cache would reach a float kernel and
return silent garbage. `src/vt/ops.cpp` therefore keeps an explicit CPU-or-CUDA
list there whose message names the missing part, and
`tests/vt/test_cuda_fp8_kv_cache.cpp` gates it on both `kMETAL` and `kROCM`.

**Where a refusal is asserted decides what it proves.** The e5m2 refusal exists
in THREE places — the `ReshapeAndCacheFp8` op wrapper (`src/vt/ops.cpp`), the
CPU kernel (`src/vt/cpu/cpu_cache.cpp`) and `ReshapeAndCacheFp8KernelCuda` — and
only the third is a CUDA guarantee. The wrapper check is device-independent and
sits ABOVE both the device checks and `GetOp`, so a case that calls
`vt::ReshapeAndCacheFp8` with device tensors and asserts a bare throw cannot
distinguish any of the three. G5 was written that way. MEASURED, not argued:
deleting the CUDA kernel's `VT_CHECK` on a CPU build produces `ninja: no work to
do` and leaves the file 7 cases / 10 assertions `SUCCESS!`; deleting the op
wrapper's leaves W1's `test_ops_fp8_kv_cache` GREEN at 8/511, because execution
then falls through to the CPU kernel's own check and W1's case asserts
`CHECK_THROWS_AS(..., std::runtime_error)` rather than a message. Only deleting
the wrapper AND the CPU kernel check together turns W1 red (7/8 cases, 510/511
assertions). So what W1 pins is "e5m2 is refused somewhere on the CPU path", and
a layered refusal needs an assertion that NAMES its layer.

G5 is now written the only way that reaches the kernel guard: resolve the
registered provider with `GetOp(OpId::kReshapeAndCacheFp8, DeviceType::kCUDA)`,
call it directly, and require the message to contain both
`cuda reshape_and_cache_fp8` and `fp8_e5m2`, which no other layer produces. It
then calls the same pointer with e4m3 to prove the guard refuses one kind rather
than disabling the kernel. Bypassing the wrapper is deliberate and is the point
of the case; the production path still goes through it, and G1/G1b/G2 gate that.

## Owed

- **The W2 device gates are UNEXECUTED** (#1593). `tests/vt/test_cuda_fp8_kv_cache.cpp`
  G2 (provider registration, CUDA build), G3 (store byte parity over the f32,
  bf16 and f16 sources), G4 and G4b (paged-read parity, decode + prefill, for
  the f32 and the bf16 query/output instantiation) and G5 (the CUDA kernel's own
  e5m2 refusal, reached through the registered provider) all need a device.
  Neither the implementing session nor the repair session had one — `nvcc` is
  absent on `mudler-ubuntu-box` and the GPU fleet was leased for the #1574
  campaign — so none of them has run and none has a red-first mutation.
  **They do COMPILE.** CI job `cuda-fat-build` (`.github/workflows/ci.yml:773`)
  built both changed CUDA translation units for `80;86;87;89;90a;100a;103a;110;
  120a;121a` under `-Werror=all-warnings` and passed on `4d71e776e`
  ([run 32495320287](https://github.com/mudler/vllm.cpp/actions/runs/32495320287/job/96812232428)),
  but it configures `-DVLLM_CPP_BUILD_TESTS=OFF`, so it never links or runs this
  file. G1/G1b (provider routing and the Metal/ROCm refusal) are the only cases
  that executed, and they run on the CPU leg. The first `rc` lease that touches
  this row must run `ctest -R test_cuda_fp8_kv_cache` and record the result here
  before W2 counts as measured. Until then the wave table's `DONE` means "landed
  and gated", not "measured on hardware".
- **A CPU-only gate cannot see a CUDA defect, and this one was PROVEN blind.**
  The W2 review deleted the entire production call site for the CUDA fp8 read
  and the focused gate stayed 100% green; inverting the CUDA store's scale
  direction produced `ninja: no work to do`, because a CPU build's
  `compile_commands.json` contains ZERO `.cu` translation units — 1046 entries
  in the review's configure and 1030 in the repair session's, none of them CUDA.
  That is the honest consequence of the state above, not a defect in the gate,
  and it is why the device run is the binding evidence.
- **Three W1 read-side citations point at the WRONG upstream lines** and are
  outside the W2 change's authority to edit: `include/vt/fp8_kv.h:92`,
  `include/vt/ops.h:1129` and `src/vt/cpu/cpu_paged_attn.cpp:164` each cite
  `scaled_vec_conversion<float, uint8_t>` at `quant_utils.cuh:302-308`, which at
  pin `555967922` is the generic primary template plus the header of the
  `<uint16_t, uint8_t>` (fp8 -> half) specialization. The `<float, uint8_t>` one
  is at `:419-429`. W2 corrected its own four copies; these three are owed and
  tracked by [#1636](https://github.com/mudler/vllm.cpp/issues/1636), which also
  owes the "CUDA TUs are UNCOMPILED" clause in
  [`.agents/engine-matrix.md`](../engine-matrix.md) and
  [`.agents/quantization-matrix.md`](../quantization-matrix.md).
- **Nothing reaches the fp8 KV path from a production entry point yet**, on
  either backend. `vt::ReshapeAndCacheFp8` and `PagedAttentionArgs::kv_cache_dtype`
  have no caller outside their tests; W1 landed in that state and W2 does not
  change it. **`KV-FP8` W3 owns the wiring** — half-sized KV blocks in the runner,
  `--kv-cache-dtype` threaded from the CLI, and the checkpoint `k_scale`/`v_scale`
  path — and it is tracked by #1593 alongside W2.
- **Metal and ROCm have no fp8 KV arm.** Both refuse by name (see above). Neither
  has a row yet; they belong with W5's per-head/e5m2 work or a backend row.
- fp8_e5m2 and per-attention-head scales stay refused on both backends (W5).

## Risks/decisions

- **Storage as `DType::kI8`, interpretation as a separate enum.** vLLM's kernel
  carries `cache_t = uint8_t` + the `Fp8KVCacheDataType` template param; we mirror
  that exactly (the dtype.h "the byte never guesses its semantic type" rule,
  `include/vt/dtype.h:20-32`), which avoids adding fp8 to the `DType` enum (and
  its `-Wswitch` blast radius across every backend) while staying faithful.
- **Additive op, not a signature change.** `ReshapeAndCacheFp8` is a NEW op, not
  a widened `ReshapeAndCacheFn` — changing the shared alias would break the
  CUDA/Metal registrations' `static_cast` (touching Metal, forbidden). The read
  side rides additive default-inert `PagedAttentionArgs` fields, so every float
  caller is byte-identical.
- **Codec reuse, not a third copy.** `include/vt/fp8_kv.h` is the canonical
  vt-layer fp8-KV codec home; it is bit-identical to the landed
  `vllm::F32ToF8E4M3`/`F8E4M3ToF32` and to the `cpu_ops.cpp` in-file
  `F32ToFp8`/`Fp8ToF32`, which a later cleanup should consolidate onto it.
- **e5m2 refused, not silently mis-stored.** The config parses `fp8_e5m2` (full
  mirror of the surface) but the CPU compute refuses it with a named-later-brick
  reason — shipping an fp8-e5m2 codec unvalidated against the (offline) oracle
  would violate RED-first.
- **Honest residual.** W1 is a correctness brick; the real *memory/throughput*
  win (the point of the feature) is the GPU store/read + the halved-block runner
  integration, both DGX-blocked and named W2-W4.
