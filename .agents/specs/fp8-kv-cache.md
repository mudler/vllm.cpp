# fp8 KV cache (`cache_dtype=fp8*`) — spike + W1 + W2 + W3 (`KV-FP8`, `QUANT-KV-FP8`)

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
- **In (W3, `## W3 — the runner integration` below):** half-sized KV blocks in
  the real runner, `--kv-cache-dtype` threaded from the server flag through the
  checkpoint's own `kv_cache_quant_algo` to the block sizing, and the checkpoint
  `k_scale`/`v_scale` path with its declared-but-absent arm named rather than
  defaulted.
- **Out (named later bricks):** fp8_e5m2 compute on either backend,
  per-attention-head scales, the Metal and ROCm fp8-KV arms (both refuse by name
  — see `## W2` below), `--calculate-kv-scales` (upstream's deprecated dynamic
  scale), the C-ABI exposure of `--kv-cache-dtype`, the 16 architectures whose
  attention blocks W3 refuses rather than routes, and the vendor
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
| W3 | runner/spec integration: half-sized KV blocks + checkpoint k/v_scale threading + `--kv-cache-dtype` | DONE (code + CPU gate landed; see `## W3` and `## Owed`) |
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

## W3 — the runner integration (#1593)

Issue: [#1593](https://github.com/mudler/vllm.cpp/issues/1593), the same issue
that carries W2. W3 is what makes the fp8 KV cache a SERVED capability instead
of a pair of kernels: half-sized KV blocks in the real runner,
`--kv-cache-dtype` threaded from the flag to the block sizing, and the
checkpoint `k_scale`/`v_scale` path.

**Why it is on the critical path.** Benchmark campaign
[#1574](https://github.com/mudler/vllm.cpp/issues/1574) measures us against vLLM
and SGLang on `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121`. We already beat vLLM's AR
baseline there (11.06 vs 9.71 tok/s) — with **bf16 KV against their fp8**, so we
move twice the KV bytes for the same tokens. Until W3 the comparison was not
matched, and [#415](https://github.com/mudler/vllm.cpp/issues/415) attributes a
prefill gap to exactly this.

**What turning it on COSTS, unmeasured and therefore not claimed.** An fp8 KV
cache takes the subject OFF every fast attention kernel this engine has.
`qwen3_5.cpp` makes `kv.dtype == DType::kBF16` a term of both `fa2_prefill` and
`fa2_decode`, so FA-2 prefill and all three FA-2 decode topologies are disabled;
`src/vt/cuda/cuda_paged_attn.cu` routes an fp8 read only through tiled prefill
and block decode, because the WMMA ladder, the vendored FA-2 launchers and the
vectorized decode-opt/GQA kernels are bf16-native by construction. So the 11.06
vs 9.71 tok/s lead above was measured on the bf16 path, and matching vLLM's KV
dtype plausibly LOSES it. Nothing here has measured which way the sum goes: half
the KV bytes and twice the pool against a slower attention kernel is an
empirical question and this session had no GPU. It is recorded so that the next
`--kv-cache-dtype fp8` benchmark is read as a NEW measurement rather than as a
regression, and `docs/USAGE.md` says the same thing to an operator. Routing the
fp8 read through the fast kernels is owed below.

### The resolution chain, mirrored

Four upstream steps, in upstream's own order. Every anchor was read in
`/home/mudler/_git/vllm` at `555967922` (`git rev-parse HEAD` =
`5559679229bc961848b121ccdeaa8fa5d79bec98`).

1. **`--kv-cache-dtype` vs the checkpoint** — `resolve_kv_cache_dtype_string`
   (`vllm/utils/torch_utils.py:374-392`) + `get_kv_cache_quant_algo_string`
   (`:310-362`) + `MODELOPT_TO_VLLM_KV_CACHE_DTYPE_MAP` (`:64-67`). Ported as
   `vllm::ResolveKvCacheDTypeString` (`include/vllm/config/cache.h`,
   `src/vllm/config/cache.cpp`) and called ONCE, from
   `LoadedEngine::FromModelDir`, exactly where `EngineArgs.create_engine_config`
   calls it before constructing `CacheConfig` (`vllm/engine/arg_utils.py:
   1915-1929`). An explicit value is returned unchanged and the checkpoint is
   never consulted (`:380-381`) — the operator outranks the checkpoint, which
   `attention.py:279-290` restates in its own comment.

   **Which file the declaration is read FROM is also upstream's, and W3 first
   shipped it inverted.** `vllm/transformers_utils/config.py:751-761` takes
   `config_dict["quantization_config"]` and consults `hf_quant_config.json` only
   when that is `None`, under upstream's own comments: ModelOpt writes the
   inline document from 0.31.0 on, and the standalone file is what 0.29.0 and
   before wrote. `vllm::ReadQuantConfigJson` now reads them in that order.
   Reversed, a repository that was re-quantized in place — inline document
   added, stale `hf_quant_config.json` left beside it — resolves to the OLD
   declaration, which for the KV half means quantizing a cache nobody asked to
   quantize, at half the page, silently. G10 gates the order at both the
   resolver and the loader.
2. **String to storage dtype** — `kv_cache_dtype_str_to_dtype` (`:394-401`) over
   `STR_DTYPE_TO_TORCH_DTYPE` (`:32-52`), where every fp8 CacheDType maps to
   `torch.uint8`. W1's `vllm::v1::ParseCacheDType` already did this; W3 adds no
   parsing.
3. **Storage dtype to bytes** — `GPUModelRunner.__init__` resolves ONE
   `self.kv_cache_dtype` (`vllm/v1/worker/gpu_model_runner.py:484-486`) and every
   attention spec is built with it; `AttentionSpec.real_page_size_bytes`
   (`vllm/v1/kv_cache_interface.py:204-218`) is linear in
   `get_dtype_size(self.dtype)`. Ported as `vllm::v1::ApplyCacheDType`
   (`src/vllm/v1/kv_cache_interface.cpp`), called from
   `LoadedEngine::ApplyResolvedCacheDType` on the PROBE config **before**
   `ResolveNumBlocks` reads its geometry. That ordering is the feature: the
   probe's `KVBytesPerBlock` is the divisor knob 2 sizes the pool with, so an
   fp8 page halves the divisor and doubles the block count at the same
   `--kv-cache-memory`. Applying it afterwards would serve the same pool in half
   the bytes instead of twice the pool.
4. **The scales** — `BaseKVCacheMethod.process_weights_after_loading`
   (`vllm/model_executor/layers/quantization/kv_cache.py:74-156`), ported as
   `vllm::ResolveKvCacheScales`
   (`include/vllm/model_executor/layers/quantization/kv_cache.h`).

### The trap this checkpoint sets

**First, the fact that changes what the trap is.**
`r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121` @ `36f717a2` declares
`kv_cache_quant_algo: "FP8"` in
`hf_quant_config.json` — and that file is NOT the one that gets read.
MEASURED from the live artifact 2026-08-22: its
`config.json:quantization_config` carries `quant_method: "modelopt"`,
`quant_algo: "MIXED_PRECISION"` and **no `kv_cache_*` key at all**, and an
inline document suppresses the legacy file entirely
(`transformers_utils/config.py:751-761`, and the same order in
`vllm::ReadQuantConfigJson`). Running the pinned
`get_kv_cache_quant_algo_string` over those exact bytes returns `None`.

**So the #1574 subject auto-selects nothing, on either engine, and
`--kv-cache-dtype fp8` has to be typed on BOTH sides of the comparison.** The
competitors' own `serve.sh` already passes it, so the campaign is consistent
rather than blocked; what is wrong is any sentence that says this checkpoint
takes the declared-fp8 path by itself. G10's `#1574 checkpoint declares NOTHING`
case pins it with both real documents in one directory.

**The scale trap is real and unchanged**, and it is what the flag runs into: the
checkpoint ships **ZERO** `k_scale`/`v_scale` tensors. MEASURED 2026-08-21 from
the public `model.safetensors.index.json`: 2001 tensors, none of them named
`k_scale`, `v_scale` or `kv_scale`.

So the default scale 1.0 has to be reached DELIBERATELY, by a path that knows
the algorithm was declared and the tensors were absent — not by falling off the
end of a missing-tensor lookup. The two are indistinguishable at runtime and
produce identical output, right up to the first checkpoint that declares no
algorithm at all, at which point the accidental path silently invents a scale
for a cache nobody asked to quantize.

Upstream keeps them apart STRUCTURALLY and this port mirrors that.
`process_weights_after_loading` reaches the scale block at all only under
`is_quantized_kv_cache(layer.kv_cache_dtype)` (`kv_cache.py:100-102`); INSIDE
it, both scales still holding the `KVCacheScaleParameter` sentinel `-1.0`
(`:18-30`) is the separate "no scales were loaded" arm that takes 1.0 and warns
(`:112-116`, `:150-156`). `KvScaleOrigin` names all four arms, and
`ScalesForFp8Store` REFUSES `kNotQuantized` by name rather than answering 1.0.
`G2` gates the difference: two calls with identical numbers out, distinguished
only by `origin`, and the refusal message names `kv_cache_quant_algo`.

### The store and the read

`PagedKvCache` gains `fp8_kind`/`k_scale`/`v_scale`, carried from the layer's
own `AttentionSpec` by the runner. `dense_attn::WriteKvCache` and
`dense_attn::ApplyKvCacheQuant`
(`include/vllm/model_executor/models/kv_cache_route.h`) are the ONE place that
decides float versus fp8, and `IsFp8KvCache` refuses a view whose storage dtype
and fp8 interpretation disagree — a `kI8` page with no fp8 kind, or an fp8 kind
over a float page, is a mis-sized cache and never a mode.

**Routed in W3:** the shared seam `dense_attn::AttnBlock`
(`include/vllm/model_executor/models/dense_attn_block.h`) — reached by the
Qwen3 dense family (`qwen3.cpp:185`), Qwen3-MoE (`qwen3_moe.cpp:84`), Voxtral
(`voxtral.cpp:102`) and the Llama/Mistral/InternLM2 registries that share
`Qwen3DenseModel` — and `src/vllm/model_executor/models/qwen3_5.cpp`, the
Qwen3.5/3.8 family, which is the benchmark subject.

**The seam's routing was DEAD until the second review.** Its preamble guard
still admitted only `kBF16` and `kF32`, so `IsFp8KvCache` was false at every
call and neither fp8 arm could execute; reverting the whole of the seam's
routing left the gate at 26/26 green, because every case entered through
`Qwen3_5DenseModel::Forward`. The guard is widened the same way
`qwen3_5.cpp:5313` was, and **G12** enters an fp8 cache through
`Qwen3DenseModel::Forward` — the function `ForwardQwen3ForCausalLM` calls under
`ModelRegistry::Forward` (`qwen3_dense.cpp:113`) — so the same revert now goes
red.

**Every other architecture is refused BY NAME, and the name is usually its
OWN.** 16 architectures at 17 call sites keep their own attention preambles.
The third review traced each of them with an fp8 (`kI8`) page against a bf16
model dtype, and the split is **14 / 1 / 1**:

- **14 never reach the store.** Thirteen refuse at their own dtype guard —
  `granite:95`, `minicpm:96`, `phi3:78`, `gemma3:121`, `opt:125`,
  `stablelm:86`, `glm4:93`, `commandr:93`, `gemma:53`, `gemma2:135`, `phi:98`,
  `muse_glimmer:144` and `olmo2:94`, each saying
  `"<arch>: KV cache must be bf16 or f32"`, which names the architecture but
  neither fp8 nor the flag. `gemma4` (two sites) is the fourteenth and refuses
  EARLIER and WORSE: `gemma4.cpp:306-315` takes `kv.dtype != adt`, allocates
  `DBuf kcast(d, kv.dtype /* kI8 */, ...)`, and calls `vt::CastF32`, which
  refuses at `src/vt/ops.cpp:4087` with `"cast_f32: out must be f32"` — a
  message that names neither fp8, nor the flag, nor the architecture.
- **1 reaches the store guard.** Only `qwen3_vl:198-200` carries no guard and
  no cast, so `vt::ReshapeAndCache` is what refuses it, naming
  `vt::ReshapeAndCacheFp8` and saying the architecture is not routed for fp8 KV.
- **1 refuses with its own fp8-naming message.** `nemotron_h_device.cpp:1589-1593`
  has an explicit `else { VT_CHECK(false, "NemotronH paged forward: ... The fp8
  KV scheme the checkpoint ships k_scale/v_scale for is a SEPARATE decision with
  its own gate and is not selected here"); }` on the cast, which fires first and
  is the ONE refusal in the sixteen that tells the operator what they asked for.

Either way the failure is a sentence rather than a float path indexing a
half-sized page, which is the property that matters and which is unaffected by
the recount. What the recount changes is the message quality: the store guard's
better message is reached by 1 of the 16, not 3, and that is recorded under
`## Owed` rather than claimed away. **G7** (`test_kv_cache_fp8_wiring.cpp:910`)
hand-builds its K/V tensors and calls `vt::ReshapeAndCache` directly, so it
gates the store guard's MESSAGE and never the claim that any particular
architecture reaches it.

### Gates

`tests/vllm/entrypoints/test_kv_cache_fp8_wiring.cpp` — **31 cases / 481
assertions GREEN** on a CPU-only build, plus
`tests/vllm/entrypoints/openai/test_serve_kv_cache_dtype.cpp` — **3 cases / 26
assertions GREEN**, which drives the REAL `VllmServerMain`.

| Case | What it would let through if it were missing |
|---|---|
| G1 | the checkpoint declaration RESOLVES, an explicit flag outranks it, and the modelopt marker is exactly the three upstream can produce |
| G2 | a declared-but-absent scale collapsing into "nothing declared" |
| G3 | an fp8 page that is not EXACTLY half a bf16 page (closed form, not a ratio) |
| G4 | the same halving through the LOADER: one byte budget, 2x the blocks; and the Mamba state left alone |
| G5 | the fp8 path not being REACHED — the engine generates tokens over an fp8 cache |
| G6 | a storage dtype and an fp8 interpretation that disagree |
| G7 | an unrouted architecture writing floats into a half-sized page |
| G8 | MLA, `float16` and `fp8_e5m2` being mis-sized instead of refused |
| G9 | the store handed K and V in DIFFERENT float dtypes, which is every production weight arm |
| G10 | the loader's own resolution stanza — that it runs, which file it reads first, that the drafter-chain refusal still precedes it, and that the #1574 subject's own two documents resolve to `auto` |
| G11 | the heterogeneous per-layer specs (Gemma-4 G1b) left at full width while the pool is sized at half |
| G12 | the SHARED SEAM's fp8 routing being dead code, which it was — the guard above it admitted no fp8 cache; and, since the third review, an fp8 cache that RAN and stayed finite while holding the wrong tensor or the wrong scale in the STORE; and, since the fourth, one that dequantized with the wrong scale on the READ |
| serve | `--kv-cache-dtype` never reaching `EngineParams` from the command line |

**G12's third case is the one that measures the cache rather than the model.**
The two logit assertions above it (`differing > 0`, `max_abs < 1.0`) were a
3000x-slack bound around a 3.4e-4 signal, and the third review walked two
mutations of `kv_cache_route.h:63` straight through a 30/30 green suite: storing
V into `k_cache` and K into `v_cache` (delta 0.0247, 73x) and storing with
`k_scale * 8` / `v_scale * 8` against an unscaled read (delta 0.0064, 19x). A
bound on the logits measures this toy model's insensitivity to its KV cache, not
the cache, and no number written on that axis would have been safe.

The repair compares the CACHE BYTES of layer 0 against the bf16 run's page,
element by element, inside the envelope the FORMAT defines: e4m3fn has three
explicit mantissa bits and rounds to nearest even, so a normal is within
`2^-4 * |ref|` and a subnormal within `2^-10 * scale`. Layer 0 is the whole
population, because its K and V are functions of the embedding and the input
layernorm alone and the two runs therefore hand the store bit-identical floats;
from layer 1 on, the fp8 run's inputs already carry the previous layer's
dequantization. The scales are 0.125 (K) and 0.25 (V) — non-unit and unequal, so
a dropped, swapped, or one-sided scale IN THE STORE leaves the envelope. MEASURED
on the repaired tree: `0/320` elements outside, `281/320` normals over `10/10`
pages. Both mutations are RED against it — `316/320` outside at worst ratio
`416.9` for the K/V swap, `319/320` at worst ratio `13.9` for the scaled store.

**That envelope gates the STORE, and the fourth review found the READ still
open.** The envelope decodes the cache bytes with the case's own
`vt::LoadKvFp8E4M3` and never enters the production dequant
(`cpu_paged_attn.cpp:167`), and the one value in the case that IS downstream of
that dequant carried no assertion. Every other case in the file that asserts a
number downstream of the read runs at `k_scale == v_scale == 1`, where a k/v
scale swap on the read is arithmetically inert. Mutating the production read to
`const float v_scale = args.k_scale;` therefore left `test_kv_cache_fp8_wiring`
at 31/31 and `test_ops_fp8_kv_cache` at 8/8, both SUCCESS, while every V the
softmax saw was halved.

**The read's SCALE is closed by INVARIANCE, and EXACTLY rather than by a
tolerance** — its ROUTING is not, and that half is under `## Owed`.
e4m3fn's normal grid is relative — for `|y|` in `[2^e, 2^(e+1))` the
representable points are `m * 2^(e-3)` — and dividing by a power of two is exact
in binary floating point and shifts `e` without touching the mantissa. So for any
two power-of-two scales that both leave a value normal and unsaturated,
`s * Dequant(Quantize(x/s))` is the same float, bit for bit: the cache BYTES
differ in every element's exponent field and the floats the kernel is handed do
not. The case runs the seam twice more, at `(2^-7, 2^-13)` and `(2^-11, 2^-9)`,
and requires `memcmp`-level agreement on the logits. Both pairs move BOTH sides,
so no single wrong-scale formula reproduces itself across them and cancels out.

**That exactness is ASSERTED for LAYER 0, and the logit equality is an EMPIRICAL
result for this fixture rather than a theorem.** The `scale_exact` `REQUIRE`
reads `bf16.buf[0]` and holds `320/320` LAYER-0 elements normal and unsaturated
at all four scales (the measured magnitudes are 1.76e-4 to 1.32e-1 for K and
5.41e-5 to 4.22e-2 for V, against all-normal windows of (2.94e-4, 1.13e-2] and
(9.43e-5, 3.46e-3]). `MakeSeamConfig` sets `num_hidden_layers = 2`, so that is
320 of the 640 elements each run stores, while the logits the case compares are a
function of BOTH layers' caches. Layer 1 does NOT satisfy the precondition:
decoding each run's own `buf[1]` at its own scales, `3/320` of its elements
disagree by up to `7.62939e-06`, all K-side and an order of magnitude below layer
0's 1.76e-4 minimum, so at `kInvAKScale = 2^-7` they fall in e4m3's SUBNORMAL
region where the grid is the absolute `2^-9` step and the power-of-two covariance
does not hold — and that layer carries 2 SATURATED elements besides, the other
escape. `CHECK(inv_differing == 0)` therefore holds by the format property for
layer 0 and by ABSORPTION for layer 1: a 7.6e-6 cache perturbation vanishing in
f32 accumulation before it reaches a logit. Extending `scale_exact` over every
layer was the preferred repair and the fixture cannot satisfy it, so the prose is
narrowed instead and the fragility is recorded under `## Owed`.

The second anti-vacuity `REQUIRE` is that the two caches really do hold different
bytes. MEASURED on the repaired tree: `0/320` logits differ, max `|delta|`
exactly `0`. Three read-side mutations of `cpu_paged_attn.cpp:167` are RED —
`320/320` at max `|delta|` `0.0673` for `v_scale = args.k_scale`, `256/320` at
`1.08e-4` for `k_scale = args.v_scale`, and `320/320` at `2.47e-3` for
`v_scale = 1.0F` — and the store envelope above reads `0/320` under all three.
The K-side one is why this is stated as exact equality rather than a bound: the
three signals span 621x, so a bound sized against the largest keeps nothing for
the smallest. Even a tight 12.4x margin against `0.0673` puts the constant at
`5.4e-3`, which admits `1.08e-4` by 50x. An absolute `1e-4` would in fact have
CAUGHT the K-side one, by 8.4%, and a relative `1e-4` against the fixture's
largest logit (`0.0627671`) is `6.3e-6` absolute and catches it by 17x — the
argument is the 621x span and the refusal to fit a constant to whichever defect
was measured first, never that one number.

A bf16-versus-fp8 comparison through `LoadedEngine` in G5/G9 was considered and
is NOT what closes this. Both mutations left the whole suite green EXCEPT the new
case, G5 and G9 included, which is the direct measurement that an engine-level
token or determinism comparison cannot see a dequant defect this model absorbs.

G4, G5, G9 and G10 enter through the production entry point (the `LoadedEngine`
constructor or `LoadedEngine::FromModelDir` → `MakeKVCacheResolved` →
`ApplyResolvedCacheDType` → `ResolveNumBlocks` → the runner →
`Qwen3_5DenseModel::Forward`), not by constructing a spec or a `PagedKvCache` by
hand. The `serve` cases enter one step earlier still, at `argv`.

**G12 enters through the OTHER registered forward**, and states its harness
adaptation once. `LoadedEngine` has no in-memory overload for
`Qwen3DenseWeights`, so its paged buffers are allocated by the case rather than
by the runner — at the width the cache dtype names, one byte per element for
fp8 and two for bf16, because allocating the fp8 arm at the float width would
hide the very mis-sizing this row exists to prevent. Everything else is
production code entered through `Qwen3DenseModel::Forward`, which is what
`ForwardQwen3ForCausalLM` calls under `ModelRegistry::Forward`
(`qwen3_dense.cpp:113`); this is the same entry
`tests/vllm/models/test_qwen3_break_point.cpp` uses for the graph-break
reachability gate.

**G9 is the case the first version of this gate could not have.** Every case in
the file built its model with `MakeDenseWeights`, whose projection weights carry
no `nk` flag, so `ProjectFullAttnQkv` served them through `MatmulF32D` and the V
projection came out f32 — the same dtype the attention preamble gives K. A real
checkpoint ships raw torch Linear weights ([N=out, K=in], `nk`), which take
`MatmulBf16D` and emit the MODEL dtype, and the fp8 store then saw an f32 K
beside a bf16 V and refused the pair by name at the first forward. That is true
of the block-wise fp8 arm (`MatmulFp8BlockScaledD`), the NVFP4 arm under the
default `VT_BF16_GEMM_OUT`, and ordinary bf16 safetensors — which includes the
#1574 campaign checkpoint. The fix normalises both to bf16 on the fp8 route
exactly as the bf16-cache route already did, and bf16 rather than f32 because
bf16 is the dtype upstream quantizes from: its model IS bf16 where
`reshape_and_cache_flash` takes key/value (`cache_kernels.cu:314-401`).

**G10 also holds an ORDERING that a merge can silently drop.** `main`'s
SPEC-DRAFTER-CHAIN W1 refusal (#1522) and this row's resolution stanza both
insert as the first statement of `FromModelDir`, and both are load-bearing: that
row's G5 requires the chain refusal before any weight I/O, and
`ReadQuantConfigJson` opens a file inside `model_dir`. Taking either side of that
textual conflict drops a guarantee.
`tests/vllm/entrypoints/test_drafter_chain_reach.cpp` cannot see an inversion,
because it points at a NONEXISTENT directory and `ReadQuantConfigJson` answers ""
for one of those without opening anything. G10's ordering case points at a
directory that EXISTS and declares fp8, so an inverted order announces the
declaration first and that line is the evidence.

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
- **RESOLVED by W3 on the CPU leg, still owed on the device.** W1 and W2 landed
  with nothing reaching the fp8 KV path from a production entry point on either
  backend. W3 wires it: `--kv-cache-dtype fp8` on the server flag now sizes
  half-width KV blocks and the routed attention blocks call
  `vt::ReshapeAndCacheFp8` and the scaled read, gated end to end on CPU by
  `test_kv_cache_fp8_wiring` G5. The CUDA arm rides the SAME `PagedKvCache`
  fields and the same two routing helpers, so it is wired by construction — and
  it is still UNMEASURED for the reason the two bullets above give.
- **W3: the C ABI does not expose `--kv-cache-dtype`** (#1593). The flag reaches
  the engine through `src/vllm/entrypoints/openai/server_main.cpp` and
  `EngineParams::kv_cache_dtype`, and NOT through `include/vllm.h` /
  `src/capi/vllm_c.cpp` / `examples/cli/main.cpp`, which the W3 dispatch's
  authority did not cover (it named `src/vllm/**`, `include/vllm/**`,
  `src/vt/**`). AGENTS.md requires every shipped capability to be reachable from
  `include/vllm.h`, so this is debt and not a design: a C-ABI caller cannot ask
  for an fp8 KV cache today unless the CHECKPOINT declares one, which the loader
  does honour on every path including that one. The ABI field, its version bump
  and its `test_capi` case are owed here.
- **W3: 16 architectures are refused rather than routed, at 17 call sites**
  (#1593). W3 routes the shared seam `dense_attn::AttnBlock` and
  `src/vllm/model_executor/models/qwen3_5.cpp`. The other direct
  `vt::ReshapeAndCache` call sites — `glm4`, `minicpm`, `opt`, `gemma`,
  `gemma2`, `gemma3`, `gemma4` (two sites), `commandr`, `phi`, `phi3`,
  `muse_glimmer`, `stablelm`, `qwen3_vl`, `olmo2`, `granite` and
  `nemotron_h_device` — keep their own attention preambles and refuse
  `--kv-cache-dtype fp8` by name. Routing each is one call swapped for
  `dense_attn::WriteKvCache` plus one `ApplyKvCacheQuant`, and each needs its
  own gate.
- **W3: 15 of those 16 refuse with a message that names neither fp8 nor the
  flag** (#1593). The refusal was first described as arriving at
  `vt::ReshapeAndCache`, and the third review traced every arm with a `kI8` page
  against a bf16 model dtype. It arrives there for exactly ONE architecture.
  Thirteen carry their own `"<arch>: KV cache must be bf16 or f32"` guard, which
  fires first — `granite:95`, `minicpm:96`, `phi3:78`, `gemma3:121`, `opt:125`,
  `stablelm:86`, `glm4:93`, `commandr:93`, `gemma:53`, `gemma2:135`, `phi:98`,
  `muse_glimmer:144` and `olmo2:94`. `gemma4:306-315` is the fourteenth and is
  the worst of the set: it casts into a `kI8` destination and dies inside
  `vt::CastF32` (`src/vt/ops.cpp:4087`, `"cast_f32: out must be f32"`), which
  names neither fp8, nor the flag, nor `gemma4`. Only `qwen3_vl:198-200` reaches
  the store guard and gets the message that names `vt::ReshapeAndCacheFp8` and
  the unrouted architecture. `nemotron_h_device:1589-1593` is the sixteenth and
  the only good refusal in the set: an explicit `VT_CHECK(false, ...)` on the
  cast that names the fp8 KV scheme and says it is not selected here. The SAFETY
  property holds for all sixteen — nothing writes floats into a half-sized page
  — but an operator who typed `--kv-cache-dtype fp8` on one of the 15 is told a
  dtype or cast rule rather than what they asked for. Widening those 15 the way
  `dense_attn_block.h:358` and `qwen3_5.cpp:5313` were widened is the same edit
  that routes them, so this is owed together with the bullet above rather than
  separately.
- **W3: G12's read-side exactness is ASSERTED for LAYER 0, and the logit
  equality is EMPIRICAL for this fixture** (#1593). The `scale_exact` `REQUIRE`
  decodes `bf16.buf[0]`, so it holds e4m3's normal-and-unsaturated precondition
  on 320 elements. `MakeSeamConfig` sets `num_hidden_layers = 2`, so each run
  stores 640, and `CHECK(inv_differing == 0)` compares LOGITS that are a function
  of both layers' caches. Layer 1 measured `3/320` elements that dequantize
  differently between the two invariance runs, by up to `7.62939e-06` — all
  K-side, an order of magnitude below layer 0's 1.76e-4 minimum, and therefore
  SUBNORMAL at `kInvAKScale = 2^-7`, where e4m3 rounds on the absolute `2^-9`
  grid and the power-of-two covariance argument does not hold; the same layer
  carries 2 SATURATED elements, the other escape. That `CHECK` passes for layer 1
  by ABSORPTION — a 7.6e-6 perturbation vanishing in f32 accumulation before it
  reaches a logit — and not by the format property. It is not knife-edge today: a
  third legal pair `(2^-9, 2^-11)` against pair A also measures `0/320` logits
  differing at max `|delta|` `0`. It is fragile in one specific way: a change to
  `MakeSeamWeights`, `kSeamTokens`, `num_hidden_layers`, the thread count or the
  accumulation order could push a layer-1 discrepancy into a logit and redden a
  CORRECT tree with a defect-shaped message. The repair is to extend
  `scale_exact` over every layer, which needs a fixture whose layer-1 K clears
  e4m3's smallest normal at every invariance scale — `2^-6 * 2^-7 = 1.22e-4`
  against the measured 7.6e-5 is the margin that is missing.
- **W3: the fp8-vs-fp8 read comparison closes the SCALE half of the read and not
  the ROUTING half** (#1593). G12's invariance case compares two fp8 runs, so it
  is structurally blind to a read-side defect that is a function of BYTES and
  INDICES rather than of scales: both runs commit it identically and it cancels.
  Two mutations of production code were measured and both PASS the whole file at
  `31/31`, `487/487`, with `0/320 logits differ` — serving V out of the K page
  with K's scale (`cpu_paged_attn.cpp:174` `v_base = k_cache.data` together with
  `:167` `v_scale = args.k_scale`), which is the read-side twin of the
  `N1_KVSWAP` store mutation the envelope DOES catch; and dropping the in-page
  token offset from the V read (`:270`, `off & 0`), which is pure indexing and
  carries no scale at all. Closing that class needs a comparison against a
  REFERENCE — the bf16 run's page, or the case's own decode through the same
  math — and never a second fp8 run. The design paragraph above therefore claims
  the SCALE half only, and this bullet is the other half.
- **W3: no weight loader extracts `k_scale`/`v_scale`** (#1593). `ResolveKvCacheScales`
  mirrors all four of upstream's arms, and the loader calls it with the
  `KVCacheScaleParameter` unloaded sentinel for both scales, so every declaring
  checkpoint lands on `kDeclaredButAbsent` and serves at 1.0. That is CORRECT for
  the #1574 gate checkpoint under an explicit `--kv-cache-dtype fp8`, since it
  ships zero KV scales (and it never reaches the arm without the flag, because
  its inline `config.json:quantization_config` declares no `kv_cache_*` key) —
  but the two
  checkpoint-loaded arms (`kCheckpoint`, `kCheckpointKvScale`) are unit-gated and
  unreached, and a calibrated checkpoint would silently serve uncalibrated. The
  per-layer scale-tensor read, and a per-layer (rather than per-engine) scale on
  `AttentionSpec`, are owed.
- **W3: `--calculate-kv-scales` is refused, not implemented** (#1593). Upstream's
  dynamic on-the-fly scale (`config/cache.py:111`) is deprecated there for
  removal in v0.19; `ResolveKvCacheScales` refuses it BY NAME rather than
  silently taking the static arm, and no flag exposes it.
- **W3: an fp8 KV cache is served by the SLOW attention kernels only** (#1593).
  `fa2_prefill` and `fa2_decode` in `src/vllm/model_executor/models/qwen3_5.cpp`
  both require `kv.dtype == DType::kBF16`, and
  `src/vt/cuda/cuda_paged_attn.cu` routes an fp8 read to tiled prefill and block
  decode only. Every other launcher — the WMMA ladder, the vendored FA-2
  kernels, the vectorized decode-opt and GQA kernels — is bf16-native by
  construction and would need an fp8 instantiation. Owed by this row (`KV-FP8`)
  under #1593. Until it lands, `--kv-cache-dtype fp8` trades attention
  throughput for KV bytes at an unmeasured exchange rate, and the row says so in
  `## W3`, in `docs/USAGE.md` and in `docs/FEATURES.md` rather than implying the
  memory win is free.
- **W3: the heterogeneous per-layer fp8 arm is SIZING-only** (#1593).
  `ApplyCacheDType` retypes `KVCacheConfig::per_layer_attn_specs` as well as the
  group specs, which is what keeps `KVBytesPerBlock` and the runner's own
  per-layer allocation agreeing, and G11 gates that arithmetic. No shipped
  architecture can spend it yet: the only model that populates that vector is
  Gemma-4 (G1b), and Gemma-4 is one of the 16 architectures whose attention block
  refuses the fp8 store by name. So a Gemma-4 run with `--kv-cache-dtype fp8`
  gets a correctly halved pool and then a named refusal at the first forward,
  which is the intended order. Routing Gemma-4 is owed with the other 15 above.
- **W3: the resolver accepts one producer-only document upstream REFUSES**
  (#1593). `GetKvCacheQuantAlgoString` takes the modelopt marker from
  `producer["name"] == "modelopt"`, which mirrors the injection
  `ModelArchConfigConvertorBase._normalize_quantization_config`
  (`transformers_utils/model_arch_config_convertor.py:208-247`) performs before
  `torch_utils.py:319` ever reads the key — RUN, not transcribed, on 2026-08-22:
  `nvidia/Llama-3.3-70B-Instruct-FP8`'s producer-only `hf_quant_config.json`
  answers `None` before normalization and `'fp8_e4m3'` after it, and G1's marker
  case pins all three markers.

  **The third review found that acceptance was UNGUARDED, and W3 now mirrors the
  guard.** `:224` injects only `if quant_algo is not None`, and it reads that key
  out of `quant_cfg.get("quantization", {})` — an EMPTY-object fallback, unlike
  the reader's `quant_cfg.get("quantization", quant_cfg)` at
  `torch_utils.py:321`. Three documents therefore answer `None` upstream and used
  to answer `fp8_e4m3` here: a `producer.name` of `modelopt` with no
  `quant_algo`, a `quantization.modelopt_quant_config` with no `quant_algo`, and
  a TOP-LEVEL `modelopt_quant_config` with no `quantization` key at all. All
  three are now refused, arm (g) of G1's marker case pins each of them together
  with the same document made acceptable by adding the `quant_algo`, and
  restoring the pre-repair predicate turns seven of that case's assertions red.
  No shipped fixture moved: every real document in this suite —
  `kGateCheckpointQuantConfig`, `kNoKvDeclarationQuantConfig`,
  `kInlineWeightsOnlyQuantConfig` and the #1574 subject's own two files — carries
  a `quant_algo`.

  **One arm of that injection is still not mirrored:**
  upstream RAISES `ValueError: Unknown ModelOpt quant algo: <algo>` (`:235`)
  when the producer is modelopt and the nested `quant_algo` is neither
  FP8-family nor NVFP4, and we answer `fp8_e4m3` instead. This is why the
  mirrored test above is exactly `quant_algo is not None` rather than the family
  set: the only two upstream outcomes for a `quant_algo` that IS present are
  "inject" and "raise", and taking the raise's arm collapses them into one. That
  refusal is a
  WEIGHT-half validation living in a config convertor this port does not have,
  and moving it into the KV resolver would refuse a `MIXED_PRECISION`
  checkpoint whose weights `modelopt_mixed_precision.h` loads. It is unreachable
  for the #1574 subject, whose inline `config.json` document wins, so it is
  recorded here rather than implemented; porting the convertor's validation is
  its own row.
- **W3: no engine auto-selects fp8 KV for the #1574 subject** (#1593), and the
  campaign has to type `--kv-cache-dtype fp8` on both sides. Not a defect — the
  mirror is correct on both halves and the competitors' `serve.sh` already
  passes the flag — but it means the checkpoint declaration path this row
  builds is gated by G1/G10 rather than exercised by the benchmark it was built
  for. A calibrated ModelOpt checkpoint that declares the algorithm INLINE is
  what would exercise it end to end, and this row has none.
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
