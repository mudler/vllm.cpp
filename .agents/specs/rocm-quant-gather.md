# ROCm quantized embedding gather

Row: `BACKEND-ROCM-QUANT-GATHER`.

Issue: [#3093](https://github.com/mudler/vllm.cpp/issues/3093).

Parent debt: [#2394](https://github.com/mudler/vllm.cpp/issues/2394), owned by
[`MODEL-MM-QWEN4-EXP`](cuda-quant-gather.md#owed).

Branch: `row/BACKEND-ROCM-QUANT-GATHER`.

Spec base: `6db4bef906859e864c82523c01107473f7dcca29`.

## Now

`ACTIVE`. The native HIP gather and public compressed-embedding path are implemented.
The operator's final GPU run passed all 15 CTests and the required two-device case.
The public gate reaches all 19 formats. All 76 operation outputs
meet the pinned oracle gates, and matching native/primary traces pass.
The row uses one pull request and retains the reviewed full-attention prerequisite.

The 32 original upstream fixtures were downloaded with the developer's authority
and verified at 289,655,872 bytes; the 160 exported tensors over the 16 primary
codecs pass the native-versus-plugin comparison (320 outputs, 140 byte-exact,
every remaining element inside the upstream tolerance).
The 96-request primary token matrix executed: 93 requests are token-identical,
and the three IQ3_S first-prompt cases are an oracle-side plugin grid defect
reported upstream as `vllm-project/vllm-gguf-plugin#129`.
The bounded secondary overlays execute all 12 model requests and match native tokens.
Both pristine and overlay binaries preserve all 24 recurrent-control completions and logits.
The amended native gate passes all 228 public engines with 16 physical blocks.
The live primary observer measures a 65,536-byte BF16 cache with the same physical capacity.
The comparator rejects unqualified captures, its guards are pinned by focused
negative cases registered in the preflight and CI lanes, and full preflights exit
zero at the amended heads.
The earlier four-block measurements and argument-dependent preflight skips remain archived.
The row is not ready to land or become `DONE` while the three IQ3_S cases wait on
the upstream grid correction and the secondary matched-memory claim stays scoped.

The [row evidence](../../docs/bench-evidence/rocm-quant-gather/README.md)
records executing pins, exact recipes, stage hashes, rejected attempts, and limits.

## Scope

Register `vt::OpId::kEmbeddingQuant` for ROCm through a native HIP gather.
Decode selected packed rows directly into the caller's f32 or bf16 output.
Keep the shared loader's compressed embedding residency reachable through the
public C application binary interface (ABI).

Cover every format accepted by
`src/vllm/model_executor/model_loader/gguf_keep_quant.cpp::KeepQuantGatherDType`.
Its device gate accepts no dtype argument. A partial ROCm decoder set therefore
cannot register the capability safely.

This change owns correctness, memory residency, and native production reachability.
It does not depend on PR #2782. The existing ROCm quantized dot provider already
supplies the default `keep_quant` policy input.

The following changes remain outside this row:

- F16 GEMM, fused mixture-of-experts execution, and quantized dot providers.
- Floating embedding behavior, the public ABI, and the shared dtype set.
- CUDA decoder behavior or decoder ownership shared across backends.
- Qwen4-exp model completeness, other backends, and large-checkpoint qualification.
- Performance tuning, performance ratios, and global oracle pin changes.
- Global record counts, checker semantics, and unrelated keyed records.

## Inventory

This per-row inventory owns the child identity through the canonical spec scan.
The existing `BACKEND-ROCM` matrix lifecycle remains unchanged.
Update this inventory and `Now` together when this child changes state.

| ID | Upstream source | Local anchor | Tests and evidence | Spec | State | Owner | Issue |
|---|---|---|---|---|---|---|---|
| `BACKEND-ROCM-QUANT-GATHER` | Plugin `_apply_gguf_embedding`, pinned codec sources | `vt::Embedding`, `KeepQuantGatherDType`, new ROCm gather | Focused and public gates in this spec | This file | `ACTIVE` | `row/BACKEND-ROCM-QUANT-GATHER` helper, operator verifies | #3093 |

### Complete decoder set

The geometry comes from `src/vt/dtype.cpp::FindBlockGeometry` and
`src/vt/cpu/cpu_quant_dequant.cpp::BlockToFloat` at the spec base.
The loader admits all 19 entries, including Q8_K without a weight dot kernel.

| DType | GGML ID | Elements per block | Bytes per block | Executing reference |
|---|---:|---:|---:|---|
| Q4_0 | 2 | 32 | 18 | Primary plugin |
| Q5_0 | 6 | 32 | 22 | Primary plugin |
| Q8_0 | 8 | 32 | 34 | Primary plugin |
| Q2_K | 10 | 256 | 84 | Primary plugin |
| Q3_K | 11 | 256 | 110 | Primary plugin |
| Q4_K | 12 | 256 | 144 | Primary plugin |
| Q5_K | 13 | 256 | 176 | Primary plugin |
| Q6_K | 14 | 256 | 210 | Primary plugin |
| Q8_K | 15 | 256 | 292 | Stock llama.cpp decoder |
| IQ2_XXS | 16 | 256 | 66 | Primary plugin |
| IQ2_XS | 17 | 256 | 74 | Primary plugin |
| IQ3_XXS | 18 | 256 | 98 | Primary plugin |
| IQ1_S | 19 | 256 | 50 | Primary plugin |
| IQ4_NL | 20 | 32 | 18 | Primary plugin |
| IQ3_S | 21 | 256 | 110 | Primary plugin |
| IQ2_S | 22 | 256 | 82 | Primary plugin |
| IQ4_XS | 23 | 256 | 136 | Primary plugin |
| MXFP4 | 39 | 32 | 17 | Stock llama.cpp GET_ROWS |
| IQ1_XXXS | 66 | 256 | 38 | Registered unsloth fork GET_ROWS |

The primary test includes IQ1_M, which the shared loader does not admit.
Its F16 output case exceeds `vt::Embedding`'s f32/bf16 output contract.
Record these two exclusions explicitly when reporting upstream test coverage.
They do not remove an admitted codec from this change.

## Oracle pins and executing chain

The active primary vLLM pin is `e126687a9a828d513c01a07cd69f025f27d63280`.
The plugin pin is `d4c1f0d082fc7cd4350da56689109a01c1f29d6c`.
Stock llama.cpp uses `10bf611e533d81f739128304991c5e133c6aebd8`, tag `b10451`.
The registered unsloth fork uses `36fe8e1cc7f2b3b8c92fdda0ab07600141921786`.

Read [the primary pin](../upstream-sync.md),
[the plugin record](../oracles/vllm-gguf-plugin.md),
[the stock record](../oracles/llama-cpp.md), and
[the fork record](../oracles/llama-cpp-unsloth.md).
Do not promote a source inspection into runtime qualification.

### Primary behavior and tests

The plugin implements compressed-row selection followed by selected-row decode.
The earlier CUDA spec predates that current primary behavior source.
A fused HIP kernel adapts the implementation and preserves the observable result.

Pinned plugin anchors:

- `vllm_gguf_plugin/quantization/vocal_embeds.py::_apply_gguf_embedding`, lines 79 to 98, selects packed rows and dequantizes them.
- `vllm_gguf_plugin/quantization/vocal_embeds.py::GGUFEmbeddingMethod.embedding`, lines 186 to 194, passes `self.params_dtype`.
- `vllm_gguf_plugin/ops.py::ggml_dequantize`, lines 181 to 186, selects native or Triton dequantization.
- `tests/test_kernels.py::test_gguf_embedding`, lines 102 to 124, defines upstream fixtures and comparison parameters.

The [pinned embedding implementation](https://github.com/vllm-project/vllm-gguf-plugin/blob/d4c1f0d082fc7cd4350da56689109a01c1f29d6c/vllm_gguf_plugin/quantization/vocal_embeds.py#L79)
and [pinned embedding test](https://github.com/vllm-project/vllm-gguf-plugin/blob/d4c1f0d082fc7cd4350da56689109a01c1f29d6c/tests/test_kernels.py#L102)
are the primary source anchors.
Read the executed native or Triton decoder after runtime dispatch resolves.
Record its source symbol, output dtype, and generated kernel when applicable.

Run the primary method on identical packed tables and IDs for all 16 overlapping
formats, both output dtypes, and hidden widths 256 and 1024.
Preserve every upstream tensor, selected row, repeated ID, and tolerance.
The original comparison uses `atol=0.01` and `rtol=0.04`.
The shared ABI flattens upstream's two-dimensional IDs into one contiguous vector.
Restore the original shape only in the comparison harness.

The existing active-pin runtime does not contain `vllm_gguf_plugin`.
Build the exact plugin pin in an isolated task directory when dependencies are available.
Do not change the existing runtime or install packages without the recorded authority.
A failed import, missing dependency, or missing fixture keeps this gate `PENDING`.
Do not replace primary outputs with local CPU outputs.

Qualify the plugin on the bounded model used by this row on gfx1100.
Use the normal production configuration, with no eager-mode denominator override.
Capture resolved model dtype, model revision, plugin revision, and outputs.
If the primary cannot load the candidate model, record the exact refusal and
resolve the harness adaptation before accepting the primary gate.

### Secondary formats

Stock llama.cpp's `ggml/src/ggml-cpu/ops.cpp::ggml_compute_forward_get_rows_q`
routes quantized GET_ROWS through each type's `to_float` function.
`tests/test-backend-ops.cpp::test_get_rows`, starting at line 2247, defines the
stock test vehicle. Its case generation starts at lines 8365 to 8377.
Retain applicable row widths, IDs, repetitions, and f32 output comparisons.
Batched tables and strided ID views exceed this ABI and require explicit
adaptation entries plus local refusal tests.

Q8_K is not selected by stock GET_ROWS dispatch at this pin.
Run the exact upstream `ggml/src/ggml-quants.c::dequantize_row_q8_K` over each
selected packed row. Record this as the Q8_K oracle harness adaptation.
Do not claim an upstream GET_ROWS execution for that format.
The stock pin must also run the bounded model with its supported quantized table.

The fork registers IQ1_XXXS `to_float` in `ggml/src/ggml.c`, lines 871 to 877.
Its [pinned GET_ROWS dispatch](https://github.com/unslothai/llama.cpp/blob/36fe8e1cc7f2b3b8c92fdda0ab07600141921786/ggml/src/ggml-cpu/ops.cpp#L5063)
includes type 66 in `ggml_compute_forward_get_rows` at lines 5063 to 5069.
Build that exact fork and execute a real upstream GET_ROWS graph on identical bytes.
A new scalar transcription or local `BlockToFloat` cannot stand in for this graph.

The fork must additionally load and generate from a bounded real model graph
whose token table uses IQ1_XXXS. A codec graph alone does not satisfy model execution.
The tiny qwen35 fixture described below is the first candidate.
Verify that the exact fork accepts its metadata and architecture before capture.
If it refuses, propose a compatible bounded model through the same public local path.
Do not silently replace the model run with a graph-only comparison.

Record this qualification narrowly as a synthetic gather vehicle.
Keep issue #933 and the registry's real 2.4T qualification debt unchanged.
A synthetic run cannot establish that the published 2.4T checkpoint runs.
The IQ1_XXXS oracle gate remains `PENDING` until the bounded model actually runs.
The row cannot become `DONE` while that gate remains pending.

## Fixture manifest

The primary tests use `Isotr0py/test-gguf-sample` at revision
`d82b8773934ef260d8d8a896a7c197bc69a0fac1`.
The 32 applicable files total **289,655,872 bytes**.
The table records Hugging Face metadata read on 9 September 2026 UTC.
The SHA-256 values are advertised LFS object IDs, not locally verified downloads.
Verify each complete file's size and SHA-256 before using it.

Keep the original full fixtures. Do not reconstruct unselected rows to avoid a download.
Use existing matching cached files when available.
The operator obtains the required download authority if those files are absent.
No fixture was downloaded during specification.
A missing authorization or missing fixture keeps the primary fixture gate `PENDING`.

The download URL pattern is
`https://huggingface.co/Isotr0py/test-gguf-sample/resolve/d82b8773934ef260d8d8a896a7c197bc69a0fac1/<file>`.

| File | Bytes | SHA-256 |
|---|---:|---|
| `Quant_IQ1_S_1024.gguf` | 5581184 | `dea6372848ae8ba284481148ea282dde3c56d145eba5768f7d72a1527bf71a96` |
| `Quant_IQ1_S_256.gguf` | 1395584 | `52f0fba7e7f50821a343205c62948beef7f2320a52897ac197562ad975fe8ce7` |
| `Quant_IQ2_S_1024.gguf` | 9152896 | `8938259277b08682b596056ac63c8b347a6861ef3dfa7c8dd7f5466d26e4d6d6` |
| `Quant_IQ2_S_256.gguf` | 2288512 | `80207b4810ead2fb41ff8b42fe8a1c95d67626ed26f459b4d3e06cc4b78e4e17` |
| `Quant_IQ2_XS_1024.gguf` | 8259968 | `91871db5d358233b40a687de99d6d6fe15839dee4b37eeb3d98c1f8fbf8d43fc` |
| `Quant_IQ2_XS_256.gguf` | 2065280 | `7ca899750b6ae8902b4a4e219e153ff6c1d99f7d2074b76b47bd1cc13721607a` |
| `Quant_IQ2_XXS_1024.gguf` | 7367072 | `1946509f1c1ba47146b77645859b70e95f4ec0582e161d4a5693b860e5953fe8` |
| `Quant_IQ2_XXS_256.gguf` | 1842048 | `98cd99fab6415ec0115423bf91f51915febb06e4da9e6ef7e061a90fbff8659e` |
| `Quant_IQ3_S_1024.gguf` | 12278144 | `ed04c0b1929333bc25fcda2b77d630944690de08f1f194e2849a2f00702f30f3` |
| `Quant_IQ3_S_256.gguf` | 3069824 | `1852ac6ccb0e10c8dfe47b23034f57cb3efae4c38377243b40847064cac1d7cf` |
| `Quant_IQ3_XXS_1024.gguf` | 10938784 | `6eb74c3e749f758125ad6c2aae24481bd918e65703a099557109d18302f39d77` |
| `Quant_IQ3_XXS_256.gguf` | 2734976 | `0150e9a21ecbed3f47b1b1d61bcccd0c00815e6cd84c8ef7d9f10d2f76568b72` |
| `Quant_IQ4_NL_1024.gguf` | 16073088 | `05e0ca6956e3d036dd3d9edd055e8087e0ac6376c0032613498974759c752bd1` |
| `Quant_IQ4_NL_256.gguf` | 4018560 | `d3d86e735abf49c57e02e83ad6750ac9bf216a9a884a8eaf097aebcc0422540c` |
| `Quant_IQ4_XS_1024.gguf` | 15180160 | `5ef94e3cec7c9ff42d0fbc7705f425162fd3e00253dd05a62cc17e6bc85451c4` |
| `Quant_IQ4_XS_256.gguf` | 3795328 | `6039ca86d3c50cfc8db68b0394450ac1135affb8f07890af537865fb1dd5d23c` |
| `Quant_Q2_K_1024.gguf` | 9376128 | `8383805d71bf8252c30dbd59662dafb5ce6c29a71b8e74b349b708327c541703` |
| `Quant_Q2_K_256.gguf` | 2344320 | `a5f0df4b88193d1c5102d39113ae8ab86b23678ad5bbf459d05d1b88119ec084` |
| `Quant_Q3_K_1024.gguf` | 12278144 | `af666e306231f864fbd3fb7047f3980f058d070478a60b6790e1a823d773c406` |
| `Quant_Q3_K_256.gguf` | 3069824 | `1bc430759c9768783b08118323ec795bcffa62d9551933140a22f84c7135aade` |
| `Quant_Q4_0_1024.gguf` | 16073088 | `704bd822cde61be8437216089bc0240658869732316d7bbacf7e53de4f584797` |
| `Quant_Q4_0_256.gguf` | 4018560 | `380dd142b886bc7b537e9ba3569019560d68dd3aa13385b6c848121f0e33e23e` |
| `Quant_Q4_K_1024.gguf` | 16073088 | `00577c14c413f41d179dd11f5c92e0e3f1b9ed0c2ab410a0960f493db9d32c6b` |
| `Quant_Q4_K_256.gguf` | 4018560 | `c6256d86c59bc96db1eada611ec1fccb147eb3a738e3693cd2a68a497df77f53` |
| `Quant_Q5_0_1024.gguf` | 19644800 | `18251212453fcd01d04d7e7a5c2b847f48ec8a4c063a18d2a7c15122697354f7` |
| `Quant_Q5_0_256.gguf` | 4911488 | `8b8b03108ff423bc85d99ee03b99dad7138364c9a7fcac9da97b03c89696dd88` |
| `Quant_Q5_K_1024.gguf` | 19644800 | `e0af3999f1bb469be5c6b951b134d7bf95aa2fa8f65d8656e3acd9dec35de4ee` |
| `Quant_Q5_K_256.gguf` | 4911488 | `dabf970e3c496be2c512d8afbcbea22d9ba958867a7ed50a36a71e1fd523635e` |
| `Quant_Q6_K_1024.gguf` | 23439744 | `e7d173382ec11eaa6ae981e8600dce38196203f193433df95ab83944fb4508cc` |
| `Quant_Q6_K_256.gguf` | 5860224 | `b2c8f71efa6a27204ae1c111eb1cecf4617cd3a07be5e7be80fb28e6cbe23839` |
| `Quant_Q8_0_1024.gguf` | 30359936 | `4f7f1fbeade44af1cadcf57287eec4dcbadcaf7da650edd07e49a9f699007666` |
| `Quant_Q8_0_256.gguf` | 7590272 | `c27d7a1c1acd5b8176c320eb0f4a6bf0a23e53cfe460f6fc9b86b69422bc2b17` |

## Design

### Native decoder ownership

Add independently owned ROCm decoder and codebook headers.
Port the decoder expressions and codebooks already used by the CUDA gather.
Cite both the local source and each pinned upstream definition beside the port.
Record the fused HIP launch as an adaptation of the plugin's two operations.
Do not include CUDA implementation headers from ROCm.

Read scale fields and packed payloads through byte assembly.
Odd block sizes and offsets 0 to 3 must not create aligned typed-load assumptions.
Preserve scalar arithmetic order with the existing HIP `-ffp-contract=off` flag.
Match `vt::F32ToBF16` round-to-nearest-even conversion, including tie cases.
Do not add an f32 staging tensor to the bf16 gather path.

Map each logical selected block to one HIP thread.
Use 128 threads per block and a grid capped at 1024 blocks initially.
Retain a grid-stride loop over the full selected-block count.
This bounded cap makes continuation directly testable with small fixtures.
The cap is a correctness-port default with no throughput claim.
Measure its cost before accepting a later performance default.

Compute row stride in bytes with `RowSizeBytes`.
Use full-width arithmetic for IDs, offsets, and block counts.
Do not truncate i64 IDs before bounds validation.
Reject an invalid selected row before reading its packed bytes.

Retain ROCm's synchronous bounds-error contract for this new operation.
Allocate a device error record, clear it, launch, copy the error, and report it.
Release every temporary allocation on success and on each failure path.
An empty request returns without a kernel or error-buffer allocation.
An empty vocabulary accepts no nonempty request.

The shared `vt::Embedding` checks remain authoritative for ranks, shapes,
dtypes, contiguity, whole block rows, and device equality.
Keep floating gather dispatch unchanged.
Bind the native gather to `queue.device.index` before allocating its error
record or launching work. A different ambient HIP device must not redirect
the allocation, launch, or cleanup. The two-device regression keeps the queue
and tensors on device 0, selects ambient device 1, and checks rows, bounds,
and recovery. Fresh review removes this binding and requires that gate to fail.

### Loader and production reachability

The registration must activate the existing shared residency decision.
Do not add a device-name branch to the loader.

The production chain is:

```text
vllm_engine_load
  -> LoadedEngine::FromModelDir
  -> ModelRegistry::Load
  -> LoadQwen3_5DenseModel
  -> LoadQwen3_5DenseFromGguf
  -> LoadEmbedAndHead
  -> ModelRegistry::Forward
  -> Qwen3_5EmbeddingTable
  -> EmbedInto
  -> vt::Embedding
  -> kEmbeddingQuant ROCm provider
```

Relevant local anchors at the spec base:

- `src/capi/vllm_c.cpp::vllm_engine_load`, line 824.
- `src/vllm/entrypoints/model_loader.cpp::LoadedEngine::FromModelDir`, line 3098.
- `src/vllm/model_executor/models/qwen3_5_dense.cpp::LoadQwen3_5DenseModel`, lines 87 and 147.
- `src/vllm/model_executor/models/qwen3_5_gguf_weights.cpp::LoadEmbedAndHead`, line 785.
- `src/vllm/model_executor/models/qwen3_5.cpp::EmbedInto`, line 8223.
- `src/vt/ops.cpp::Embedding`, line 1523.

Use the existing GGUF builder to create a deterministic tiny qwen35 model.
Use one full-attention layer, hidden width 256, four query heads, and one KV head.
Use head width 64, intermediate width 256, and exactly 128 vocabulary entries.
Set the full-attention interval to 1, RMS epsilon to `1e-6`, and context length to 64.
Set RoPE base to `1000000.0` and rotary width to 64.
Use tokenizer IDs 0 for unknown, 1 for beginning, and 2 for end.
Create distinct vocabulary strings for all 128 IDs and preserve their order.
Use generator seed `0x524f434d` for every codec variant and dense control.
Use `std::mt19937` outputs with explicit integer-to-float conversion, without library distributions.
Initialize dense projection values within `[-1/64, 1/64]` and norm values to 1.
Generate finite packed blocks with the same seed and codec-specific valid scale fields.
Record the generation algorithm and verify identical GGUF hashes across two regenerations.
Quantize only `token_embd.weight`. Keep projections and a separate output head dense.
A separate head prevents a quantized dot dependency for Q8_K and IQ1_XXXS.
Keep every generated model under 8 MiB and record the exact content hashes.

Run this same public path for all 19 admitted formats.
Load with public `device=0` in a HIP-only build and retain default compressed residency.
The ABI defines automatic selection as 0 and has no explicit ROCm device value.
Require measured ROCm provider selection before accepting the automatic selection.
Set `block_size=16`, `num_blocks=16`, `max_model_len=64`, and `max_num_seqs=1`.
Set `kv_cache_dtype="auto"` and require its resolved cache storage to be bf16.
The one-layer KV payload is 65,536 bytes, excluding allocation alignment and block metadata.
Match physical capacity and cache storage in each executing oracle. The primary
and native pools reserve one of their 16 blocks, leaving 240 usable cells.
Stock and fork llama.cpp pad the requested context of 64 to 256 cells.
Their cache is contiguous and has no equivalent native page-layout requirement.
Keep requested context, prompts, and generation lengths unchanged. Report each
engine's resolved capacity separately; do not call its logical limit identical.

Use two pre-tokenized prompts: `[1, 0, 63, 127, 63]` and `[1, 127, 0, 127]`.
Their lengths are 5 and 4. Do not insert tokenizer special tokens in either oracle.
Generate exactly 4 tokens per prompt with greedy sampling and `ignore_eos=1`.
Set `has_seed=1`, `seed=0x524f434d`, and keep stop lists empty.
Run each prompt after a fresh engine load, and repeat the pair 3 times per codec.
Each run includes prefill and subsequent one-token decode through `vllm_complete_tokens`.
Capture logits through the public logits processor and record generated token IDs.
Require exact IDs and logits against the equivalent dense-embedding model on
the same ROCm binary. Dense control embeddings use the same bf16 conversion.
That control isolates embedding residency from unrelated CPU/GPU arithmetic.
The primary and secondary operation gates remain independently required.

Compare all generated IDs exactly against the appropriate pinned model oracle.
Use the primary plugin for its 16 formats, stock llama.cpp for MXFP4, and
the registered fork for IQ1_XXXS. Require all 3 repeats to agree on each side.
Use identical model bytes, prompts, token counts, sampling, batching, and concurrency.
Record a load refusal instead of substituting local outputs or a different codec.
Q8_K retains its explicit upstream decoder-only adaptation because stock GET_ROWS excludes it.
Its public model still requires exact dense-control equality and independent upstream decoder bytes.
Do not describe that Q8_K control as a pinned-oracle model token result.

Enable `vt::EnableOpProviderCallStats` around the public calls.
Require positive ROCm `kEmbeddingQuant` selections for every codec.
Require no provider fallback and unchanged `vt::GetReferenceTierHits`.
A test that constructs the embedding type by hand cannot satisfy this gate.
No new test-only production entry point is allowed.

### Memory contract

The resident packed table occupies `V * RowSizeBytes(dtype, H)` bytes.
The selected output occupies `T * H * sizeof(output_dtype)` bytes.
For the model path, the output dtype is bf16 and costs `2 * T * H` bytes.
The primitive's f32 output remains an explicit caller-selected test mode.
The bounds-error record is constant-size scratch.

Capture table dtype, packed bytes, output dtype, output bytes, and scratch bytes.
Compare those values with the primary plugin on the identical tables and IDs.
The primary stages selected packed rows before decode. The fused HIP path does
not need that intermediate and must record this implementation adaptation.
Neither implementation requires full-table floating expansion for a gather.

Trace both implementations with the same available tool and workload.
Verify selected-row decode, no full-table expansion, and no host decode.
Provider counts alone cannot detect a manually coded host fallback.
A missing profiler keeps the trace gate `PENDING` until the operator supplies one.
Do not install a profiler without the recorded authority.

## Implementation files

The spec commit changes only this spec, the child issue record, and the
parent debt link.

The implementation is expected to add:

- `src/vt/rocm/rocm_embedding_quant.h`
- `src/vt/rocm/rocm_embedding_quant.hip`
- `src/vt/rocm/rocm_quant_dequant.h`
- `src/vt/rocm/rocm_quant_iq_tables.h`
- `tests/vt/test_rocm_embedding_quant.cpp`
- `tests/capi/test_rocm_embedding_quant.cpp`
- Row-owned oracle capture and comparison tools when required by the runtime gates.
- Row-owned evidence under `docs/bench-evidence/rocm-quant-gather/`.

The implementation can modify:

- `src/vt/rocm/rocm_ops.hip` for the registration.
- Both HIP source lists in `CMakeLists.txt`.
- `tests/CMakeLists.txt` for targets and real exit-77 skip behavior.
- `tests/vllm/test_gguf_keep_quant.cpp` for the changed ROCm capability expectation.
- `tests/vt/test_cuda_embedding_quant.cpp` for the combined-build unavailable set.
- `src/vllm/model_executor/model_loader/gguf_keep_quant.cpp` for stale gather comments.
- `docs/FEATURES.md` for the changed ROCm feature statement.
- This spec, its issue record, and the already linked parent debt entry.

The scoped materializer amendment below authorizes two shared loader files.
Other shared model behavior and unrelated production code require a scope
review before implementation.

### Q8_K materializer amendment

The operator approved this amendment on 9 September 2026 UTC under issue #3093.
Commit this amendment before changing either newly scoped product file:

- `src/vllm/model_executor/models/qwen3_5_gguf_weights.cpp`
- `include/vllm/model_executor/models/qwen3_5_gguf_weights.h`

The first native focused run passed 4 of 6 targets. The public test reached
Q8_K after the first 8 formats matched their dense controls. Its first Q8_K
load returned status 2, without a readable error capture. The native decoder suite separately passed all 11,698
value assertions, but its empty-shape fixture failed during tensor construction.
That malformed test fixture requires a metadata view with zero dimensions.

`LoadEmbedAndHead` already selects `KeepQuantGatherDType` through the shared
residency policy. It then calls `OwnGgufQuantBlocks`, whose guard at line 89
requires `KeepQuantDType`. The latter requires a dot kernel and refuses Q8_K.
This downstream guard would refuse the decoder-only format after the reader
accepts it. That materializer failure was a source hypothesis at this amendment.
`git log -S 'VT_CHECK(KeepQuantDType(tensor.ggml_type'` identifies commit
`429e19d6a` as the guard's introduction.

Add an explicit tensor role to `OwnGgufQuantBlocks`. Default the role to
`kMatmulWeight`, preserving the dot requirement for every existing caller.
The embedding materializer explicitly selects `kEmbeddingTable`, which checks
`KeepQuantGatherDType` instead. Preserve packed bytes, orientation, borrowing,
copying, prefaulting, and every existing matrix repack default. A gather must
not opt into a matrix repack. Do not change either shared dtype set, add a
dot kernel, or change any backend capability outside this row.

Add a smallest loader test in `tests/vllm/test_gguf_keep_quant.cpp` using the
same generated Q8_K model. Require the ordinary loader to retain the original
packed embedding bytes and `nk=false`. Require the default materializer to
refuse that decoder-only format and the matrix residency route to expand it.
Keep the public Q8_K load and completion test as the production gate.
Fresh review removes the embedding role argument and weakens the default dot
guard separately. Each mutation must fail its corresponding gate.

The operator's first native run used the manifest
`native-focused-v1.json`, SHA256
`496a12c417b9cb95036896da418d74fd8383fc6d1497f0ed94f4e457791bf479`.
Its log is `native-focused-v1.log`, SHA256
`2f21277ec55c85cf6cd4abbb669a688bdfab415de13c9bcd0588e6a7854787ac`.
Both files reside under the row's external evidence directory
`/home/vikash/.cache/rdna3-gather-impl/evidence/`.
The operator verified unchanged source, archive, and executable hashes around
the run. The Q8_K public load had 1,108 passing assertions before its failure.

### Q8_K reader amendment

The new loader-focused red identifies the first executing refusal:
`GgufFile::OpenOne` rejects GGML type 15 at `gguf_reader.cpp:506`, because
`FindGgmlTraits` has no entry for it. The result is exit 1 after one passing
fixture assertion. The log is `materializer-loader-red.log` in the same
external evidence directory. The materializer guard remains a separate
downstream hypothesis until the reader accepts the file.

The operator approved this additional scope on 9 September 2026 UTC:
`src/vllm/model_executor/model_loader/gguf_reader.cpp`. Commit this amendment
before changing that file. Add exactly the existing Q8_K geometry for GGML
type 15: 256 elements and 292 bytes per block. Stock llama.cpp pin
`10bf611e533d81f739128304991c5e133c6aebd8`, `ggml/src/ggml-common.h:370-376`,
defines `block_q8_K` as a float delta, 256 signed bytes, and 16 signed 16-bit
sums. This is already the geometry in `vt::DType::kQ8_K`.

Keep all other traits byte-for-byte equal. Preserve unknown-type refusals,
overflow checks, span checks, and whole-block divisibility. This amendment
does not add a shared dtype or change a matrix dtype set. Add the reader
regression in `tests/vllm/test_gguf_keep_quant.cpp`, checking geometry and
original packed bytes through `GgufFile::Open`. Fresh review deletes the
type-15 entry and mis-sizes its geometry independently. Both mutations must
fail a meaningful gate. After adding the trait, rerun the loader-focused red
to identify the materializer failure before changing that guard.

### Bounded oracle configuration and allocation amendment

The operator approved this scope on 9 September 2026 UTC under issue #3093.
Commit this amendment before a fresh implementer changes the capture harness.
The original generated GGUF files, 19 codecs, projections, prompts, sampling,
three repeats, four generated tokens, and logical native/primary limit remain fixed.
The amendment changes configuration and unused-input allocation bookkeeping.
It authorizes no decoder, model arithmetic, tokenizer, or shared loader repair.

#### Primary configuration

The plugin's `weights_adapter/qwen3_5.py:38-42` maps the text model type to
`Qwen3_5ForConditionalGeneration`. Its `config_parser.py:43-53` and
`weights_adapter/qwen3_5.py:225` restore that architecture during configuration
and loading. Overriding the architecture field alone cannot survive that chain.
Use the pinned vLLM `model_class_overrides` argument with this exact mapping:

```python
{"Qwen3_5ForConditionalGeneration":
 "vllm.model_executor.models.qwen3_5:Qwen3_5ForCausalLM"}
```

At the active pin, `vllm/config/model.py:319-325` defines this development
argument and lines 1032-1055 apply it in the front end and workers.
`vllm/model_executor/models/registry.py:202` already registers that text class.
Its definition in `models/qwen3_5.py:325-451` uses the same Qwen3.5 model,
forward path, and weight loader. Record the development argument as a harness
adaptation. Preserve the default compiler, graph capture, and kernel dispatch.
Do not add a vision configuration or change the oracle source.

Pass `hf_overrides={"partial_rotary_factor": 1.0}`. The unchanged GGUF requests
rotary width 64 with head width 64 and sections `[16, 8, 8, 0]`.
The pinned `vllm/transformers_utils/configs/qwen3_5.py:86` defaults the top-level
factor to 0.25. The executing Transformers `modeling_rope_utils.py:787-788`
copies that factor over the nested value during normalization. The original
configuration therefore resolves rotary width 16 and fails its section check.
The explicit top-level factor preserves the existing model geometry.
The retained CPU probe demonstrates both resolutions using the executing image.

#### Physical cache capacity

The pinned `vllm/v1/core/kv_cache_utils.py:2288-2309` allocates the requested
physical blocks and subtracts one null block for admission. Four blocks of 16
leave 48 usable tokens and cannot admit `max_model_len=64`.
The v4 primary attempt proves this refusal after successful model loading,
normal compilation, and graph capture. The secondary pins independently pad
context to 256 in `src/llama-context.cpp:288` for stock and line 285 for the fork.

Use 16 physical blocks of 16 in the native public test and primary harness.
This matches the secondary physical payload of
`256 * 1 * (64 + 64) * 2 = 65,536` bytes for BF16 K and V.
Native `LoadedEngine::ResolveNumBlocks` accepts explicit block counts;
`vllm::v1::BlockPool` reserves the null block. No product change is needed for 16 blocks.
Keep the earlier four-block measurements with their original values and mark
them superseded for matched oracle memory. Do not relabel or delete their traces.
The existing issue #2719, owned by `KV-SIZING`, tracks the separate native
admission gap. This row neither repairs that gap nor claims it closed.

The operator's v5 Q4_0 attempt exits 0 and emits `[47, 19, 4, 20]` for the first
prompt and repeat. The unchanged GGUF SHA256 is
`663597e28852d64097f7a67542675e4575ee158d3c5e6e2143aadd1b4188f4af`.
Its resolved dtype is BF16 and its cache reports 16 blocks and 256 cells.
The operator verified 2,500 inputs before and after the run. This proves one
bounded execution. It does not prove the remaining formats, repeats, token
equality, or observed cache tensor layout and allocation bytes.

#### Secondary allocation overlay

Keep pristine source trees and binaries for both recorded secondary pins.
The stock and fork create an unused recurrent copy input for this one-layer,
full-attention model. Stock `src/llama-graph.cpp:3397-3416` creates `s_copy`;
lines 3481-3489 attach hybrid inputs even though no recurrent layer consumes it.
The fork has the corresponding functions at lines 3379 and 3463.
The graph allocator allocates graph leaves and nodes. Its input setter later
writes the unallocated tensor. Stock `src/llama-context.cpp:1379-1380` explicitly
documents this unused-input failure; the existing model logs reach its buffer
assertion on the first decode. The allocation diagnosis still requires a runtime
red/green witness that identifies `s_copy`.

Build separate row-owned overlay copies of the exact pins. In
`llm_graph_context::build_inp_mem_hybrid`, retain the recurrent copy input as a
graph leaf with `ggml_build_forward_expand(gf, inp_rs->s_copy)` after creation.
This scopes one allocation repair to each pinned `src/llama-graph.cpp`.
Keep the buffer assertion and all existing input writes. A blanket null-buffer
skip could hide a required recurrent input and is not an admissible repair.
Do not change weights, codec arithmetic, graph computation, or model topology.

Qualify this overlay only for the bounded synthetic model in this row.
Record pristine pin, exact patch, patched-source digest, build flags, executable
and linked-library digests, model hashes, and runtime outputs. Describe every
model result as the pinned secondary plus that named overlay. The independent
codec gates continue to execute pristine binaries and must preserve their output
hashes. No global oracle pin or gateability record changes. Issue #933 still
owns the published 2.4T fork qualification.

Capture the pristine model failure and overlay success on identical inputs.
Run both prompts three times through fresh engines for stock MXFP4 and fork
IQ1_XXXS. Require exact token equality to native and repeated-run agreement.
Run a control with an actual recurrent layer on each pristine and overlay
binary; require identical tokens and logits. A controlled hybrid-input test may
exercise recurrent state instead if no suitable model is available, but it must
consume the recurrent copy tensor and verify state values through graph reuse.
No missing control can become a source-only pass. A fresh reviewer removes the
retained leaf and requires the bounded-model failure to return. Independently
verify the unchanged recurrent path and pristine codec output hashes.

#### Scoped implementation and verification

The fresh implementer may change `tools/rocm_quant_gather/primary_model.py`,
`tools/rocm_quant_gather/secondary.cpp`, and row-owned capture, patch, comparison,
and evidence files. They may set 16 blocks in
`tests/capi/test_rocm_embedding_quant.cpp`, preserving its public ABI entry and
all fixture bytes. An additional public-ABI capture runner is allowed if needed;
it must use `include/vllm.h` and preserve that test's complete workload.
No shared product file or checker semantics enters this amendment.

Retain meaningful failing primary and secondary launches, then capture focused
green with these corrections. Rerun all 19 public formats and dense controls at
16 blocks. Compare all 16 primary model formats, stock MXFP4, and fork IQ1_XXXS
on both prompts and all repeats. The Q8_K decoder-only adaptation remains exact.
Measure native and oracle model memory with the same tool, recording actual
cache dtype, physical payload, layout, reserved capacity, and allocation overhead
separately. A config string alone cannot establish this memory gate.
Fresh review removes the registry and RoPE corrections independently, checks
that incorrect cache capacity fails comparison, and mutates the secondary leaf
as described above. Run focused and full gates, obtain PASS review on an
immutable head, and have the operator rerun the final gates.

The [amendment evidence](../../docs/bench-evidence/rocm-quant-gather/oracle-amendment/README.md)
retains the primary launch progression, CPU RoPE probe, receipts, and hashes.
The [qualification evidence](../../docs/bench-evidence/rocm-quant-gather/oracle-qualification/README.md)
retains the implementation's runtime failures, secondary overlays, recurrent controls,
live primary cache measurements, and matched 16-block native gates.

## Tests and gates

### Primary report qualification amendment

Issue [#3113](https://github.com/mudler/vllm.cpp/issues/3113) owns the comparator repair from fresh review of `75d92614e84ff428eab49324f3b4fb2aba183ac3`.
Commit this amendment before changing the comparator or adding its regression tests.
The scope is `tools/rocm_quant_gather/compare_models.py`, focused Python tests,
and this row's evidence. Model bytes, workload, pins, and product code remain fixed.

The token matrix and separate memory capture must pass the same qualification.
Require the exact successful status and reject any exception or traceback field.
Require both recorded oracle pins and every recorded runtime identity:
vLLM `0.28.1rc1.dev132+ge126687a9`, plugin `0.0.5+d4c1f0d.gfx1100`,
Torch `2.12.0+git6bbd260`, Torch revision `6bbd26020da1c6dc198625dfcdd968b1e4e6b1c5`,
and HIP `7.2.53211`. These values come from the sealed successful captures.
They qualify this local runtime and do not advance any global oracle pin.

Require identical model and configuration seals, prompt, repeat, resolved BF16,
and every requested field, including automatic model and KV cache dtypes.
Reject missing fields and incorrect values in either capture.
Require a zero cache storage offset and verify that the complete strided view
fits its storage before checking the existing live allocator bounds.
Require the same view and storage metadata before and after generation.

Preserve the actual successful Q4_0 token and memory reports as positive controls.
Synthetic validator cases must be labeled as tests, never as oracle measurements.
First reproduce the post-generation exception acceptance through `compare`.
Then test missing and incorrect qualification fields on both report paths,
including independent exception fields and invalid storage offsets and bounds.
Mutate each new guard and each qualification call site, require focused failure,
and restore the source byte-for-byte. Run the focused suite and full preflight.
The operator owns the complete actual matrix comparison, GPU gates, and publication.

### Red-first implementation gate

Add the public production test before the native provider.
Confirm that compressed-gather selection is absent on the base and capture the red result.
Matching dense outputs alone must not pass the test.
Then add decoder tests, implement the provider, and get focused green.

The decoder suite must cover:

- All 19 formats, with both i32 and i64 IDs and both f32 and bf16 outputs.
- One-, three-, and five-block rows, plus primary widths 256 and 1024.
- First, middle, and last rows, repeated IDs, and reversed row order.
- Explicit packed-table offsets 0 to 3, including odd block byte sizes.
- Multiple launch blocks and a selected-block count greater than `1024 * 128`.
- Exact f32 bytes and bf16 words against scalar controls and committed oracle vectors.
- Finite random block cases with deterministic seeds and recorded block bytes.
- Negative IDs, `id == vocab`, and i64 values that would wrap into valid i32 IDs.
- Successful execution immediately after a bounds-error exception.
- Empty requests, empty vocabulary, ragged widths, and invalid ranks and shapes.
- Invalid table, ID, and output dtypes, noncontiguous views, and device mismatch.

The decoder-set test must enumerate through the actual last enum `kIQ3_S`.
Do not copy the older CUDA loop endpoint `kIQ4_XS`.
Keep the committed vectors for IQ4_NL, Q5_0, IQ1_S, IQ1_XXXS, IQ2_XS,
IQ4_XS, and IQ3_S in the regression set.
The upstream runtime comparisons supply independent evidence for every format.

### Focused native command

Resolve `.env` and the local compute preference before any GPU command.
The operator owns GPU scheduling and holds the recorded mutex for this local host.
Fleet devices require their lease if the task moves to one.
Build only in this task's ignored build directory, with at most four compile jobs.

```sh
cmake -S . -B build-rocm-quant-gather -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DVLLM_CPP_CUDA=OFF \
  -DVLLM_CPP_HIP=ON \
  -DVLLM_CPP_HIP_ARCHITECTURES=gfx1100 \
  -DVLLM_CPP_BUILD_TESTS=ON
cmake --build build-rocm-quant-gather -j 4
ctest --test-dir build-rocm-quant-gather --output-on-failure \
  -R '^(test_rocm_embedding_quant|test_capi_rocm_embedding_quant|test_gguf_keep_quant|test_ops_embedding_quant|test_gguf_dequant|test_backend_cross_device)$'
```

The final implementation must make these target names and commands accurate.
A missing GPU can produce an honest exit-77 skip in generic CI.
The acceptance run on gfx1100 requires execution and cannot pass with a skip.
Run the existing ROCm regressions and the full applicable preflight after focused green.
Run CPU-only configuration checks to verify that additive HIP files remain gated.
Run the combined CUDA/HIP expectation tests when that build is available.
A missing combined backend resource remains explicitly `PENDING` if applicable.

### Fresh review mutations

A fresh reviewer reviews the immutable implementation head.
Use a scratch copy and restore every file byte-for-byte after each mutation.
Require the focused test to fail for the intended reason after each mutation.

1. Remove the ROCm registration.
2. Delete the production `vt::Embedding` call in `EmbedInto`.
3. Force the loader's embedding residency back to expanded bf16.
4. Corrupt each of the 19 decoders independently through scale, payload, or codebook logic.
5. Replace the byte row stride with an element stride.
6. Truncate i64 IDs to i32 before bounds validation.
7. Replace bf16 rounding with truncation.
8. Remove the bounds-error record or its reporting.
9. Remove grid-stride continuation.
10. Remove the reader's Q8_K trait, then independently mis-size that trait.
11. Remove the embedding materializer role and independently weaken its default dot guard.
12. Permit a matrix repack for the embedding role.
13. Remove the queue-device binding before the error allocation and launch.

The first three mutations must fail the public production test.
The remaining mutations must fail their named focused cases.
Record mutation diffs, commands, exit codes, failure excerpts, and restoration hashes.
A fresh implementer repairs each finding. The coordinator does not repair findings.
The operator reruns the final focused and full gates itself after a PASS review.

### Acceptance results

Report exactly one state per applicable obligation: satisfied, narrowly waived,
pending a named authority or resource, or failing.
The current result inventory is:

| Obligation | Result | Authority or remaining evidence |
|---|---|---|
| Committed spec before implementation | Satisfied | Original spec and materializer/reader amendments precede product edits; bounded-oracle amendment precedes its harness implementation |
| All 19 native decoder and error contracts | Satisfied on sealed native v4 | Nine cases and 11,784 assertions; required two-device case passes 16 assertions |
| All 19 public production paths | Satisfied | Sealed matched16 run passes 228 completions and 2,850 assertions at 16 physical blocks |
| Pinned-oracle model token equality | Pending a named upstream authority | The 96-request primary matrix executed; 93 requests are token-identical to native. The three IQ3_S first-prompt cases are oracle-side: the pinned plugin's IQ3_S grid decodes 62 where the format-defining llama.cpp grid has 15 (150 of 2048 byte positions, +3.333%), native is bit-exact with llama.cpp, and a CPU substitution experiment reproduces the plugin tokens exactly. Reported upstream as `vllm-project/vllm-gguf-plugin#129`; the row cannot pass those three cases before that grid is corrected. All 12 stock MXFP4 and fork IQ1_XXXS requests match native. |
| Primary plugin operation execution | Satisfied | Exact native plugin, 64 synthetic outputs; original tolerances unchanged |
| Primary bounded-model execution | Satisfied | Operator v5 emits four tokens from the exact Q4_0 fixture with normal compilation and graph capture; the full 96-request primary matrix executed and its comparison is recorded (93 token-identical, the three IQ3_S cases classified above) |
| Original primary fixture coverage | Satisfied | The 32 fixtures were downloaded with the developer's authority and sha256-verified at 289,655,872 bytes (`/home/vikash/models/test-gguf-sample`, receipt `gather-3113-repair/fixture-download-receipt.json`). `export-upstream` produced 160 tensors over the 16 primary codecs; native versus pinned-plugin comparison passes 320 outputs with 140 byte-exact and every remaining element inside the upstream `atol=0.01 rtol=0.04` (`original-fixture-coverage/compare/report.json`). IQ1_M and the F16 output case remain the documented exclusions; Q8_K, MXFP4, and IQ1_XXXS are secondary codecs outside the 32-file set and keep their separate evidence. |
| Stock secondary operation execution | Satisfied | Q8_K direct pinned decoder and MXFP4 real GET_ROWS, including applicable stock geometry |
| Stock secondary bounded model | Satisfied with bounded overlay | Six MXFP4 requests match native; pristine/overlay recurrent tokens and logits remain exact |
| IQ1_XXXS fork operation execution | Satisfied | Real pinned fork GET_ROWS; exact outputs and applicable stock geometry |
| IQ1_XXXS fork bounded model | Satisfied with bounded overlay | Six IQ1_XXXS requests match native; pristine/overlay recurrent tokens and logits remain exact |
| Operation memory format and identical-tool traces | Satisfied | Identical rocprofv3 1.3.5; native allocation events and separate primary tensor telemetry |
| Native model memory format | Satisfied at amended cache capacity | All 228 runtime allocations report BF16, 16 physical blocks, and 65,536-byte payload; independent model traces remain required below |
| Matched oracle model memory | Narrowly waived, scoped | The native and primary pair both measure a 65,536-byte BF16 cache (16 physical blocks of 16 cells); the live primary tensor and its allocator block agree. The secondary llama.cpp oracles pad the requested 64-cell context to that runtime's own 256-cell minimum, so the matched-memory claim is scoped to the native/primary pair, and secondary model executions are token/logit comparisons that make no memory claim. Identical-tool backing-allocation measurement of the secondary runtime remains outside this claim. |
| CPU regressions | Satisfied | Amended CPU build passes six tests and explicitly skips two ROCm-only tests |
| ROCm regressions | Satisfied | Clean HIP build; all 15 CTests and required two-device case pass in sealed matched16 run |
| Applicable local preflight | Satisfied | Full preflights exit 0 at the amended heads `fc0fae3ff` (gather), `6607faea2` (F16), `6fd1650c4` (BF16 MoE), and the shared checkout, each with its argument-dependent skips recorded separately. |
| Combined CUDA/HIP expectation execution | PENDING | No combined build was available for this qualification |
| Fresh mutation review | Satisfied for the comparator chain | Fresh review of `c5bb0f1f7..92813f4b` returned PASS with two findings for the repair loop and no blocking defects; the guard-pinning repair `fc0fae3ff` closed them and a second fresh review returned PASS with no findings. Both reports are retained. The first is `/home/vikash/.cache/rdna3-gather-oracle-review/review-92813f4b/REVIEW.md`, sha256 `968f862b7b62dcc7c0a80ddcdb9d39b7330a651046db233c818ac9ec818cced7`, with the reviewing session's scratch artifacts beside it. The second is `/home/vikash/.cache/gather-3113-repair/review-fc0fae3ff/REVIEW.md`, sha256 `a8fc4589d96cd936bb7f48764c955d83c48c80d1e64677e54b043eb347fe2998`, with that session's four guard-removal logs and `pristine.sha256` beside it. Each file is an operator transcription of its reviewing session's final report, not a file the reviewing session wrote itself. |
| Operator verification | Satisfied for the comparison and preflight | The operator ran the 96-request token comparison, the comparator's guard-removal red receipt, the original-fixture coverage comparison, and the full preflight at `fc0fae3ff`. |

Performance is not an acceptance claim in this correctness row.
No benchmark ID, throughput ratio, latency claim, or performance ceiling changes.
This exclusion does not discharge the parent row's performance parity debt.
A future performance claim requires identical workloads, correctness first, and
measured memory, latency, and throughput axes under the repository gates.

## Evidence

The spec author inspected the code, owning records, current issues, pull requests,
and symbol history before claiming implementation scope.
At the inspected base, ROCm registers floating embedding but no quantized gather.
The existing quantized dot registration already supplies the loader policy default.

Source inspection on 9 September 2026 UTC verified these local read-only checkouts:

- Stock source: `/home/vikash/oracle/llama.cpp-b10451-clean`, HEAD `10bf611e533d81f739128304991c5e133c6aebd8`.
- Active vLLM source: `/home/vikash/oracle/gfx1100-active-2773/source`, HEAD `e126687a9a828d513c01a07cd69f025f27d63280`.

These paths describe measured local evidence. They are not defaults for another host.
Resolve every execution path from the operator's supplied environment.

The baseline `scripts/agent-preflight.sh --quiet` exited 0 on the spec base.
It reported no failures and five argument-dependent skips. These were
`check-arm-isa-build.py`, `check-cpu-isa-build.py`, `check-cuda-fat-gencode.py`,
`check-pr-size.py`, and `check-triton-aot-multiarch.py`.
This result is not an all-green preflight. The spec changes no build or product file.
The implementation's final staged preflight exits 0 with no failures and the same five argument-dependent skips.
The explicit CPU ISA check passes. Final commit classification accompanies the immutable handoff.
Implementation evidence must add exact source and binary revisions, build commands,
model and fixture hashes, environment, mutex identity, contention, and exit codes.
Keep oracle outputs and restoration hashes in this row's evidence directory.

Implementation captures, matching operation traces, oracle output archives, and
remaining qualification failures are retained in the linked row evidence.
The first public red was invalidated by the separate full-attention state defect;
only the subsequent zero-provider-selection result is the intended implementation red.
Q8_K reader and materializer refusals were then captured in their executing order.
The queue-device test captured invalid stream use before the binding repair.
Do not commit downloaded sample GGUF files.

## Risks and stop conditions

The registration changes compressed residency for every admitted embedding codec.
A missing decoder therefore affects model loading beyond the test's first format.
Odd byte strides, bf16 ties, and i64 bounds require independent mutation evidence.
A provider counter can prove dispatch and still miss hidden host decode.
The memory trace closes that evidence gap.

Return `NEEDS_CONTEXT` when a required path, fixture, pin, or authority is unavailable.
Keep the associated gate `PENDING` and continue independent authorized work.
Return `NEEDS_DECISION` for a required scope or model-harness change.
Stop acceptance if the oracle cannot run, a codec remains untested, or production
mutation evidence does not fail for the intended reason.
Do not weaken a tolerance or remove a format after observing a failure.
Do not mark the row `DONE` from partial gates or source feasibility.

## Owed

Issue #3093 owns this row's remaining implementation and verification gates.
Parent issue #2394 retains the other backend gather arms and closes only when
its remaining scope is discharged.
Issue #933 retains real 2.4T fork qualification, independent of the bounded
synthetic vehicle in this row.

Add an `Outcome` section only when this row reaches `DONE`.
Record measured results, rejected alternatives, and the reasons for final defaults.
