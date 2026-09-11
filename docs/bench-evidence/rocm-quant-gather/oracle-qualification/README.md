# Bounded model oracle qualification

Row: `BACKEND-ROCM-QUANT-GATHER`. Issue: [#3093](https://github.com/mudler/vllm.cpp/issues/3093).

The committed amendment `e35328644b8a0b57f97388965575c20d122a32dc` precedes this harness implementation.
The original 19 GGUF fixtures, projections, prompts, sampling, and four generated tokens remain unchanged.
The native and primary executions use 16 physical blocks of 16 cells and a logical limit of 64 tokens.
The earlier four-block captures remain in the parent evidence directory.
They are superseded only for the matched model-memory comparison.

## Captures and provenance

[`capture-manifest.json`](capture-manifest.json) seals each file in
[`qualification-captures.tar.gz`](../qualification-captures.tar.gz).
Each execution directory retains exact argument arrays, environment, input hashes, operator receipts, and raw output.
The root operator executed all GPU and secondary-oracle commands under the shared GPU mutex.
The helper compiled with at most four jobs and ran the CPU-only regression build.

Primary vLLM stays at `e126687a9a828d513c01a07cd69f025f27d63280`.
The GGUF plugin stays at `d4c1f0d082fc7cd4350da56689109a01c1f29d6c`.
The runtime image is `sha256:80aab4c182a1f3eeebe286173977e57fcaf10a049b41f475655b35d285de31dc`.
The primary source and production compiler, graph capture, and kernel dispatch remain unchanged.
The harness selects the pinned text class and preserves the original rotary width through the two amended arguments.

Stock llama.cpp stays at `10bf611e533d81f739128304991c5e133c6aebd8`.
The IQ1 fork stays at `36fe8e1cc7f2b3b8c92fdda0ab07600141921786`.
The [stock patch](stock-retain-hybrid-copy.md) and [fork patch](fork-retain-hybrid-copy.md)
each retain `s_copy` as one graph leaf in `build_inp_mem_hybrid`.
The raw `.patch` artifacts were re-homed to documents because `.patch` has no admitted evidence class.
Complete pristine and overlay inventories prove that each patch changes only its intended source file.
These overlays qualify this bounded synthetic model only; global oracle pins and issue #933 remain unchanged.

## Observed failures and corrections

The original primary harness fails during renderer construction.
The pristine stock and fork models abort during the first decode.
Debugger captures identify a nonnull `s_copy` tensor with null buffer and data immediately before its setter asserts.
The first stock debugger breakpoint missed that setter; its failed witness remains archived beside the corrected capture.

The secondary overlays execute both original model fixtures.
All 12 stock MXFP4 and fork IQ1_XXXS completions match the native 16-block captures exactly.
Each format covers both prompts and three fresh engines.
The nine pristine codec outputs retain their prior hashes, including the Q8_K direct-decoder adaptation.
Q8_K has no secondary model-execution claim.

The separate recurrent control contains an actual recurrent layer followed by full attention.
Both pins execute it through pristine and overlay binaries on both prompts and all three repeats.
All 24 captures preserve tokens and every one of the 512 F32 logits byte-for-byte.
Four additional graph controls consume `s_copy` and preserve exact state values through three graph reuses.
The original gather models remain unchanged; the control's shape and tensor hashes are archived separately.

The first two memory observer attempts hang after the worker's allocator snapshot.
The final observer converts the layout Enum to its name and returns a complete JSON string from the worker.
The v3 capture exits 0, emits `[47, 19, 4, 20]`, and observes the same cache before and after generation.
The failed attempts and operator cleanup receipts remain evidence; neither is an oracle configuration pass.

## Physical cache and native gates

The executing primary cache has BF16 dtype, shape `[16, 2, 16, 64]`, and strides `[1024, 16384, 64, 1]`.
Its resolved layout is `LHBNC`, with stride order `[0, 2, 1, 3, 4]`.
The tensor payload, storage, and active allocator block each contain 65,536 bytes.
The allocator backs that block with a 2,097,152-byte segment shared with other allocations.
The process allocator totals are separate fields; they are not KV payload measurements.
Identical-tool model traces remain required to identify the backing allocations independently.

Native runtime records cover all 228 public engines at BF16, 16 blocks, and 4,096 bytes per page.
The complete focused and ROCm regression command passes all 15 CTests.
The required case with a different ambient device passes independently on two visible GPUs.
The CPU build passes six tests and explicitly skips the two ROCm-only tests.
The initial CPU regex selected no tests; the corrected command requires a nonempty test set.

## Remaining gates

The 96 primary model captures, identical-tool model-memory audit, final staged preflight,
fresh mutation review, and operator rerun remain pending at this evidence checkpoint.
The original 32 upstream fixtures still require developer authority for their 289,655,872-byte download.
The combined CUDA/HIP execution remains pending an available combined build.
No throughput, latency, or performance-ratio claim is made by this correctness change.
