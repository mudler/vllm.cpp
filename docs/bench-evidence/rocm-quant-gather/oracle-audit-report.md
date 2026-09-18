**PASS: frozen operation-oracle evidence.** No finding invalidates the 64 primary operation outputs or six secondary F32 operation outputs. This is a bounded source and evidence audit, not fresh review of the gather implementation. Model qualification, production reachability, mutation testing, original fixture coverage, and trace acceptance remain outside this PASS.

The reviewed spec is `.agents/specs/rocm-quant-gather.md` at `2e5adfb85a255528bffa5d3cdb016efb2bab70d8`. The read-only review worktree is `/home/vikash/vllm.cpp-rdna3-gather-oracle-review`. It remains clean. The auditor performed no GPU run, dependency installation, download, tracked-file edit, publication, merge, or CI action.

The independent identity audit verified:

- All 80 files in primary freeze `6bc9501cf0abe9a8de36bcc652b636ff5a26df3dbb650ad38dcc0ab098fee9f8`.
- All 53 files in secondary freeze `4baa265386ee5d67a0dfb056b42a4650d65a29a8446e3df5a30a7778980b6f21`.
- All 138 plugin artifact files, including extension `592af79c210496e96eaa7bd409ae0a67e7de8c334aa9f782082c7639ba300138`.
- All 150 plugin archive files against Git blobs at `d4c1f0d082fc7cd4350da56689109a01c1f29d6c`, the extracted files, and 122 original installed source files. Archive SHA-256: `090d8fd5b6cc52404508fbd4d77f38c8f9d04696b98c778e9220447f3310df39`.
- All 3,425 stock archive files against Git blobs at `10bf611e533d81f739128304991c5e133c6aebd8` and the extracted files.
- All 3,317 fork archive files against Git blobs at `36fe8e1cc7f2b3b8c92fdda0ab07600141921786` and the extracted files.
- The active vLLM source HEAD is `e126687a9a828d513c01a07cd69f025f27d63280`, with no tracked changes. Executing runtime copies of `vllm/utils/torch_utils.py` and `vllm/platforms/rocm.py` equal that source byte-for-byte.

The primary harness SHA-256 is `1ddd29da328db6817b5b15119c228cad55b1f2d4e5acc143c80d70966a5c2d75`. At `primary.py:88-102`, it loads original packed bytes as uint8, restores four i32 file IDs to a 2-by-2 torch.long tensor, and executes the registered plugin operation. The dense control calls upstream `gguf.dequantize`, converts the dense tensor to the requested output dtype, and calls `torch.embedding`. Its `atol=0.01`, `rtol=0.04`, ID arrangement, widths 256 and 1024, and dtype conversion agree with plugin `tests/test_kernels.py:102-124`. IQ1_M and F16 exclusions are explicit.

All 64 output hashes match report `867aaf215887b2c18f46b20b9211592550274099f721b4fff73cefb2f171dd38`. Every output is finite and nontrivial. The three distinct selected rows differ, and repeated row 0 reproduces identical bytes. Every input ID vector is `[0,127,64,0]`. Output and selected-packed byte counts agree with the tensor geometry. These are generated 128-row tables. They do not preserve or replace the 32 original upstream GGUF fixtures.

The actual primary call chain is supported by source and the operator's successful run:

- Plugin `quantization/vocal_embeds.py:112-118` registers the operation through active vLLM `utils/torch_utils.py:1068-1079`. `platforms/rocm.py:502` supplies the CUDA dispatch key used for HIP tensors.
- Plugin `vocal_embeds.py:79-96` flattens IDs, selects packed rows with `torch.index_select`, dequantizes only those selected rows, and restores the output shape.
- Plugin `ops.py:93-101,181-186` uses exactly the predicate logged as `native_extension`. All 64 recorded values are true. The harness does not alter that predicate or dispatch.
- `csrc/torch_bindings.cpp:43-44` selects `ggml_dequantize`. Generated `csrc/gguf/gguf_kernel.hip:88-102` allocates the requested output dtype and dispatches `ggml_get_to_cuda<scalar_t>`.
- Generated `dequantize_hip.cuh:530-569` selects the format decoder. Its launch wrappers are at lines 440-526. Generated source copies and hashes are retained in `generated-primary/` and `executing-source-artifacts.json`.

This establishes the native HIP path by source-plus-run evidence. It is not a profiler trace. No Triton kernel execution or invocation-parity claim is made. The capture's dense control intentionally expands a full table; total-process memory cannot therefore be attributed entirely to the compressed plugin operation.

The secondary harness SHA-256 is `09773359f3db39d5c759d03e37e42f4e466f6e33cb063cd2eb9bbf1353b04dbb`. Its CMake target links the exact private CPU oracle builds. No decoder is transcribed in the harness.

- `secondary.cpp:55-61` calls stock `ggml/src/ggml-quants.c:2807-2815::dequantize_row_q8_K` directly for each selected row. Stock GET_ROWS excludes Q8_K at `ggml/src/ggml-cpu/ops.cpp:5023-5052`. This is the specified adaptation.
- `secondary.cpp:68-80` builds and computes an actual GET_ROWS graph for MXFP4 and IQ1_XXXS. Stock `ggml.c:3891-3909` creates the F32 output. `ggml-cpu.c:1864-1866,3432-3437` executes the CPU graph.
- Stock `ggml-cpu/ops.cpp:4850-4890,5032-5051` selects `to_float`. `ggml.c:751-756` routes MXFP4 to `ggml-quants.c:569-586`.
- Fork `ggml-cpu/ops.cpp:5063-5069` admits IQ1_XXXS. `ggml.c:871-876` routes it to `ggml-quants.c:2727-2751` and the fork's own codebook.

All six operator output files match their recorded hashes and the author's prior outputs byte-for-byte. The stock binary hash is `7b9a2e2d0dc95afbf592b5b1678bdc67e0fe2204529f1afe3b0b1f03566f28c2`. The fork binary hash is `cad537f239dc0c4ca70e5b80548f39f297117494269ca2b9b5de8191aa2f4bc1`. This capture uses contiguous synthetic tables and IDs. It does not claim the complete upstream test parameter matrix or secondary BF16 execution.

The operator requested an additional numerical explanation. `arithmetic-explanation.json` reproduces all elements of 16 captured files: Q6_K and IQ3_S, two widths, two output dtypes, and both native and primary arms. These CPU calculations are diagnostic transcriptions of source expressions. They are not replacement oracle results.

Q6_K differs through intermediate precision. Plugin `dequantize_hip.cuh:248-257` rounds `scale * quant` to FP16 and multiplies in FP16 before output conversion. The local implementation uses F32 products. For example, `115 * -19 = -2185` becomes `-2184` in FP16. With delta `1/2048`, the outputs are `-1.06689453125` and `-1.06640625`.

IQ3_S differs through its codebook, not rounding. Across 512 words, 150 byte positions in 146 words differ exclusively as plugin `62/4 = 15.5` versus local `15`. Plugin `ggml-common_hip.h:686` defines that table, and `dequantize_hip.cuh:347-353` applies its quarter-scale factor. The local codebook matches the stock representation. The 3.333% difference relative to local magnitude fits the original 4% relative tolerance. It is not byte equality or model token equality.

The diagnostic initially rejected a changed native header hash. That failed guard remains in `explain_arithmetic-v2-source-guard.py`. The successful calculation uses a copied current header with SHA-256 `56b14d63308a6c3d8c19a6619217470fac88dd58599f95b4d9bf919f854f732d`, after the author's comment corrections, and table hash `d267d25ff1dd3f67749c251f87047f4b70c9650a99d04b524ff6c3313596a349`. Its comparison files are the frozen v2 runtime outputs. It makes no exact-v2 source-seal claim. The operator comparison report is `84bac44682927c28a9921e0dd464c1474039d53633fe53d008a14887cf807024`.

One medium-severity qualification concern remains: secondary model cache geometry differs from the committed specification. `secondary.cpp:99-106` requests context 64 and BF16 KV. Both logs report 256 KV cells and approximately 0.06 MiB of KV payload. Stock `src/llama-context.cpp:288,294` explicitly pads context to 256. The spec requires 64 cells and 16,384 bytes. Record and resolve this upstream padding adaptation before claiming matched model memory.

Both secondary bounded models abort on the first decode with SIGABRT. The observed stack includes `llm_graph_input_mem_hybrid::set_input` and `ggml-backend.cpp:194::GGML_ASSERT(buffer)`. Source `llama-graph.cpp:1078-1079` is consistent with an unallocated recurrent `s_copy` input. That object attribution remains an inference. Neither run emits tokens. Primary bounded-model qualification remains PENDING as separately recorded by the operator. The 32 original fixtures, model token equality, model cache adaptation, trace gate, and fork issue #933 remain PENDING under their existing ownership.

Reproduction commands:

```sh
python3 /home/vikash/.cache/rdna3-gather-oracle-review/audit_frozen.py
python3 /home/vikash/.cache/rdna3-gather-oracle-review/explain_arithmetic-current-source.py
```

Both commands exited 0. The first writes `independent-audit.json`; the second writes `arithmetic-explanation.json`. Neither command uses a GPU or changes tracked files.

`REV-MUTATION` and product `REV-FULL-GATE` are N/A for this explicitly scoped read-only evidence audit. The startup `scripts/agent-preflight.sh` completed with exit 1: one `test_agent_onboard` failure and 12 skips. The failure expected scratch Git branch `master` but inherited `main`. Its isolated rerun passed all 39 tests with `GIT_CONFIG_GLOBAL=/dev/null`, an external `TMPDIR`, and matching `GIT_CEILING_DIRECTORIES`. Exact command, environment, and output are retained in the isolated-test artifacts. The startup preflight is not reported as green. No second full preflight ran.
