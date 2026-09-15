# gfx1100 attention validation

Validated on an RX 7900 XTX with HIP 7.15, Clang 23, and rocWMMA 2.2.1.
Native source: `f7b5f8dd9`. Build: Release, `-O3 -DNDEBUG -ffp-contract=off`.
Primary: vLLM `e126687a9a828d513c01a07cd69f025f27d63280`, with production
compilation and graphs. Both engines use the identical pinned
[Gemma 3 4B BF16 text checkpoint](../../USAGE.md#gemma-3-4b-text-weights).

## Correctness and resources

| Gate | Result |
|---|---|
| Original and expanded model workloads | 16 configurations, 3072 exact tokens including warm-ups |
| Controls | WMMA/scalar prefill, graph/eager decode, cache blocks 16/32 |
| Decoder fixtures | 20/20 byte-exact against the primary |
| Prefill fixtures | 10/10 in both controls, unchanged tolerances |
| Focused CTest regressions | 20/20 pass, including Gemma 1B's 48 exact tokens |
| Mutation checks | Detect deleted dispatch, stale graph inputs, and unpinned graph scratch |
| Compiler and runtime resources | Zero VGPR/SGPR spills and private scratch in all four decoder templates and WMMA prefill |
| Compilation | gfx1100, gfx1200, gfx1201; 657/657 affected host translation units |

Physical RDNA4 execution remains unavailable. Independent human review remains
required. The detailed report records unavailable unrelated checkpoint cases
and the scope of each compilation check.

## Performance

Medians of three alternating rounds, eight requests per run, concurrency one,
and 32 output tokens per request. All 18 runs match all 256 measured tokens.
The benchmark binary remains unchanged across runs. Decode rate uses
`1000 / mean time per output token` on both engines.

| Axis | Block 16 primary | Block 16 native | Block 32 primary | Block 32 native |
|---|---:|---:|---:|---:|
| Prefill tokens/s | 6304.61 | 6996.77 | 6689.32 | 6948.02 |
| Decode tokens/s | 66.48 | 66.58 | 44.65 | 70.01 |
| Mean time to first token, ms | 136.805 | 123.271 | 128.937 | 124.136 |
| Mean request latency, ms | 603.474 | 588.743 | 822.624 | 566.926 |
| Sampled peak host RSS, GB | 8.41 | 1.47 | 8.33 | 1.74 |

Block-16 decode is a parity-level result. Block-32 decode is 1.568 times the
primary's throughput on this workload. WMMA prefill is 2.31 and 2.29 times
the corrected scalar control. A separate 4096-token-capacity comparison
measures 11.12 GB native versus 11.33 GB primary whole-device VRAM. The
primary's larger default cache reservation remains the speed denominator.

## Detailed evidence and reproduction

The [archived report and receipts][archive] contain exact build/run commands,
checkpoint hashes, output tokens, resource counts, traces, all repetitions,
and failed experiments. The snapshot is preserved by the separate
`evidence/rdna3-wmma-2026-09-15` tag in `VikashLoomba/vllm.cpp`.
Evidence links use the immutable snapshot commit. Runtime code, regression
fixtures, and validation tools remain in this change. All 44 regression tensors
use one `cases.bin` and a manifest of offsets, sizes, shapes, and hashes.
`tools/rocm_attn_wmma/compiled_gemma_primary.py` regenerates the complete bundle.

[archive]: https://github.com/VikashLoomba/vllm.cpp/blob/b962497b67c7f82d331aa657b25cc5c96533d611/docs/bench-evidence/rocm-rdna3-attention-wmma/README.md
