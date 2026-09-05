# T21: Q5_K/Q4_K keep-quant with V-head row permutation

**Date:** 2026-08-26
**Branch:** `row/GFX1100-TG200`
**Status:** ADOPTED

## Change

The GDN (linear attention) layers' `attn_qkv` (Q5_K, 24 tensors [2560,8192])
and `attn_gate` (Q4_K, 24 tensors [4096,2560]) were expanded to bf16 at load
time because the V-head row reorder (`ReorderVRows`) classified them as
`kTransformedWeight`. The reorder is a ROW permutation — quantization blocks
are along the K (column) dimension and are self-contained per row — so it is
block-safe. T21 routes these tensors as `kMatmulWeight` to allow keep-quant,
copies the blocks via `OwnGgufQuantBlocks(mmap_src=nullptr)`, and applies
`ReorderVRows` to the block bytes at load time.

The forward pass already dispatches quantized `nk=true` weights through
`vt::MatmulBT` → `matmul_bt_quant`, so no forward-pass change was needed.

**Env gate:** `VT_GDN_ROWPERM_KEEP_QUANT=0` forces the old bf16 expansion path
for A/B isolation. Default is enabled (1).

## Files changed

- `src/vllm/model_executor/models/qwen3_5_gguf_weights.cpp`:
  - Added pointer-based `ReorderVRows(uint8_t*, ...)` overload for `OwnedBytes`
  - Added `VT_GDN_ROWPERM_KEEP_QUANT` env gate and `rowperm_role` routing
  - `attn_qkv` and `attn_gate` sections: new `kKeepQuant` path with in-place
    block row permutation

## Gate

```
[doctest] test cases:  16 |  16 passed | 0 failed | 0 skipped
[doctest] assertions: 839 | 839 passed | 0 failed |
```

## A/B

Interleaved pairs ×5, `--max-tokens 256 --temperature 0 --seed 0`, pinned
analytic prompt, campaign config (11 flags, `VT_NORM_QUANT_FUSED=1` set by
test internally). Loadavg 0.61–1.55.

|Pair|OFF tok/s|ON tok/s|
|---|---|---|
|1|87.336|91.053|
|2|87.460|90.994|
|3|87.381|90.840|
|4|87.303|90.867|
|5|87.168|90.862|
|**Median**|**87.381**|**90.840**|

**Improvement:** +3.9% (90.840 / 87.381 − 1). 5/5 pairs consistent.

## Body coherence

Outputs diverge at line 21: OFF says "RNNs/Transformers", ON says
"RNNs/LSTMs" — both valid descriptions of the same concept. Divergence is
expected: Q5_K integer dot product vs bf16 float MAC produces different
logits, causing a different argmax token that cascades through autoregressive
generation. Both outputs are coherent English covering the same topics.

Not byte-identical (1041 vs 1068 bytes). This is expected for a quantized vs
bf16 GEMV path change.

## Analysis

The +3.9% improvement is less than the projected ~14%. The projected savings
assumed 1023 MB/tok of bf16 read amplification eliminated at ~547 GB/s, but
the actual savings is ~0.4 ms/tok × 547 GB/s ≈ 219 MB. The discrepancy is
likely because:

1. The Q5_K GEMV kernel (`KQuantGemvMmvqK<Q5_K>`) has lower effective
   bandwidth on small grids (n=2560) than the 450 GB/s assumed.
2. The `wvSplitKSml` bf16 GEMV is more efficient on these specific grids than
   the 700 GB/s assumed, reducing the savings from removing those calls.
3. Additional dispatch overhead for the new Q5_K GEMV calls.

On an idle host, the improvement scales to ~107 tok/s (from 103 baseline).

## Path to 200 tok/s

T21 brings the projected idle-host throughput to ~107 tok/s. The remaining
path:
1. Improve Q5_K GEMV bandwidth on small grids (n=2560)
2. Fuse `QuantizeQ8KK` (0.54 ms/tok, 40 calls/tok, 78% threads idle)
3. Improve overall GEMV bandwidth to ~700 GB/s
4. Q8 KV cache or RmsNorm fusion
