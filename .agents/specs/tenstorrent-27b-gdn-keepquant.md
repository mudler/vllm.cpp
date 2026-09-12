# SPEC: W4d W4 — packed row-reorder keeps the GDN projections block-encoded

Row: `BACKEND-TENSTORRENT-KEEPQUANT`. Issue:
`ISSUE-LOCAL-01M2AA4ZVD9EWJG5NNQD5DWZXS` (the 27B DRAM OOM) and
`ISSUE-LOCAL-01M2ACXRJYFW7R7BP2ABQS3VY2` (the placement probe, landed in
#3165). Parent spec: [tenstorrent-keepquant.md](tenstorrent-keepquant.md).

## Problem (measured 2026-09-12, /tmp/census3.log + /tmp/chunk64-run.log)

The Qwen3.8-27B Q4_K_M engine holds ~12 GiB of bf16 projection forms on the
P150 beside its ~17 GiB keep-quant word shadows: 97 x [10240,5120] bf16
persistent stagings (9.7 GiB, attn_qkv) plus the dev-slot attn_qkv/GDN
forms (2.2 GiB + 2.6 GiB). The banks run at 99 percent during generation
and the engine OOMs. The file carries these weights q6_K/q4_K/q5_K
(attn_qkv q6_K = 21.5 MiB packed vs its 100 MiB bf16 staging — 4.7x).

## Root cause (code-anchored)

`qwen3_5_gguf_weights.cpp:1104-1112`: when the GDN V-head reorder is active
(`num_v != num_k && num_v % num_k == 0` — true on the 27B), the GDN
projection family (`attn_qkv`, `attn_gate`, `ssm_out`) is routed as
`kTransformedWeight`, which `RouteGgufTensor`
(gguf_keep_quant.cpp:244) can never take keep-quant: the reorder was
implemented on the DEQUANTIZED elements (`ReorderVRows`), so the loader
dequantizes to bf16, reorders, and stages bf16 — the expand arm — even
though the running device executes keep-quant dots for every one of these
encodings (ffn_down, same q6_K, decodes on-core today).

## Design

The reorder permutes WHOLE V rows (`row_off = 2 * key_dim`, rpk whole
rows). A q6_K/q4_K/q5_K row is a contiguous run of whole blocks
(K = 5120 = 20 x 256 blocks per row), so the permutation acts on row
byte-ranges and never touches block internals:

1. `ReorderVRowsPacked(packed_bytes, row_bytes, in_dim, row_off, num_k,
   rpk)` — the byte-level twin of `ReorderVRows`: copy whole row
   byte-ranges in the same permutation. Pure host-side memcpy loop.
2. In the GDN projection load: when `reorder` is active AND
   `KeepQuantDType(ggml_type)` AND `DeviceKeepQuantSupported(dt, dev)`,
   build the kept tensor from the PACKED bytes with the packed reorder and
   route `kKeepQuant` (role stays `kMatmulWeight` — the tensor is verbatim
   blocks, just row-permuted). The bf16/expand arm remains the fallback for
   every other case (ragged K, unsupported encoding, cpu_ref).
3. The GDN forward's in_proj matmul consumes the kept tensor through the
   existing `kMatmulBTQuant` chunk decode — no new kernel; the op already
   runs q6_K (ffn_down today).

The same treatment applies to `attn_gate` (in_proj_z) and `ssm_out`
(column reorder — NOTE: out_proj's reorder permutes COLUMNS which sit
INSIDE blocks, so ssm_out is block-UNSAFE and STAYS expand-bf16; only the
row-reorder family converts. That is ~2/3 of the 9.7 GiB).

## Expected effect

~6-7 GiB of bf16 staging becomes ~1.6 GiB of words; with the warm-slot
release (built, `VT_TT_RELEASE_WARM_ROWS`) and the bounded gate context
(`max_model_len=1024`, both in the W3 tree) the 27B fits with margin and
the chunk knob becomes unnecessary.

## Tests (red-first)

1. Packed-reorder equivalence: `ReorderVRowsPacked` output, run through
   `DecodeKeepQuantWordsF32`, equals `ReorderVRows(DequantGgufRowToF32)`
   bit-for-bit (all four encodings, reorder on/off). RED today: the
   function does not exist.
2. Census regression: a 27B-shaped load (or the focused W2 pattern at
   [10240,5120] q6_K with reorder on) asserts NO bf16 slot staging for the
   reordered projection.
3. The 0.8B vehicle gate + backend suite (71/71) + bit-exact sweeps are
   the no-regression net; the 27B gate verdict is the e2e proof.

## Gates

The 27B gate case (`qwen3.8-27B GGUF Q4_K_M ... Tenstorrent,
checkpoint-gated`) against the re-captured device pair, plus the recipe
levers that remain (chunk knob removal is the success signal: default
chunks must survive).

## Stop conditions

- If the packed reorder cannot reproduce the dequant reorder bit-for-bit
  on any encoding, that encoding STAYS expand-bf16 (stop, do not band).
- If the TT keep-quant dot rejects a reordered row shape the bf16 arm
  handled, stop and re-scope.

## Owed

- The V-column (out_proj) keep-quant path is owed a block-safe column
  permutation or stays expand-bf16 — recorded here, not silently.
- The embed-table dequantizing gather (2.54 GiB bf16 token_embd) stays
  owed to MODEL-MM-QWEN4-EXP W6a.
