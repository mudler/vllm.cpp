# Leaf spec: GGUF keep-quantized loader — quantized weights resident, dequant as oracle

**Row:** `QUANT-GGUF-KEEPQ-LOADER` (quantization-matrix.md, leaf of the
`QUANT-GGUF-COMPUTE` block) · **status:** spike complete, `READY` ·
**upstream pin:** llama.cpp local fork `237ad9b96` (b9892) ·
**parent evidence:** B4 decision row
([parity-ledger.md L290](../parity-ledger.md#L290)): load-time bf16 expansion
costs 2.65× peak RSS (7.43 GiB vs llama.cpp 2.80 GiB on a 2.68 GiB file) and
forces every GEMM through bf16. This leaf keeps supported GGUF weight tensors
in their native block encoding from file to GEMM, and turns the current
dequant-to-bf16 path into the explicit `VT_CPU_REF` parity-oracle mode.

## Scope

- **In scope:** loader-side storage/routing so 2-D matmul weights of the
  supported types (**Q8_0, Q4_K, Q5_K, Q6_K, Q3_K, Q4_0** — the set our
  dequant dispatch already executes) stay in raw ggml block bytes; a
  `GgufQuantWeight` host struct ([N,K] native orientation, `nk=true`
  semantics) consumed by the `QUANT-GGUF-CIQ-GEMM` op; per-tensor routing
  policy (quant-capable 2-D weights stay quantized; norms/biases/other 1-D
  and non-GEMM tensors keep today's dequant); the `VT_CPU_REF=1` switch that
  forces the full dequant-to-bf16 path (parity oracle, existing goldens);
  fate of the unmerged B4 bench branch (decided below).
- **Out of scope:** the GEMM kernels themselves (`QUANT-GGUF-CIQ-GEMM`), the
  threadpool, new encodings (i-quants etc. stay with their encoding rows),
  CUDA residency of quant blocks (future backend row), safetensors loaders.
- **Dispatch behavior:** per tensor — if `(ggml_type ∈ supported set) ∧
  (tensor is a 2-D matmul weight) ∧ ¬VT_CPU_REF` → keep blocks, route the
  matmul through the block-dtype path; else → today's
  `DequantGgufRowToBf16`/`ToF32` materialization, byte-identical to current
  behavior. Unsupported types keep today's explicit rejection.

## Upstream chain

Pinned local fork `/home/mudler/_git/llama.cpp` @ `237ad9b96`.

| Component | Upstream anchor | What it establishes |
|---|---|---|
| Weights stay file-typed | `src/llama-model-loader.cpp:1047` (`create_tensor` keeps the file's `ggml_type`), `:1385` (`load_data_for` copies/mmaps raw block bytes) — llama.cpp never dequantizes weights at load | the structural memory/compute win B4 measured |
| Type/geometry truth | `ggml/include/ggml.h:389-432` (type ids), `ggml/src/ggml-common.h:242-365` (block structs) | block_bytes/block_elems per type; already mirrored by our reader traits |
| Row size contract | `ggml/src/ggml.c` `ggml_row_size`/`ggml_nbytes` | rows are whole blocks; K % block_elems == 0 for GEMM weights — validation rule for keep-quant eligibility |
| Repack-at-load hook | `ggml/src/ggml-cpu/repack.cpp:4727` (`init_tensor` sets `tensor->extra`), `:4745-4830` (buffer type) | when the CIQ GEMM's repack tier lands, repack happens at load on the resident blocks — this loader owns that call point |
| CPU GEMM orientation | `ggml/src/ggml-cpu/ggml-cpu.c:1245-1443` (src0 = weight, rows = output features) | GGUF disk order [out, in] row-major IS ggml's src0 layout and IS our `MatmulBT` [N,K] `nk=true` orientation — keep-quant needs NO transpose (block rows cannot be transposed without requantizing) |

## Our baseline

- Expansion point (to be replaced):
  [qwen3_5_gguf_weights.cpp:105-155](../../src/vllm/model_executor/models/qwen3_5_gguf_weights.cpp#L105)
  — `DqBf16`/`OwnBf16` dequant whole tensors to bf16; `OwnBf16T` (`:134-144`)
  additionally TRANSPOSES to [K,N] for the `nk=false` `vt::Matmul` path (a
  transpose the quant path must not and cannot do).
- Reader (stays): [gguf_reader.cpp:191](../../src/vllm/model_executor/model_loader/gguf_reader.cpp#L191)
  — type recognition + `GgmlTraits` block geometry; tensor data pointers into
  the mapped file.
- Dequant kernels (stay, as oracle):
  [gguf_dequant.cpp:246-279](../../src/vllm/model_executor/model_loader/gguf_dequant.cpp#L246).
- Weight structs precedent:
  [qwen3_5_weights.h:37-80](../../include/vllm/model_executor/models/qwen3_5_weights.h#L37)
  — `OwnedTensor.nk` orientation flag; `Nvfp4Weight` already proves the
  "keep packed codes + route to a packed GEMM" pattern in this codebase.
- Engine routing: [qwen3_5.cpp:410-436,555-585](../../src/vllm/model_executor/models/qwen3_5.cpp#L410)
  — matmul helpers route on `w.nk`; a block-dtype `OwnedTensor` with
  `nk=true` flows through `MatmulBT` call sites without touching the forward.
- **Unmerged dependency:** branch `bench/quant-gguf-compute-b4-cpu-floor`
  (commit `7c91a42`, cited-not-merged by the B4 ledger row) adds the
  dense-arch (`qwen35`) GGUF loader path + F16/BF16 row dequant that the 2B
  bench file needs; gguf ctest 4/4 green on the branch.
- Honest gaps: no keep-quant storage; no `VT_CPU_REF` switch (the name exists
  only in docs); the 2B bench model cannot load from main.

## Port map

| Upstream / source | Local | Notes / deviations |
|---|---|---|
| llama.cpp keep-file-type residency | `GgufQuantWeight` in [qwen3_5_weights.h](../../include/vllm/model_executor/models/qwen3_5_weights.h): `{OwnedTensor blocks /*u8 raw*/, block dtype, n, k}`, or `OwnedTensor` with a block `vt::DType` + `nk=true` once CIQ G1 lands (preferred: one struct, no parallel type) | bytes COPIED from the mmap into 64B-aligned owned buffers (matches every other weight; mmap-backed zero-copy is a recorded follow-up, llama.cpp supports both) |
| per-tensor policy | extend the per-tensor mapping in [qwen3_5_gguf_weights.cpp](../../src/vllm/model_executor/models/qwen3_5_gguf_weights.cpp): matmul weights (attn q/k/v/o, GDN in/out, MoE expert/shared/router-adjacent GEMMs, dense ffn, lm_head, embed-as-GEMM) keep-quant when eligible; norms (`OwnNormMinus1`), embeddings-as-lookup, conv weights keep dequant | mirrors llama.cpp: it also computes norms in f32; the (w−1) RMSNorm rewrite REQUIRES dequant for norm weights — unchanged |
| `VT_CPU_REF` oracle switch | env read in the loader: force-dequant everything (today's exact path, same goldens) | the oracle the parent row promised: compute-in-quant is gated against llama.cpp, while VT_CPU_REF preserves our bit-stable reference |
| bench branch `7c91a42` | merge into main as work row L1 | decision recorded below |
| repack call point | loader calls the CIQ repack tier per tensor after residency (when that tier lands) | keeps repack transient bounded to one tensor |

## Tests to port

| Upstream test | Local tier / file | Notes |
|---|---|---|
| llama.cpp `tests/test-gguf.cpp` (parser/metadata/tensor-span cases) | T-unit: extend [tests/vllm/test_gguf.cpp](../../tests/vllm/test_gguf.cpp) | cases asserting block-quant tensor byte spans/geometry survive loading (offsets, sizes, alignment) |
| llama.cpp `tests/gguf-model-data.cpp` harness idea | T-unit: extend [tests/vllm/test_gguf_dequant.cpp:25](../../tests/vllm/test_gguf_dequant.cpp#L25) | round-trip: resident blocks → `DequantGgufRowToF32` == direct file dequant, byte-identical (proves residency is lossless) |
| (ours) loader routing | T-unit: extend [tests/vllm/test_gguf_qwen36_loader.cpp](../../tests/vllm/test_gguf_qwen36_loader.cpp) + [test_model_loader_gguf.cpp](../../tests/vllm/test_model_loader_gguf.cpp) | policy table: which tensors kept quant vs dequanted, per file type mix; `VT_CPU_REF=1` ⇒ all-dequant |
| (ours) oracle equivalence | T-parity [tests/parity/test_qwen36_gguf_engine.cpp:143](../../tests/parity/test_qwen36_gguf_engine.cpp#L143) | with `VT_CPU_REF=1`: existing greedy goldens byte-identical (loader refactor must not perturb the oracle path) |
| (ours) e2e keep-quant | same gate file, new cases | greedy vs llama.cpp oracle with keep-quant active — checked in SKIPPED until `QUANT-GGUF-CIQ-GEMM` G4 lands (tracked reason: no quant GEMM yet), per test-porting.md rule 6 |

## Gates

1. **Losslessness:** resident-block dequant == direct-from-file dequant,
   byte-identical, all six types (unit).
2. **Oracle stability:** `VT_CPU_REF=1` end-to-end run reproduces today's
   goldens byte-identically (28/28 assertions, 16/16 greedy on APEX files).
3. **Memory:** loading the 2B Q8_K_XL with keep-quant active: peak RSS during
   load+first-forward ≤ file size (2.68 GiB) + model activations + 15%
   (measured with `/usr/bin/time -v`, same recipe as B4); the 7.43 GiB
   expansion is gone. (Full-run RSS parity vs llama.cpp is the CIQ leaf's
   gate 5.)
4. **No behavior change while inert:** with the CIQ GEMM absent or
   `VT_CPU_REF=1`, ctest gguf suite green and unchanged (4/4 as on the bench
   branch, plus new units).
5. **L1 (branch merge):** gguf ctest 4/4 green on main after merging
   `7c91a42`; the B4 recipe becomes reproducible from main (loader arm).

## Dependencies

- `QUANT-GGUF-CIQ-GEMM` G1 for the block `vt::DType` (storage typing) — L2
  can co-land with it; L1 and the `VT_CPU_REF` switch (L3) have no deps.
- Consumed by CIQ gates 3-5 (E2E/perf/memory) — this leaf must land before
  those can run.
- Hardware: none special (CPU loader work; memory gate on the 20-core x86
  box). Models: 2B Q8_K_XL + APEX 35B files (checkpoint-gated skips apply).
- No GPU lock (CPU-only); idle-box rule applies to the RSS measurement.

## Work breakdown

| W | Row (claim-sized) | Content | Depends |
|---|---|---|---|
| L1 | merge bench branch | merge `7c91a42` (`bench/quant-gguf-compute-b4-cpu-floor`) into main: dense-arch `qwen35` GGUF loader + F16/BF16 row dequant; gguf ctest green; ledger note that B4 loader arm is now on main | - |
| L2 | quant residency | `GgufQuantWeight`/block-`OwnedTensor` storage + loader keep-quant path + losslessness units (gate 1) | CIQ G1 |
| L3 | routing + oracle switch | per-tensor policy table, `VT_CPU_REF` env, oracle-stability gate 2, routing units | L2 |
| L4 | memory gate + closure | RSS measurement (gate 3), matrix `M`-cell notes, ledger row | L2, L3 |

## Risks/decisions

- **DECISION — bench-branch fate:** merge `7c91a42` into main (L1). It is
  loader-only, tested, and the B4 evidence row cites it; leaving
  measurement-enabling code unmerged makes the recorded floor
  non-reproducible from main. The branch is deleted after merge. (This
  resolves the "cited-not-merged" flag in the B4 ledger row.)
- **Copy vs mmap residency:** start with copied owned buffers (uniform with
  every existing weight path, no file-lifetime coupling); mmap zero-copy is
  a recorded follow-up if load-time RSS or load latency matters later —
  llama.cpp supports both, so either mirrors upstream.
- **Embedding table:** stays dequanted-bf16 initially (`kEmbedding` is a
  gather, not a GEMM; llama.cpp dequantizes rows on the fly instead). A
  quantized-gather op is a possible follow-up row if the table dominates
  RSS on small models — record with measurements, not now.
- **Norm-weight (w−1) rewrite** and V-column reorders
  ([qwen3_5_gguf_weights.cpp:84-103](../../src/vllm/model_executor/models/qwen3_5_gguf_weights.cpp#L84))
  are value transforms — such tensors CANNOT stay quantized; the policy
  table must route them to dequant regardless of type. Covered by routing
  units.
- No product calls beyond the branch-merge decision; residency semantics
  mirror llama.cpp throughout.
