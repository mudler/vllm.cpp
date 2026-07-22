# Leaf spec: GGUF compute-in-quant GEMM — ggml tensor-traits port into vt::cpu

**Row:** `QUANT-GGUF-CIQ-GEMM` (quantization-matrix.md, leaf of the
`QUANT-GGUF-COMPUTE` block) · **status:** `ACTIVE` — **G1 landed** (2026-07-22, `CLAIM-QUANT-GGUF-COMPUTE-1`); G2-G8 open, so no encoding computes in quant yet ·
**upstream pin:** llama.cpp local fork `237ad9b96` (b9892) ·
**parent evidence:** B4 decision row
([parity-ledger.md L290](../parity-ledger.md#L290)) — the "vendor IF proven
faster" criterion is MET; this leaf is the structural fix: keep GGUF blocks
resident and quantize the ACTIVATION to int8 to meet them (llama.cpp's
mul_mat), instead of expanding weights to bf16 at load. Research estimate on
top of the threadpool: ~3.3× decode, ~10× prefill, ~3.3× weight RAM
([expansion map wave 3](expansion-map-2026-07-10.md)).

## Scope

- **In scope:** quantized-block weight storage in `vt::` (new block dtypes +
  traits); activation quantization `f32→Q8_0/Q8_K` (`from_float`); the
  per-type `vec_dot` GEMM core dispatched exactly like ggml's
  `type_traits_cpu` table; portable generic C++ kernels FIRST for the types
  our GGUF families need — **Q8_0, Q4_K, Q5_K, Q6_K, Q3_K** (APEX 35B
  Compact/Balanced + the Q8_K_XL bench file) plus already-materialized
  **Q4_0**; then per-arch SIMD tiers (x86 AVX2, Arm NEON/dotprod/i8mm for
  GB10) and the repack/interleave tier (`q4_Kx8`-style) as separate work
  rows. CPU backend only.
- **Out of scope:** loader/persistence of the quant weights (leaf
  `QUANT-GGUF-KEEPQ-LOADER`); the threadpool itself (leaf
  `QUANT-GGUF-CPU-THREADPOOL`); i-quants/TQ/MXFP4/NVFP4 GGUF types (their
  own encoding rows: `QUANT-GGUF-IQ2_S`, `QUANT-GGUF-IQ4_XS`, …); CUDA GGUF
  compute (future backend row); llamafile/KleidiAI/AMX accelerators (see
  Risks/decisions).
- **Dispatch behavior (mirrors ggml exactly):** for weight type T,
  `vec_dot_type(T)` selects the activation quant (Q8_0 for Q4_0/Q8_0; Q8_K
  for Q3_K/Q4_K/Q5_K/Q6_K per `type_traits_cpu`); src1 (activations) is
  quantized once into scratch, then each output element is one
  `vec_dot(K, w_row, act_row)`. Repack tier, when present for
  (type, arch, N%8), replaces this with interleaved `gemv/gemm` — same
  selection precedence as `ggml_repack_get_optimal_repack_type`.

## Upstream chain

Pinned local fork `/home/mudler/_git/llama.cpp` @ `237ad9b96`.

| Component | Upstream anchor | Content |
|---|---|---|
| Trait table | `ggml/src/ggml-cpu/ggml-cpu.c:211-406` | `type_traits_cpu[]`: `from_float`, `vec_dot`, `vec_dot_type`, `nrows` per type; Q8_0→(q8_0,q8_0,nrows 2 on i8mm) `:262-271`, Q3_K `:295-300`, Q4_K `:301-310`, Q5_K `:311-316`, Q6_K `:317-326`, Q4_0 `:230-239` |
| GEMM driver | `ggml-cpu.c:1245-1443` | src1→`vec_dot_type` quant into `wdata` (`:1313-1349`, block-split across threads), chunk grid + atomic stealing (threadpool leaf), `num_rows_per_vec_dot` 2-row mmla guard `:1426-1433` |
| Chunk worker | `ggml-cpu.c:1155-1243` | 16×16 tile, `vec_dot` per output, dst column memcpy |
| Block layouts | `ggml/src/ggml-common.h:242-245` (q8_0), `:305-310` (q3_K), `:317-327` (q4_K), `:334-345` (q5_K), `:352-357` (q6_K), `:361-365` (q8_K: f32 d, i8 qs[256], i16 bsums[16]) | byte-exact structs; our gguf_dequant.cpp already mirrors the weight-side layouts |
| Activation quant reference | `ggml/src/ggml-quants.c:238` (`quantize_row_q8_0_ref`), `:2696` (`quantize_row_q8_K_ref` incl. bsums) | portable reference; CPU wrappers `ggml-cpu/quants.c:45,117` |
| Generic vec_dot kernels (portable C++ tier) | `ggml-cpu/quants.c:174` (q4_0·q8_0), `:400` (q8_0·q8_0), `:566` (q3_K·q8_K), `:645` (q4_K·q8_K), `:720` (q5_K·q8_K), `:800` (q6_K·q8_K) | integer block dots + bsums tricks; this is the byte-for-byte port source for tier 0 |
| x86 SIMD tier | `ggml-cpu/arch/x86/quants.c:1170` (q8_0·q8_0), `:1900` (q4_K·q8_K), `:2288` (q6_K·q8_K), … | AVX2/AVX512 variants, generic fallback tail |
| Arm SIMD tier | `ggml-cpu/arch/arm/quants.c:1076` (q8_0·q8_0), `:2260` (q4_K·q8_K), `:2890` (q6_K·q8_K), … | NEON/dotprod/SVE/i8mm (`nrows==2` mmla paths) — GB10's tier |
| SIMD abstraction | `ggml-cpu/simd-mappings.h`, `arch-fallback.h` | per-ISA macro mapping; our tiers keep the same generic-fallback naming discipline |
| Tensor-traits hook | `ggml-cpu/traits.h:20-36`, `traits.cpp:9-38` | `tensor_traits::{work_size,compute_forward}` + `extra_buffer_type` probe — the seam repack plugs into |
| Repack tier | `ggml-cpu/repack.cpp:4153-4250` (traits class), `:4253` (`forward_mul_mat`), `:4528-4726` (`ggml_repack_get_optimal_repack_type`: q4_K_8x8 on AVX2/i8mm, q6_K_8x8/8x4, q8_0_4x4/4x8, q5_K, q2_K …), `:4727` (extra hook), `:4745-4830` (buffer type); activation `ggml_quantize_mat_t<4/8, Q8_0>` `:318-357`; generic gemv/gemm: q4_K `:887,:1822,:1905`, q5_K impl `:551,:647`, q6_K impl `:357,:447`, q8_0 `:1274-1368`; arch SIMD: `arch/x86/repack.cpp`, `arch/arm/repack.cpp` | prefill engine: 4/8 interleaved weight rows × 4/8 activation rows per kernel call |

Runtime-trace: CPU dispatch is static given (type, arch, N alignment); the B4
oracle run (20-thread llama.cpp, repack active on AVX2) is the traced
denominator. When gating on GB10 Arm, confirm which tier llama.cpp selects
there (i8mm repack expected) before comparing.

## Our baseline

- **No quantized compute exists.** All executable GGUF weights expand to bf16
  at load: [qwen3_5_gguf_weights.cpp:105-155](../../src/vllm/model_executor/models/qwen3_5_gguf_weights.cpp#L105);
  GEMMs then run the naive bf16 loops
  [cpu_ops.cpp:30-57](../../src/vt/cpu/cpu_ops.cpp#L30).
- Dequant kernels (weight-side block decode, byte-for-byte ports already
  reviewed against the same pin):
  [gguf_dequant.cpp:43-219](../../src/vt/cpu/cpu_quant_dequant.cpp#L48),
  dispatch `:246-279` — reuse their layout knowledge; they REMAIN as the
  `VT_CPU_REF` oracle path.
- `vt::` typing: [include/vt/dtype.h:21](../../include/vt/dtype.h#L21) has no
  block dtypes (`kF32,kF16,kBF16,kI8,kI32,kI64`); `vt::Tensor` assumes
  elementwise strides. [include/vt/ops.h:62](../../include/vt/ops.h#L62)
  `kMatmulBT` is the Linear-orientation GEMM
  ([N,K] row-major weight) — the exact orientation ggml's src0 has.
- Precedent for quant-weight structs + orientation flag:
  [qwen3_5_weights.h:37-80](../../include/vllm/model_executor/models/qwen3_5_weights.h#L37)
  (`OwnedTensor.nk`, `Nvfp4Weight` packed+scales kept raw in [N,K]).
- Tests today: [tests/vllm/test_gguf_dequant.cpp:25-223](../../tests/vllm/test_gguf_dequant.cpp#L25)
  (dequant units), [tests/parity/test_qwen36_gguf_engine.cpp:143](../../tests/parity/test_qwen36_gguf_engine.cpp#L143)
  (same-file greedy vs llama.cpp oracle goldens) — nothing exercises a quant
  GEMM. Honest gap: the whole `C` column of the GGUF encoding table.

## Port map

| Upstream | Local | Notes / deviations |
|---|---|---|
| `ggml.h` type ids + `ggml-common.h` block structs | extend `vt::DType` with `kQ4_0,kQ8_0,kQ3_K,kQ4_K,kQ5_K,kQ6_K,kQ8_K` + block traits (block_elems, block_bytes) in `src/vt/dtype.cpp`; cross-checked against `GgmlTraits` ([gguf_reader.cpp:191](../../src/vllm/model_executor/model_loader/gguf_reader.cpp#L191)) | design decision recorded below; `kQ8_K` is activation-only (never a weight/storage dtype in files) |
| `type_traits_cpu[]` (`ggml-cpu.c:213-471`) | new `src/vt/cpu/cpu_quant_traits.{h,cpp}`: `{from_float, vec_dot, vec_dot_type, nrows}` per block dtype | same table shape so upstream diffs port mechanically |
| `quantize_row_q8_0_ref` / `quantize_row_q8_K_ref` (`ggml-quants.c:238,2696`) | `src/vt/cpu/cpu_quant_act.cpp` | byte-for-byte incl. Q8_K bsums; f32 input (our activations are bf16 → widen first, matching ggml's f32 src1 contract) |
| generic `ggml_vec_dot_*` (`ggml-cpu/quants.c:174,400,566,645,720,800`) | `src/vt/cpu/cpu_quant_dot.cpp` (tier 0, pure portable C++) | keep `*_generic` names in comments + upstream line cites per kernel |
| `ggml_compute_forward_mul_mat` src1-quant + chunk loop (`ggml-cpu.c:1245-1443,1155-1243`) | new op `OpId::kMatmulBTQuant` in `include/vt/ops.h` + CPU kernel in `cpu_ops.cpp`; `vt::MatmulBT` call sites route on `b.dtype` being a block dtype (no signature change) | scratch for quantized activations: per-call 64B-aligned buffer sized `rows × row_size(vec_dot_type, K)` mirroring `ggml_graph_plan` wdata (`ggml-cpu.c:2752-2980`); reuses `ParallelFor` from the threadpool leaf |
| `arch/x86/quants.c`, `arch/arm/quants.c` | `src/vt/cpu/arch/{x86,arm}/cpu_quant_dot_{avx2,neon}.cpp` (tier 1) | compile-time ISA gates + runtime CPU-feature probe mirroring `ggml_cpu_has_*`; generic fallback always built (CI runs it) |
| `repack.cpp` traits + gemv/gemm + `quantize_mat_t` | `src/vt/cpu/cpu_quant_repack.cpp` (tier 2) | weight repack at LOAD (from the keep-quant loader's resident blocks) selected per (dtype, arch, N%8) exactly as `:4528-4726`; brings the 4×8/8×8 GEMM used for prefill |
| `traits.h` `tensor_traits`/`extra_buffer_type` classes | NOT ported as a polymorphic hook | our dispatch is the static op registry; ggml needs the hook for pluggable backends we don't have — recorded deviation, revisit if a second accelerator (AMX/KleidiAI-class) is ported |

## Tests to port

| Upstream test | Local tier / file | Notes |
|---|---|---|
| llama.cpp `tests/test-quantize-fns.cpp` (thresholds `:17-28`: total quant error ≤0.002 4-bit+, ≤0.0040 3-bit, reference error ≤0.0001, dot-product error ≤0.02, ≤0.04 low-bit) | T-unit `tests/vt/test_ops_quant_dot.cpp` (new) | per-type: quantize→dequant RMSE bounds AND `vec_dot` vs f64 reference dot on synthetic rows; same thresholds, cite each |
| llama.cpp `tests/test-backend-ops.cpp` MUL_MAT cases (`:4260-4450`, NMSE ≤ `max_nmse_err()` `:1133-1151` ≈5e-4) | T-unit, same file | `kMatmulBTQuant` vs the dequant-to-f32 composite (`DequantGgufRowToF32` + `MatmulBTKernel`) at model shapes (K=hidden, N=proj dims, M∈{1,4,32,512}) per type |
| llama.cpp `tests/test-quantize-perf.cpp` | tooling only (bench harness), not a ctest gate | microbench per vec_dot tier; evidence for the ledger, never a pass/fail gate |
| our existing goldens | T-parity [tests/parity/test_op_parity.cpp:34](../../tests/parity/test_op_parity.cpp#L34) | add `matmul_bt_quant` runner + goldens; **goldens+runner land in the SAME commit or the op goes in `PendingRunnerOps()`** (workflow.md anti-stale-golden gate) |
| same-file engine gates | T-e2e [tests/parity/test_qwen36_gguf_engine.cpp:143](../../tests/parity/test_qwen36_gguf_engine.cpp#L143) + the 2B Q8_K_XL file | greedy 16/16 vs same-file llama.cpp oracle with compute-in-quant ACTIVE; new goldens (see Risks — tokens may legitimately differ from the bf16-expansion path; the oracle is llama.cpp, not our old path) |

## Gates

1. **Op correctness:** ported test-quantize-fns bounds green for all six
   types (generic tier on CI; SIMD tiers on their boxes); MUL_MAT NMSE
   ≤ 5e-4 vs the dequant-f32 composite at all listed shapes.
2. **Oracle discipline:** `VT_CPU_REF=1` env forces the dequant-to-bf16 +
   bf16-GEMM path (loader leaf provides it): existing goldens stay
   byte-identical there — the parity oracle is never lost. Compute-in-quant
   is NOT bit-identical to it by design; its correctness bar is the
   llama.cpp same-file oracle.
3. **E2E token gate:** greedy 16/16 token-exact vs same-file llama.cpp CPU
   (`237ad9b96` build) on (a) Qwen3.5-2B-UD-Q8_K_XL, (b) APEX 35B Compact,
   (c) APEX 35B Balanced — the existing gate harness recipe.
4. **Perf (B4-anchored, same box/file/recipe as
   [parity-ledger.md L290](../parity-ledger.md#L290), threadpool active, 20
   threads, ≥2 reproducing runs):**
   - tier 0 (generic vec_dot) milestone: decode ≥ 0.25× llama.cpp tg32
     (≥3.2 t/s), prefill ≥ 0.1× pp128;
   - tier 1 (arch SIMD) milestone: decode ≥ 0.7× tg32;
   - tier 2 (repack) closes the row: **match or beat** llama.cpp on decode
     t/s, prefill t/s AND peak RSS on the same file/box (mirror-as-floor;
     `BACKEND-GATE-CPU-LLAMACPP` consumes this evidence; GB10 Arm rerun
     recorded there too).
5. **Memory:** peak RSS with compute-in-quant ≤ 1.15× llama.cpp same-run
   (2.80 GiB on the 2B) — the bf16 expansion (7.43 GiB) must be gone
   (loader leaf) and no hidden f32 staging may reintroduce it.

## Dependencies

- `QUANT-GGUF-CPU-THREADPOOL` (`ParallelFor` + chunk loop; perf gates
  meaningless without it). Op-level tier-0 kernels + unit tests may land
  before it (single-thread), but the row cannot pass gate 4 without it.
- `QUANT-GGUF-KEEPQ-LOADER` for gates 3-5 (resident quant weights + the
  `VT_CPU_REF` switch). Unit gates 1-2 need only synthetic blocks.
- Hardware: 20-core x86 box (AVX2 tier + gates); GB10 (Arm NEON/i8mm tier);
  CI runs generic tier. No GPU lock; CPU runs need an otherwise-idle box per
  benchmark-protocol.
- Oracle: llama.cpp fork build at `237ad9b96` (same as B4; keep the local
  NVFP4/Q1_0 fork patch noted — irrelevant to these six types).
- Models: 2B Q8_K_XL + APEX 35B Compact/Balanced GGUFs (already on the boxes;
  checkpoint-gated tests skip when absent, as today).

## Work breakdown

| W | Row (claim-sized) | Content | Depends | Status |
|---|---|---|---|---|
| G1 | block dtypes + traits | `vt::DType` block entries, traits table, `kMatmulBTQuant` op skeleton + generic-composite fallback, trait cross-check vs `GgmlTraits` | - | **DONE** 2026-07-22 |
| G2 | activation quant | `quantize_row_q8_0/q8_K` ports + scratch sizing; unit tests | G1 | open |
| G3 | tier-0 vec_dot kernels | the six generic `vec_dot` ports + GEMM wiring + test-quantize-fns/backend-ops ports + parity goldens | G1, G2 | open |
| G4 | e2e enablement | route `MatmulBT` call sites on block dtype; engine gates (3) + `VT_CPU_REF` A/B; first B4-recipe measurement (tier-0 milestone) | G3, loader L2-L3, threadpool W2 | open |
| G5 | x86 AVX2 tier | `arch/x86` vec_dot ports + feature probe; tier-1 milestone measurement | G3 | open |
| G6 | Arm NEON/i8mm tier | `arch/arm` ports incl. `nrows==2` mmla; GB10 measurement | G3 | open |
| G7 | repack tier | repack-at-load + gemv/gemm + `quantize_mat_t`; selection parity vs `:4528`; closes gates 4-5 | G4 + loader L2 | open |
| G8 | ledger + matrix closure | full A/B series, ledger row, flip `C`/`E`/`P` cells per encoding row (Q8_0/Q4_K/Q5_K/Q6_K/Q3_K/Q4_0), roadmap update | G4-G7 | open |

G5/G6 are parallel; G7 may start once G4's layout is fixed.

## Risks/decisions

- **`vt::DType` extension vs side-struct** (internal design, decided here):
  block dtypes IN the enum mirror ggml's "type carries layout" model, keep
  `Tensor` self-describing, and let dispatch key on `b.dtype` — the
  alternative (parallel `QuantTensor`) would fork every op signature. Blocked
  dtypes are storage-only: any elementwise op on them is a `VT_CHECK` failure.
- **Goldens shift:** enabling compute-in-quant changes CPU e2e numerics (not
  bit-equal to bf16 expansion). Decision: goldens for quant-GGUF engine gates
  are regenerated against the llama.cpp oracle in the SAME change that flips
  the default, with the `VT_CPU_REF` path keeping the old goldens alive. No
  silent tolerance widening.
- **Repack memory transient:** repack-at-load temporarily holds source+dest
  blocks per tensor; do it tensor-by-tensor (llama.cpp repacks in-buffer per
  tensor) to keep gate 5.
- **Accelerator hooks not ported** (llamafile sgemm `ggml-cpu.c:1286-1310`,
  AMX, KleidiAI): each is an extra tier behind the same dispatch seam;
  llama.cpp's B4 arm did not need them to hit the floor numbers (stock CPU
  build). Revisit only if gate 4 tier-2 stalls below 1.0× — record then.
- **`nrows==2` boundary guards** (`ggml-cpu.c:1426-1433`) must be ported with
  the mmla kernels or i8mm results corrupt on odd shapes — covered by the
  non-divisor shape cases in the ported MUL_MAT tests.
- Upstream behavior is not reopened: quant math, thresholds, and tier
  selection all mirror llama.cpp; the only product-ish calls are the two
  recorded above (enum design, golden regeneration protocol).

## G1 result (2026-07-22)

**Landed.** `vt::DType` gains the seven block entries (`kQ4_0 kQ8_0 kQ3_K kQ4_K
kQ5_K kQ6_K kQ8_K`) with block geometry in
[dtype.cpp](../../src/vt/dtype.cpp#L32) and the storage-only rule enforced —
`SizeOf` on a block dtype THROWS, so any elementwise kernel that reaches one
fails loudly. `RowSizeBytes` mirrors `ggml_row_size` (rows are whole blocks).
The `type_traits_cpu` mirror is [cpu_quant_traits.cpp](../../src/vt/cpu/cpu_quant_traits.cpp#L1)
behind the new public [vt/quant.h](../../include/vt/quant.h#L1); `vec_dot_type`
and `nrows` are populated, `vec_dot`/`from_float` stay `nullptr` until G3/G2.
`OpId::kMatmulBTQuant` + [the CPU kernel](../../src/vt/cpu/cpu_quant_gemm.cpp#L1)
run the generic dequant-composite fallback for every type.

**Deviation from the port map (recorded).** The six `dequantize_row_*` decoders
were MOVED, not duplicated: they now live at
[cpu_quant_dequant.cpp](../../src/vt/cpu/cpu_quant_dequant.cpp#L1) as the traits
table's `to_float` column (upstream keeps `to_float` in `ggml.c`'s
device-neutral `type_traits`; vt has no such second table, so it rides the CPU
one), and the GGUF loader
[delegates](../../src/vllm/model_executor/model_loader/gguf_dequant.cpp#L75).
One implementation serves both the loader oracle and the GEMM fallback. The
code is byte-identical to what it replaced, so numerics are unchanged, and
`test_gguf_dequant` plus a new byte-for-byte equivalence case gate that.

**Trait cross-check — NO mismatch.** [test_ops_quant_traits.cpp](../../tests/vt/test_ops_quant_traits.cpp#L1)
compares THREE independent statements of the same llama.cpp facts: vt's block
geometry table, the GGUF reader's `GgmlTraits`, and the block-struct arithmetic
written out from `ggml-common.h`. All seven types agree on block_elems,
block_bytes and type id; the id map round-trips; `Q8_K` (id 15) is correctly
absent from the reader's table because it is activation-only. 8 cases /
5,694 assertions green.

**Gates at G1:** clean CPU `-Werror` build (0 warnings); full CPU ctest
**151/151**; gguf ctest 4/4. **DGX CONFIRMATION RUN (`dgx.casa`, `~/work/vllm.cpp-ciq-g1`, production flags `-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc -DVLLM_CPP_CUTLASS_DIR=/home/mudler/cutlass-4.5.0 -DVLLM_CPP_TRITON=ON`, one `flock $HOME/gpu.lock` for the whole series):** clean CUDA `-Werror` build to 100%, **0 warnings**. Regression set ALL UNCHANGED — 27B **235/235**, 35B **315/315**, Qwen3-Coder **6/6** (strict 5/6, max gap 0 nats), Qwen3-dense **16/16** (strict 11/16, max gap 0.25 nats), OPT **6/6** (36 assertions), DeepSeek-V2-Lite **8/8** (223 assertions). `test_qwen36_gguf_engine` run STANDALONE (it OOMs co-scheduled) on the real APEX files: **28/28 assertions, 2/2 cases, 16/16 tokens each on Compact and Balanced** — the k-quant dequant path is byte-stable across the decoder move. Gates 1-5 of this spec are
untouched: they need G2/G3 (op correctness) and the loader leaf (e2e/perf/RSS).
