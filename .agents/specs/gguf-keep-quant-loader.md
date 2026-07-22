# Leaf spec: GGUF keep-quantized loader — quantized weights resident, dequant as oracle

**Row:** `QUANT-GGUF-KEEPQ-LOADER` (quantization-matrix.md, leaf of the
`QUANT-GGUF-COMPUTE` block) · **status:** `ACTIVE` — **L1+L2+L3 landed** (2026-07-22, `CLAIM-QUANT-GGUF-COMPUTE-1`): block residency, the total per-tensor routing policy and the `VT_CPU_REF` oracle switch all exist and are gated. **Keep-quant is now DEFAULT ON wherever the running device can execute the quant GEMM** (CIQ G4 flipped it, `CLAIM-QUANT-GGUF-CIQ-G4-1`), so a CPU load keeps its blocks and a CUDA load still expands. L4 (the RSS gate) **open and NOT met**: peak RSS fell 7.428 → 6.401 GiB (1.16×), still **2.29× llama.cpp** against a ≤1.15× bar — the residual is diagnosed below ·
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
  [gguf_dequant.cpp:246-279](../../src/vllm/model_executor/model_loader/gguf_dequant.cpp#L53).
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

| W | Row (claim-sized) | Content | Depends | Status |
|---|---|---|---|---|
| L1 | merge bench branch | merge `7c91a42` (`bench/quant-gguf-compute-b4-cpu-floor`) into main: dense-arch `qwen35` GGUF loader + F16/BF16 row dequant; gguf ctest green; ledger note that B4 loader arm is now on main | - | **DONE** 2026-07-22 |
| L2 | quant residency | `GgufQuantWeight`/block-`OwnedTensor` storage + loader keep-quant path + losslessness units (gate 1) | CIQ G1 | **DONE** 2026-07-22 |
| L3 | routing + oracle switch | per-tensor policy table, `VT_CPU_REF` env, oracle-stability gate 2, routing units | L2 | **DONE** 2026-07-22 |
| L4 | memory gate + closure | RSS measurement (gate 3), matrix `M`-cell notes, ledger row | L2, L3 | **MEASURED, NOT MET** 2026-07-22 (see below) |
| L5 | mmap residency + tied-head sharing | stop COPYING block bytes out of the mapped file (llama.cpp supports both; L2 chose copy deliberately) and stop materialising the vocab matrix twice for a tied `lm_head` — the two changes L4's measurement identifies as ~1.9 GiB of the remaining 3.6 GiB gap | L4 | open |

## Risks/decisions

- **DECISION — bench-branch fate:** land `7c91a42`'s content on main (L1). It
  is loader-only, tested, and the B4 evidence row cites it; leaving
  measurement-enabling code unmerged makes the recorded floor
  non-reproducible from main.
  **LANDED 2026-07-22 as a SQUASH, not a merge — `66233e6`, single parent
  `2ff7252`.** `7c91a42` is therefore NOT an ancestor of main. Two reasons it
  could not be a true merge: it was cut at `83010c7`, **422 commits** behind,
  and did not apply cleanly (main had replaced `FromModelDir`'s hardcoded
  dense-vs-MoE GGUF split with the `ModelRegistry` seam, so the branch's
  `LoadedEngine::IsDenseArch` no longer exists — its intent was re-expressed
  through that seam); and as a standalone commit it fails the
  documentation-checkpoint gate it predates, which would redden CI on any push
  range containing it. Provenance is cited in the `66233e6` commit message
  instead.
  Consequences, recorded so they do not surprise anyone: the
  **"cited-not-merged" flag in the B4 ledger row resolves by CONTENT, not by
  ancestry**; `git branch -d bench/quant-gguf-compute-b4-cpu-floor` will
  **refuse** (not merged by ancestry) and needs `-D`, which is safe because the
  content is on main; and the local ref `quant-gguf-compute-g1-l1` → `9bc86f0`
  is retained deliberately as the only record of the true merge lineage (both
  `2ff7252` and `7c91a42` as parents) and the only thing keeping `7c91a42`
  reachable if the bench branch is deleted.
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

## L1 result (2026-07-22) — the bench-branch merge

`7c91a42` (`bench/quant-gguf-compute-b4-cpu-floor`) did **not** apply cleanly:
main has moved 422 commits since the branch point (`83010c7`), and one of those
replaced `LoadedEngine::FromModelDir`'s hardcoded dense-vs-MoE GGUF split with
the `ModelRegistry` seam. Three of the branch's four files auto-merged;
`src/vllm/entrypoints/model_loader.cpp` conflicted.

**Resolution.** Main's side was kept in full — the branch's inline
`LoadedEngine::IsDenseArch(config)` split is obsolete machinery (the symbol no
longer exists). The branch's INTENT was re-expressed through the seam that
replaced it:

- [`HfConfigFromGguf`](../../src/vllm/model_executor/models/qwen3_5_gguf_weights.cpp#L212)
  now maps the GGUF `general.architecture` key onto the registered architecture
  ID — `qwen35` (dense) to `Qwen3_5ForConditionalGeneration`, the MoE keys to
  `Qwen3_5MoeForConditionalGeneration` — instead of unconditionally claiming the
  MoE wrapper.
- [`LoadQwen3_5DenseModel`](../../src/vllm/model_executor/models/qwen3_5_dense.cpp#L60)
  gains a `ModelSource::Kind::kGguf` branch calling the branch's
  `LoadQwen3_5DenseFromGguf`, replacing its blanket "does not support GGUF
  weights" throw.

That is strictly better than the branch's version: dense GGUF dispatch now goes
through the same registry every other architecture uses, so it composes with
the arch-additivity work that landed in the interim. The branch's other three
files (the dense weight loader, the header declaration, and the F16/BF16 row
dequant cases) merged unchanged.

**Gate 5 met:** gguf ctest **4/4** green on main after the merge (`test_gguf`,
`test_gguf_dequant`, `test_gguf_qwen36_loader`, `test_model_loader_gguf`) and
full CPU ctest 151/151. On dgx, `test_qwen36_gguf_engine` run STANDALONE (it
OOMs co-scheduled) against the real APEX files passes **28/28 assertions, 2/2
cases, 16/16 tokens each on Compact and Balanced**, and the full model
regression set is unchanged (27B 235/235, 35B 315/315, Qwen3-Coder 6/6,
Qwen3-dense 16/16, OPT 6/6, DeepSeek-V2 8/8) on a clean `-Werror` CUDA build.
The B4 loader arm is reproducible from main; the branch may be deleted. Gates
1-4 remain owed by L2/L3.

## L2 + L3 result (2026-07-22)

**Landed together.** GGUF weights can now stay in their native ggml blocks from
file to weight struct, every tensor the loader touches is routed by an explicit
total policy, and the dequant path is reachable as an oracle. What has NOT
changed is what a production load does: keep-quant is default OFF.

**L2 — residency.** The spec's preferred shape was taken: ONE struct, no
parallel type. [`OwnGgufQuantBlocks`](../../src/vllm/model_executor/models/qwen3_5_gguf_weights.cpp#L20)
copies a tensor's raw blocks into an `OwnedTensor` whose `dtype` is the block
`vt::DType` (CIQ G1) and whose orientation is the file's own `[N=out, K=in]`
with `nk = true` — GGUF disk order IS ggml's src0 layout IS our `MatmulBT`
orientation, so residency needs no transpose and performs none (a block
encoding cannot be transposed without requantizing). It refuses, loudly, a
non-keep-quant encoding, a `K` that is not a whole number of blocks
(`ggml_row_size`'s precondition), and any slice outside the reader's validated
tensor span. Stacked `[E, out, in]` expert tensors split by BYTE RANGE: an
expert is a whole number of rows, hence of blocks, so no block is ever cut.
Bytes are COPIED, uniform with every other weight path (mmap zero-copy stays a
recorded follow-up; llama.cpp supports both).

**L3 — routing + oracle.** [`gguf_keep_quant.{h,cpp}`](../../src/vllm/model_executor/model_loader/gguf_keep_quant.cpp#L1)
owns the decision as a pure function of `(role, encoding, shape)` plus two
switches. Six roles partition every tensor in the file:
`kMatmulWeight` and `kStackedExpertWeight` (taken verbatim — the only two that
can keep blocks), `kTransformedWeight` (the `(w-1)` RMSNorm rewrite,
`ssm_a = log(-x)`, and the V-head reorders), `kEmbeddingTable` (a gather, not a
GEMM), `kConvWeight` and `kVector`. Keeping blocks additionally requires the
right rank, one of the six executable encodings, an executable `vec_dot`, and
`K % BlockElems == 0`. `VT_CPU_REF=1` forces expansion regardless — the parity
oracle the parent row promised. `VT_GGUF_KEEP_QUANT=1` opts INTO residency;
**its default is OFF and CIQ G4 owns flipping it**, because until the model's
`MatmulBT` call sites route on block dtype nothing can consume a block-typed
weight.

**Totality is proven three ways, not asserted.** (1) The role switch carries no
`default:` label, so a new role that forgets to state its residency is a
`-Werror=switch` BUILD failure. (2) Every tensor the loader touches goes through
`GgufLoadPolicy::Route` — the verbatim GEMM weights via `OwnMatmulWeight`, and
everything else via a `RequireExpand` assertion that fails loudly if the policy
ever tried to keep a value-rewritten tensor — and the policy's `audit` hook
records what it saw, so the test asserts `routed == the file's complete tensor
list` on both a dense and a MoE fixture. (3) A unit walks 6 roles × 12 ggml
encodings × 6 shapes and compares against an expectation written out LONGHAND,
never derived from the implementation, with both outcomes shown to occur
(12 keep / 420 expand) so the table cannot pass vacuously.

**Gate 1 (losslessness) — PROVEN PER ENCODING.** `Q4_0, Q8_0, Q3_K, Q4_K, Q5_K,
Q6_K` each get their OWN test case (a failure names the encoding): resident
bytes `memcmp`-equal to the file span, and dequantizing the resident blocks
BYTE-IDENTICAL to the direct-from-file expansion in both f32 and bf16. Block
bytes are pseudo-random rather than encoder-produced, because the property must
hold for every bit pattern a file can contain; all six block structs place
their f16 fields at even offsets and all six block sizes are even, so clearing
bit 6 of every odd-indexed byte bounds every scale finite (asserted) without
otherwise restricting the data. At LOADER level the same gate runs end to end:
dequantizing a kept weight and applying the loader's own transpose reproduces
the expanded `[K,N]` bf16 tensor byte for byte, per weight, on both a dense and
a MoE fixture (including per-expert, which is what catches a wrong stacked
slice offset).

**Gate 2 (oracle stability) — MET.** With `VT_CPU_REF=1` and keep-quant
REQUESTED, nothing stays quantized and every weight is bit-identical to the
historical load; on dgx `test_qwen36_gguf_engine` under `VT_CPU_REF=1` is
**28/28 assertions, 2/2 cases, 16/16 tokens** on both APEX files — the same
result as without it.

**Mutation-tested (10 mutants, 10 caught).** Residency byte corruption; the
embedding table skipping the policy; the oracle switch ignored; the master
switch ignored; the stacked-expert K taken from the wrong axis; the ragged-K
guard dropped; value-transformed tensors made keep-quant eligible; the stacked
slice always reading expert 0; a dropped `nk` flag; a swapped shape order — all
caught. The expert-offset mutant SURVIVED the first battery and exposed a real
coverage hole (the dense fixture has no experts); the MoE fixture and its
per-expert byte-range assertions were added in response, and it is caught now.

**Nothing moved.** Keep-quant is default OFF, and a `nullptr` policy reads the
environment, so a production GGUF load is byte-for-byte what it was — asserted
directly by a case comparing the env-driven load against an explicit
all-expand policy. On dgx the full regression set is UNCHANGED, each gate run
STANDALONE: 27B **235/235**, 35B **315/315**, Qwen3-Coder **6/6** (138
assertions), Qwen3-dense **16/16** on both 0.6B and 4B (664), OPT **6/6** (36),
DeepSeek-V2-Lite **8/8** (223), and `test_qwen36_gguf_engine` **28/28** with
16/16 tokens on Compact and Balanced. Clean CUDA `-Werror` build to 100%,
**0 warnings**, production flags. Golden corpus md5 (475 files) identical
before and after: `2965ef5772b556d3f3f86fedf4221b2f`. Dev box: clean CPU
`-Werror` full rebuild, 0 warnings, full CPU ctest **154/154** (153 + the new
suite). The new units are green on dgx's **aarch64** with identical counts
(17 cases / 5,574 assertions), and `test_gguf_dequant` (11/202),
`test_ops_quant_traits` (8/5,615) and `test_ops_quant_dot` (16/78,052) are
unchanged.

**Gates 3-5 remain owed.** Gate 3 (peak RSS ≤ file size + activations + 15%) is
L4's and cannot be measured until keep-quant is ON, which needs CIQ G4; gate 4
(no behavior change while inert) is met as stated above. Weights STAYING
quantized is not the same as COMPUTING in quant: the per-encoding `C` cells stay
`-` until G4.

## L4 result (2026-07-22) — the default is ON, gate 3 is MEASURED and NOT MET

CIQ **G4** routed `vt::MatmulBT` onto `kMatmulBTQuant` and flipped this leaf's
master switch: [`GgufLoadPolicy::FromEnv`](../../src/vllm/model_executor/model_loader/gguf_keep_quant.cpp#L95)
now defaults `keep_quant` to `GgufQuantComputeAvailable()` — "is
`kMatmulBTQuant` registered for the device this process will run on" — instead
of `false`. `VT_GGUF_KEEP_QUANT=0` remains the opt-out and `VT_CPU_REF=1` still
overrides everything. G4 also added `expand_nk`, which stops TRANSPOSING a
matmul weight that has to expand (see the CIQ leaf); it rides the same
availability condition, is off under `VT_CPU_REF`, and is bit-exact on the CPU
GEMM by construction.

**Gate 3 measured (binding recipe, idle `dgx.casa` aarch64, one `flock`, 3 reps,
`Qwen3.5-2B-UD-Q8_K_XL.gguf`):** peak RSS **7,788,196 KB → 6,712,368 KB**, i.e.
**7.428 → 6.401 GiB**, a 1.16× reduction. The bar (file size + activations +
15 %, ≈ 3.1 GiB) is **NOT met**, and the competitor ratio is **2.29× llama.cpp**
(2.798 GiB) against the CIQ leaf's gate-5 bar of 1.15×.

**The residual is fully accounted for, and only ~25 % of it is the bf16
expansion this leaf was about.** On this file:

| term | size | why it is still resident |
|---|---|---|
| touched file pages | 2.68 GiB | L2 COPIES blocks into owned buffers; the mapped source pages stay resident for the copy |
| f16 → bf16 expansion | 1.615 GiB | 56 of the file's tensors are `f16`, which is not a block encoding, so keep-quant cannot apply |
| second copy of the tied head | 0.947 GiB | `token_embd.weight` is materialised once as the embedding table and again as `lm_head` |
| kept `q8_0` blocks | 1.062 GiB | the intended residency — this is the part that WORKED |
| **total** | **≈ 6.35 GiB** | vs 6.401 GiB measured |

So the two loader changes worth doing next (work row **L5**) are
**mmap-in-place residency** (removes the 2.68 GiB double-count; recorded as a
follow-up in Risks/decisions from the start) and **sharing the tied `lm_head`
with the embedding table** (removes 0.947 GiB). Together ≈ 1.9 GiB of the
3.6 GiB that separates us from llama.cpp. The remaining 1.6 GiB is the f16
expansion, which disappears if the elementwise GEMM learns to consume f16 rows
directly — that is the CPU track's new #1 lever, not this leaf's.

**Nothing else moved.** The oracle arm (`VT_CPU_REF=1`) reproduces the pre-G4
RSS to 256 KB (7,787,940 vs 7,788,196) and the pre-G4 latency to 0.15 %, and its
output token stream is byte-identical to both the BEFORE and the AFTER arm (one
md5, `d235db12f2cd304007530286a1755c95`) — gate 2 holds end to end with the
default flipped.
