# Qwen3.5/3.6 GDN merged BF16 input projections

**Row:** `KERNEL-GEMM-BF16` · **consumer:** `SERVE-GATE-ONLINE` ·
**status:** spike complete, `READY` · **priority:** roadmap order 0.

The finalized exact-c2 trace identifies one complete model-topology mismatch:
the 27B path launches four BF16 input projections in each of 48 GDN layers,
while vLLM v0.25.0 launches two merged projections. This spec ports that
missing `MergedColumnParallelLinear` behavior as two attributable checkpoints:
`in_proj_ba` first, then `in_proj_qkvz`. Trace durations remain diagnostic;
only same-binary and fresh-oracle gates can earn performance credit.

## Scope

### In scope

- Qwen3.6-27B's unquantized BF16 GDN input projections at TP=1:
  `in_proj_b + in_proj_a -> in_proj_ba [96,5120]`, then
  `in_proj_qkv + in_proj_z -> in_proj_qkvz [16384,5120]`, in raw torch
  Linear `[N,K]` orientation for `vt::MatmulBT`.
- Exact logical row order `[q,k,v,z]` and `[b,a]`, matching the vLLM mapper.
- One resident owner per merged weight. The rollback path uses non-owning row
  views of that owner; it never retains packed and split copies together.
- BF16 merged outputs and inner-contiguous row-strided logical views:
  `mixed_qkv [T,10240]` and `z [T,6144]` with stride 16384; `b/a [T,48]`
  with stride 96.
- Stride-aware CPU/CUDA consumers where required: causal convolution for
  `mixed_qkv`, gated RMSNorm for `z`, and GDN post-conv / g-beta preparation
  for BF16 `b/a`. Loads upcast to F32 in registers as upstream does.
- Independent process-cached toggles `VT_GDN_MERGED_BA` and
  `VT_GDN_MERGED_QKVZ`, plus master rollback `VT_GDN_MERGED_PROJ=0`.
- Dense and paged forwards, CUDA-graph capture/replay, loader/storage tests,
  token correctness, memory accounting, exact node traces, and performance.

The real 27B topology is H=5120, 64 layers with `[GDN,GDN,GDN,full] x16`,
Hk=16, Hv=48 and Dk=Dv=128. Therefore 48 GDN layers use key_dim=2048,
value_dim=6144, conv_dim=10240, qkvz width=16384 and ba width=96.

### Dispatch and mode matrix

| Mode | qkv/z | b/a | Required behavior |
|---|---|---|---|
| 27B CUDA default, W1 | existing split BF16 | merged BF16 `ba` | One BA GEMM; qkv/z unchanged. |
| 27B CUDA default, W2 | merged BF16 `qkvz` | merged BF16 `ba` | Two total input GEMMs per GDN layer. |
| 27B CUDA master/sub-toggle off | split logical views from the merged owner | split logical views from the merged owner | Four legacy GEMMs, no duplicated resident weights. |
| 27B CPU | split logical views initially | split logical views initially | Preserve the reference arithmetic; a CPU merged default needs its own benchmark checkpoint. |
| 35B native | existing FP8 qkv/z | existing BF16 b/a | Inert in W1/W2; qkv/z have quant scales and belong to `KERNEL-GEMM-FP8`. |
| 35B `VT_DENSE_NATIVE=0`, GGUF, synthetic fixtures | existing split fields | existing split fields | Inert; correctness tests prove no accidental selection. |
| Future LoRA/TP>1 | upstream may create separate projections or shard packed rows | upstream may disable TP for BA | Explicit fallback until those rows are independently ported and gated. |

The first leaf does not silently generalize 27B BF16 packing to FP8, GGUF,
LoRA, tensor parallel, Qwen3-Next's interleaved GQA layout, or other models.
It does not change FP4 tactics, recurrence/WY, attention, RMSNorm arithmetic,
scheduler behavior, lm_head, or generic cuBLASLt heuristic selection.

## Upstream chain

The parity source is vLLM v0.25.0 target
`702f4814fe54fabff350d43cb753ae3e47c0c276`; the same topology is present at
the current port pin `e24d1b24fe96a56ba8b0d653efa076d03eb95d6c`.

| Layer | Exact source | Behavior to mirror |
|---|---|---|
| Qwen3.5 checkpoint mapper | `vllm/model_executor/models/qwen3_5.py:200-210` | Stack q/k/v shards then z into `in_proj_qkvz`, and b then a into `in_proj_ba`. |
| Packed-module declaration | same file `:278-288,393-396` | Advertise the two physical merged modules for text and multimodal wrappers. |
| GDN construction | `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:465-496` | Construct one qkvz and one ba projection. |
| Output widths | same file `:566-614` | Qwen3.5 sizes are `[key,key,value,value]` and `[Hv,Hv]`; Qwen3-Next interleaving is a distinct mode. |
| CUDA execution | same file `:908-943` | Invoke qkvz once and ba once, then expose `[qkv,z]` and `[b,a]` logical slices. CUDA makes b/a contiguous before the core; our stride-aware consumer is a recorded launch-saving deviation. |
| CPU execution | same file `:1025-1043` | The same two projections precede CPU core execution. |
| Physical packed parameter | `vllm/model_executor/layers/linear.py:580-636,665-808` | `MergedColumnParallelLinear` owns one concatenated output-dimension parameter and loads shards into exact row ranges. |
| Unquantized apply | same file `:182-228,551-569`; `vllm/model_executor/layers/utils.py:92-99` | One quant-method apply reaches one `torch.nn.functional.linear` per physical projection. |
| Runtime resolution | finalized c2 evidence below | Runtime shapes resolve to one BF16 tensor-core qkvz and one BF16 BA GEMM per GDN layer. Algorithm/tactic tuning follows the corrected shapes; it is not a substitute for packing. |

Nearest executable upstream tests are
`tests/kernels/mamba/test_gdn_forward_core_split.py:74-255` and
`tests/kernels/test_fused_gdn_post_conv.py:1-154`. Neither directly asserts
the Qwen mapper's physical packing or GEMM count, so the mapper, merged-linear
loader and forward code above are also executable specification and require a
local structural regression.

## Our baseline

| Surface | Current anchor | Exact gap |
|---|---|---|
| Weight ownership | `include/vllm/model_executor/models/qwen3_5_weights.h:138-166` | Four independent qkv/z/b/a owners. |
| Real 27B loader | `src/vllm/model_executor/models/qwen3_5_dense_weights.cpp:159-179` | Loads four raw-NK BF16 tensors separately. |
| Dense forward | `src/vllm/model_executor/models/qwen3_5.cpp:2057-2124` | Four independent GEMMs; comment incorrectly says separation is retained at TP=1. |
| Paged forward | same file `:2299-2343` | The binding path also issues qkv, z, b and a separately. |
| BF16 GEMM | same file `:781-805`; `include/vt/ops.h:476-488`; `src/vt/cuda/cuda_matmul.cu:183-255` | Each raw-NK owner reaches a distinct cuBLASLt TN call. |
| Strided tensors | `include/vt/tensor.h:14-31`; `src/vt/tensor.cpp:70-79` | Slices retain real strides, but current GDN consumers reject or ignore them. |
| Causal-conv consumer | `src/vt/ops.cpp:780-811`; `src/vt/cuda/cuda_gdn.cu:248-503` | Requires contiguous x and indexes rows as `row*C`. |
| BA consumer | `src/vt/ops.cpp:1466-1487,1516-1569`; `src/vt/cuda/cuda_gdn.cu:646-728` | Requires separate contiguous F32 a/b although vLLM's merged projection emits model-dtype BF16. |
| z consumer | `src/vt/ops.cpp:912-929`; `src/vt/cuda/cuda_gdn.cu:753-805` | Gated RMSNorm requires a contiguous rank-2 gate and cannot express `[T,Hv,Dv]` with a wider outer stride. |
| Packed-view precedent | `src/vllm/model_executor/models/qwen3_5.cpp:1423-1458`; `src/vt/ops.cpp:760-779` | Full-attention merged QKV already owns one output and teaches its fused consumer to honor row strides. |

The completed evidence root is
`~/work/vllm.cpp-executed-path-c2/179a0fc2afc1c33b63d14de8e50d3fde976c7356`.
Status / summary / manifest / artifact-set SHA-256 values are
`9e0143fa...7b57` / `0ef6a124...0273` / `2556cfd0...2f21` /
`cc248ad2...823a`. Every one of 12 local B=2 graph ranges is invariant at
1,011 kernels; the oracle has 1,522 steady B=2 windows at 1,160 kernels.

| BF16 geometry per window | Local | vLLM v0.25.0 |
|---|---:|---:|
| qkv / packed qkvz | 48 calls, grid `(8,80,1)` | 48 calls, grid `(8,128,1)` |
| z | 48 calls, grid `(8,48,1)` | included in qkvz |
| b+a / packed ba | 96 calls, grid `(4,1,1)` | 48 calls, grid `(8,1,1)` |
| lm_head | 1 | 1 |
| Total BF16 GEMMs | **193** | **97** |

Thus `(4-2) x 48 = 96` explains the entire BF16 call-count residual. Diagnostic
family medians are 51.662672 vs 48.798042 ms (+2.864630 ms), but cross-profiler
durations do not bind. A shape-level decomposition ranks BA first at roughly
1.882 ms of diagnostic headroom and qkvz second at roughly 0.476 ms. Total
kernel counts must not be forced equal: vLLM has 149 more kernels overall in
other families.

Retaining both layouts would duplicate
`(16384+96)*5120*2*48 = 8,100,249,600` bytes (about 7.545 GiB) of 27B host
weights. That design is prohibited. BA-only duplication is also prohibited.

## Port map

| Upstream behavior | Local destination | Port / deviation |
|---|---|---|
| One qkvz and one ba physical parameter | `include/vllm/model_executor/models/qwen3_5_weights.h`; `src/vllm/model_executor/models/qwen3_5_dense_weights.cpp` | Add merged raw-NK owners; 27B loader concatenates checkpoint rows once and leaves corresponding split owners empty. |
| Merged loader shard order | dense-loader helpers plus focused loader test | Copy exact `[q,k,v,z]` and `[b,a]` row ranges; validate shapes/dtypes/nk flag and fail on missing/mismatched shards. |
| Rollback without duplicate storage | `src/vllm/model_executor/models/qwen3_5.cpp` | Resident merged weight is sliced on output rows for legacy GEMMs. Dim-0 raw-NK slices remain contiguous. |
| BA projection | dense and paged GDN blocks | W1 emits one BF16 `[T,2Hv]`, exposes b then a views, and passes them to the fused or unfused downstream path. |
| BF16/strided b/a | `include/vt/ops.h`, `src/vt/ops.cpp`, `src/vt/cpu/cpu_ops.cpp`, `src/vt/cuda/cuda_gdn.cu`, `src/vt/cuda/cuda_glue.cu` | Accept F32 or BF16 inner-contiguous b/a with explicit row strides; upcast per element. Avoid cast and split-copy launches. |
| qkvz projection | dense and paged GDN blocks | W2 emits one BF16 `[T,conv_dim+value_dim]` and exposes mixed/z logical views. |
| Strided mixed/z consumers | causal-conv and gated-RMSNorm files above | Honor element strides. Gated RMSNorm accepts a rank-3 `[T,Hv,Dv]` gate so token-boundary padding is representable; output/core semantics stay unchanged. |
| Same-binary attribution | model dispatch helpers | Master plus leaf toggles are resolved outside hot loops; fallback performs exact split calls from merged owners. |
| Trace/perf record | `tools/bench/online_gate.py`, existing DGX driver, live status surfaces | Reuse the fail-closed B=2 node trace and exact online gate; add only the expected packed-family/count assertions. |

## Tests to port

Every ported test names the upstream module/case in its source header. A mode
we do not yet support is checked in as `SKIPPED` with the row dependency.

| Upstream executable specification | Local tier / required assertion |
|---|---|
| `qwen3_5.py` mapper plus `linear.py::MergedColumnParallelLinear.weight_loader` | New focused 27B dense-loader test: exact row order, byte equality, shapes, `nk`, empty split owners, one-owner byte accounting, malformed/missing shard failures. |
| `test_gdn_forward_core_split.py:74-255` | Extend `tests/vllm/models/test_qwen27_paged_forward.cpp`: its eight state-dtype/workload semantics or the closest existing matrix run with merged BA/qkvz and split rollback. |
| `test_fused_gdn_post_conv.py:1-154` | Extend `tests/vt/test_ops_gdn.cpp`: BF16/F32 b/a, row offsets, padded leading dimensions, canaries, B=1/2/4/16/32, fused and unfused g/beta equivalence. |
| Qwen GDN CUDA/CPU forward `:908-943,1025-1043` | Dense/paged small-model tests compare merged vs split outputs and state updates; exact projection-level BF16 oracle goldens cover BA rounding. |
| `tests/lora/test_layers.py:753-830,916-1010` packed-column slicing | Port applicable non-LoRA packed-column slice numerics; LoRA adapters remain `SKIPPED (MODEL-LORA)` rather than omitted. |
| `tests/lora/test_qwen35_densemodel_lora.py` | Tracked `SKIPPED (MODEL-LORA)` fallback case; confirms why separate projection dispatch remains part of the contract. |
| `tests/quantization/test_auto_round.py:673-775` | Track `in_proj_qkvz` name/mixed-config cases as `SKIPPED (QUANT-AUTOROUND)` until that quantizer exists; do not claim them as passing coverage. |

Local mandatory regressions also include:

- CUDA causal-conv and gated-RMSNorm strided-view operator vectors with
  non-zero offsets, padded outer strides, guard canaries and capture/replay;
- packed versus split GEMM numerics at B=1/2/4/16/32 and real dimensions;
- 27B real loader/model gate, **235/235 + 16/16** token correctness for default
  and every rollback arm from the same 64-plan fixture;
- 35B native and legacy-fallback correctness plus GGUF loader tests proving
  W1/W2 dispatch inertness;
- exact graph node-family/count assertions and zero new copy/cast nodes.

## Gates

### G0 — record, build and CPU safety

- The spike is merged before implementation; each leaf owns a claim and updates
  README, BENCHMARKS, roadmap, matrix, ledger and state at its checkpoint.
- Clean warning-as-error CPU build, full CTest, ASan/UBSan and record/doc
  checkers pass. Split-reference behavior stays byte-identical where selected.
- Loader accounting proves exactly one owner and zero host RSS/PSS increase.

### G1 — W1 merged BA correctness and safety

- Merged `[b,a]` loader/view/operator tests pass for all declared B values,
  fused/unfused paths and CUDA graph replay; compute-sanitizer reports zero
  errors/leaks and no capture allocation/free/sync/D2H appears.
- Real 27B default and `VT_GDN_MERGED_BA=0` each pass 235/235 plus 16/16 tokens
  from one frozen plan map. 35B/GGUF gates prove zero selection.
- No b/a cast, split-copy or materialization kernel is introduced.

### G2 — W1 exact structure and component performance

Under one uncontended lock, profile default/fallback with
`nsys --cuda-graph-trace=node`. Default must have exactly **145** BF16 GEMMs per
B=2 window: 48 qkv + 48 z + 48 ba + 1 lm_head. With no new nodes, local total
falls from 1,011 to **963**; all 12 ranges remain invariant. Fallback stays 193.
FP4=208, recurrence=48, FA2 main/combine=16/16, producers=144+64, memcpy=7 and
memset=1 remain unchanged.

Then run same-binary c2/c16 BA-default versus BA-off in AB/BA/AB order, three
repetitions, one frozen plan map and one lock. All 40 timing and 8 memory axes
must be no worse, correctness is a precondition, and the result is recorded
even if negative. W2 may not start until this checkpoint is complete.

### G3 — W2 merged qkvz correctness, structure and component performance

- Repeat G1 for qkvz strided causal-conv and gated-RMSNorm consumers.
- Exact default B=2 structure becomes **97** BF16 GEMMs: 48 qkvz + 48 ba +
  1 lm_head. With no new copy/cast nodes local total becomes **915**. Do not
  require equality with vLLM's 1,160 total kernels.
- Run the same c2/c16 AB/BA/AB 40+8-axis component against
  `VT_GDN_MERGED_QKVZ=0`; record a passing or failed checkpoint before stacking
  another speed-sensitive change.

### G4 — conditional exact oracle grid and memory

After the selected merged defaults pass token, safety, structure and component
gates, rerun the full cache-off input-1024/output-128 c1/c2/c4/c8/c16/c32 grid
against a fresh vLLM v0.25.0 `702f481` + FlashInfer 0.6.13 denominator. Three
interleaved repetitions, every throughput/latency axis and all memory axes must
be direction-normalized >=1.0. A miss leaves parity open and resumes the scan.
No 35B performance command runs before 27B reaches 124/124.

The first implementation reproduction entry point is:

```sh
cmake -S . -B build-gdn-merge -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DVLLM_CPP_CUDA=ON \
  -DVLLM_CPP_TRITON=ON -DVLLM_CPP_SERVER=ON \
  -DVLLM_CPP_CUDA_ARCHITECTURES=121a
cmake --build build-gdn-merge --target test_ops_gdn \
  test_qwen27_dense_forward test_qwen27_paged_forward \
  test_qwen27_paged_engine test_qwen36_paged_engine server --parallel
flock /tmp/gpu -c 'ctest --test-dir build-gdn-merge \
  -R "(test_ops_gdn|test_qwen27|test_qwen36_paged_engine)" \
  --output-on-failure'
```

Immutable model, trace and online-series commands follow the existing
`scripts/dgx-online-serving.sh` and `docs/BENCHMARKS.md` recipes, with the
commit SHA, toggle arm and frozen fixture recorded before execution.

## Dependencies

- `SERVE-GATE-ONLINE` and its finalized `179a0fc` B=2 trace contract.
- Current 27B frozen 64-plan FlashInfer fixture and vLLM v0.25.0 oracle.
- `KERNEL-GDN-AOT-BF16`, `KERNEL-GDN-SCRATCH`, `KERNEL-GEMM-NVFP4-W4A4`
  and `KERNEL-ATTN-FA2` remain unchanged correctness/trace preconditions.
- CUDA 13.0.88, sm_121a GB10, Nsight Systems 2025.3.2 and one uninterrupted
  `/tmp/gpu` ownership window for each trace/A-B series.
- `KERNEL-GEMM-FP8`, tensor parallelism, LoRA and AutoRound are explicit
  future dependencies for the modes kept out of W1/W2.

## Work breakdown

| Leaf | Owned work | Entry/exit |
|---|---|---|
| W1A | Add the 27B `in_proj_ba` owner, exact loader packing and split weight views; port loader/storage tests. | No duplicate bytes; CPU/split behavior green. |
| W1B | Add BF16/strided b/a consumers and one merged BA GEMM in dense/paged forwards with master/leaf fallback. | G0/G1 green. |
| W1C | Run exact 145-node-family trace and c2/c16 BA AB/BA/AB. | G2 disposition committed before W2. |
| W2A | Add `in_proj_qkvz` owner/loader and strided causal-conv/gated-RMSNorm consumers with fallback. | G0/G1-equivalent tests green. |
| W2B | Run exact 97-node-family trace and c2/c16 qkvz AB/BA/AB. | G3 disposition committed. |
| W3 | Conditional fresh vLLM grid and host/GPU memory campaign. | 124/124 closes 27B; otherwise select the next trace-grounded lever. |

Each performance-sensitive leaf is isolated in one same-binary toggle and gets
its own documentation/benchmark checkpoint. W2 is never silently combined with
an ungated W1.

## Risks/decisions

- **BF16 BA changes a rounding point.** This is intentional upstream parity,
  not a free numerical substitution. Oracle/operator and 16/16 token gates
  decide adoption; any failure leaves W1 default-off.
- **Views can erase the launch win if materialized.** Consumers must honor
  strides directly. A cast/split-copy kernel fails the structural gate even if
  output is correct.
- **Packed and split owners are forbidden.** The split rollback slices the
  packed owner. This avoids a 7.545 GiB duplicate-weight design and protects
  the already-failing host-memory axes.
- **Launch count is not throughput.** BA is ranked first by a diagnostic
  decomposition, but only AB/BA/AB and the fresh grid bind. Neither a 48-call
  reduction nor the 2.864630 ms cross-profiler delta is a speed claim.
- **Total graph counts differ for valid reasons.** Assert the BF16 family and
  unchanged local families, never equality with vLLM's 1,160 total nodes.
- **35B qkv/z is FP8.** It stays outside this BF16 leaf; forcing it into the
  same representation would lose quantization-scale semantics.
- **Upstream CUDA copies b/a after slicing.** Our direct strided consumption is
  a deliberate surpass-side deviation, recorded because it removes work while
  preserving the same BF16 values and formulas.
- **Rollback boundary:** `VT_GDN_MERGED_PROJ=0`; leaf toggles isolate BA and
  qkvz. Any correctness, sanitizer, graph, memory or component regression keeps
  the affected leaf off and records the failed disposition.
