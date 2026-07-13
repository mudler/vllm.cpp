# W3-G — FlashAttention-2 GQA split-KV decode

Status: **READY** on 2026-07-13. This is a spike and execution contract, not an
implementation or performance claim. The binding 27B result remains immutable
`3f256ab` at **55/124 pass, 69 fail**.

## Scope

Primary row: `KERNEL-ATTN-FA2`. Consumer row: `SERVE-GATE-ONLINE`. The
hardware/reference disposition is recorded by `BACKEND-GATE-CUDA-VLLM`.

The first implementation slice mirrors the exact FlashAttention-2 pure-decode
path selected by vLLM v0.25.0 for the failing Qwen3.6-27B gate workload:

- CUDA build with `VLLM_CPP_FLASH_ATTN`, BF16 query/KV/output, paged KV,
  head-dim 256, block size divisible by 16, pure decode
  (`num_tokens == num_reqs`), no ALiBi, no soft cap, no dropout and no sliding
  window;
- the 27B topology only: 24 query heads / 4 KV heads, or GQA ratio 6. The
  Qwen3.6-35B-A3B gate model is 16/2, ratio 8, and is deliberately ineligible;
- upstream's `seqlenq_ngroups_swapped` interpretation, exact split-count
  heuristic, paged-KV main kernel and split-KV combine kernel;
- a default-on `VT_FA2_DECODE` selection. `VT_FA2_DECODE=0` restores the
  current F32 `PagedAttentionDecodeOptKernel` route in the same binary;
- model-side BF16 query/output selection only when the CUDA adapter is eligible,
  so no F32↔BF16 cast kernel is introduced. The existing FA2 prefill selection
  and `VT_FA2_PREFILL` remain independent and unchanged;
- CUDA-graph capture/replay with stable scratch pointers and no allocation,
  free, D2H copy or stream synchronization inside capture or replay.

Out of this slice: the 35B ratio-8 path; FP16/F32/FP8 Q or KV; head dimensions
other than 256; MHA or other GQA ratios; mixed prefill/decode calls; local or
sliding-window attention; ALiBi, sinks, soft cap, dropout, cascade attention,
quantized KV, FA3/FA4, and non-CUDA backends. Each stays on the current fallback
and needs its own row-sized expansion/gate before support can broaden.

The scope does not retire the backend-neutral paged-attention contract. It
modernizes one executed CUDA specialization after vLLM v0.25 retired its old
`paged_attention_v1/v2` implementation. No accepted speed number or broader
support state changes until every gate below passes.

## Why this lever is first

The post-W3-F multi-lens scan converged independently on decode attention. In
the immutable binding trace, our ratio-6 gate workload cannot use the hard-coded
ratio-8 GQA kernel (`kDecGqaQG=8`) and therefore executes
`PagedAttentionDecodeOptKernel<float,bfloat16,float>` **22,893** times for
**8,793.238 ms**, averaging **384.102 us** and consuming **3.083%** of local GPU
kernel time. vLLM's retained trace executes the FA2 split-KV main kernel
**23,616** times for **7,061.921 ms** plus the combine kernel **23,488** times
for **123.245 ms**. Because the trace windows and call counts differ, these are
diagnostic attribution—not an engine speed ratio. Per observed layer call they
identify roughly 80 us, or about 1.28 ms over 16 full-attention layers, of
structural headroom.

The next-ranked executed mismatch is the normal BF16→FP4 producer: local scalar
loads/stores versus vLLM's vectorized producer, with about 0.50 ms/forward of
diagnostic gap. It stays second and must not be stacked into W3-G. Host weight
release remains the separate leading PSS/RSS repair.

Trace evidence:

- ours SQLite:
  `~/work/vllm.cpp-online-gate/evidence/3f256abdbb558e162bf8a2196284deb119648560/trace/27/ours.sqlite`,
  SHA-256 `99cbd04d860fa77b8cf7c340d554dc72a3fd24a236839b1bd6106285c4c593f8`;
- vLLM kernel export in the same directory, SHA-256
  `e4e916d17dade7b3a90756e21dd8bebd0bb64d9a4f8ecc404144d2411e0a3565`;
- trace status SHA-256
  `9762c1e65bcedfc9e5c5c1a71fac4c68b1d35a6216e54510b6d312f7c481d0c6`.

## Upstream chain

The behavior is owned jointly by vLLM orchestration and its exact FA2
dependency; the dependency, not vLLM's Python layer, implements the swap,
heuristic and kernels.

| Layer | Pinned source | Required behavior |
|---|---|---|
| vLLM backend | `vllm/v1/attention/backends/flash_attn.py:674-721,935-977` at v0.25.0 target `702f4814fe54fabff350d43cb753ae3e47c0c276` | Records Hq/Hkv, selects FA2, passes paged cache, `seqused_k`, block table and `num_splits` to `flash_attn_varlen_func`. |
| Dependency pin | `cmake/external_projects/vllm_flash_attn.cmake:39-43` at `702f481` | Pins `vllm-project/flash-attention` commit `2c839c33742309ec41e620bf837495ec9926c56e`, already vendored locally. No dependency refresh is needed. |
| Pure-decode swap | `csrc/flash_attn/flash_api.cpp:587-646` at FA2 `2c839c33` | For `max_seqlen_q==1`, Hq>Hkv, no window/dropout/ALiBi and aligned D, interprets Q as `[B,Hkv,G,D] -> [B,G,Hkv,D]`, sets `seqlen_q=G`, `h=Hkv`, clears `cu_seqlens_q`, and uses temporary output storage when the supplied output has the original layout. |
| Split heuristic/workspace | same file `:262-327` | Uses block-N 64 at D=256; returns one split when work reaches 80% of `2*numSM`, otherwise selects the smallest eligible split with at least 85% of the best wave efficiency; partial LSE/output are F32. |
| Paged dispatch | same file `:698-714` | Only the swapped pure-decode branch enters `set_params_splitkv`; ordinary paged varlen remains restricted to at most one split. This corrects the old local comment that FA2 never split paged decode. |
| Result layout | same file `:754-779` | Restores `[B,Hq,D]` output and the canonical LSE layout after the swapped decode. |
| Kernel ABI | `src/flash_attn/src/flash.h`, `flash_fwd_kernel.h:541-548,1034-1044,1289-1293`, `flash_fwd_launch_template.h` at FA2 `2c839c33` | `Flash_fwd_params` accepts independent batch/row/head strides; split main writes F32 accumulators and combine writes the requested output strides. |
| Upstream tests | `tests/kernels/attention/test_flash_attn.py:95-217` at `702f481` | `test_varlen_with_paged_kv`, including pure-decode lengths `(1,523)/(1,37)/(1,2011)`, paged layouts, output supplied/allocated variants, GQA and reference comparison. |

The executable vLLM trace is decisive: it contains both
`flash_fwd_splitkv_kernel<...bf16...>` and
`flash_fwd_splitkv_combine_kernel<...>` on the exact 27B workload. Source-only
interpretations that treat the wrapper's ordinary-varlen `num_splits>1`
rejection as applying to pure decode are false; `seqlenq_ngroups_swapped` is a
separate dependency path.

## Our baseline

| Surface | Current anchor | Gap |
|---|---|---|
| Decode dispatch | `src/vt/cuda/cuda_paged_attn.cu:208-210,1854-1923` | The fused GQA specialization is fixed at ratio 8. Ratio 6 falls to one block per query head, re-reading K/V across the six heads. |
| FA2 adapter | `src/vt/cuda/cuda_flash_attn_fa2.cu:1-52,126-265` | It is prefill-only, pins `num_splits=1`, owns only final LSE scratch and contains a now-false claim that decode already matches vLLM. |
| Paged dispatcher | `src/vt/cuda/cuda_paged_attn.cu:2400-2522` | `fa2` is restricted to prefill. Decode uses F32 Q/output even though the executable vLLM path is BF16. |
| Model dtype selection | `src/vllm/model_executor/models/qwen3_5.cpp:952-980,2660-2780` | BF16 Q/output is selected only for FA2 prefill; pure decode is explicitly kept F32. |
| CUDA graph | `src/vllm/model_executor/models/qwen3_5.cpp:4425-4890` | Cold eager step prewarms scratch, next step captures, and block-table column changes invalidate/re-warm a size slot. FA2 decode scratch must fit this lifecycle without invalidating pointers held by other captured sizes. |
| Tests | `tests/vt/test_ops_paged_attn.cpp:785-1090` | FA2 coverage is mixed/ragged prefill at ratio 8. There is no ratio-6 pure-decode split/combine, fallback, graph replay or scratch-lifecycle vector. |

The local FA2 source is the same dependency commit vLLM v0.25 uses. W3-G adds
only the missing thin adapter/dispatch/lifecycle surface and tests; it does not
fork or rewrite the tuned kernel.

## Port map

| Upstream/dependency source | Local destination | Port/deviation |
|---|---|---|
| `flash_api.cpp` pure-decode eligibility and swap | `src/vt/cuda/cuda_paged_attn.cu`; `src/vllm/model_executor/models/qwen3_5.cpp` | Mirror exact eligibility and BF16 dtype selection. Bound the first default to ratio 6 so 35B ratio 8 remains inert. |
| `flash_api.cpp::num_splits_heuristic` / `set_params_splitkv` | `src/vt/cuda/cuda_flash_attn_fa2.cu` | Port arithmetic verbatim, using the runtime device SM count and block-table capacity as the capture-stable K upper bound. Device `seqused_k` remains the actual per-request length. |
| Tensor transpose/temporary output | same adapter | Prefer the dependency's supported independent Q/O strides to present the exact logical `[B,G,Hkv,D]` view directly over `[B,Hkv,G,D]`. This removes two layout copies but preserves the same indexing. Tests must compare it against an explicit host transpose/scatter reference; if any vendored kernel path assumes contiguity despite the ABI, use explicit capture-safe pack/unpack buffers instead. |
| F32 final/partial LSE and output accumulators | same adapter plus `src/vt/cuda/cuda_flash_attn_fa2_internal.h` and `src/vt/cuda/cuda_backend.cu` if a lifecycle hook is required | Key stable storage by device, stream, padded batch and block-table columns. Cold eager execution allocates; capture/replay only reuses. DestroyQueue releases every allocation after synchronization. Never free/regrow storage referenced by another live graph. |
| `run_mha_fwd_splitkv_dispatch<bf16,256,false>` | existing vendored instantiations and adapter | Reuse the exact compiled FA2 main/combine implementation. Pure decode is non-causal after the swap, as upstream specifies; all logical queries still represent one time position. |
| vLLM toggle/config behavior | model + CUDA dispatch | `VT_FA2_DECODE=0` is a diagnostic same-binary fallback. Default-on behavior is allowed only after correctness, trace and every-axis gates pass; until then the implementation checkpoint remains `ACTIVE/GATING`. |

The adapter must not use one grow-and-free global buffer: another captured graph
size could retain its old pointer. It also must not derive capture geometry from
a growing actual sequence length. `block_table.shape[1] * block_size` is the
stable upper bound for one slot; the graph driver already destroys and cold-
warms the slot when the column count changes.

## Tests to port

Port the executable semantics from upstream
`tests/kernels/attention/test_flash_attn.py::test_varlen_with_paged_kv` into
`tests/vt/test_ops_paged_attn.cpp`, retaining the upstream file/case citation in
the test header.

| Vector | Local tier and assertion |
|---|---|
| Upstream pure decode `(1,523)/(1,37)/(1,2011)`, Hq/Hkv 8/2, D256, block16, BF16 | CUDA op test; compare FA2 output with the composed F32 paged reference on the same BF16-rounded Q/K/V using upstream-class `atol=1.5e-2`, `rtol=1e-2`. |
| Exact gate ratio Hq/Hkv 24/4 | CUDA op test at B=1/2/4/8/16 and block-spanning sequence lengths around 1,024-1,152; prove FA2 ON and fallback each match the reference. |
| Layout interpretation | Compare direct-stride output with an explicit `[B,Hkv,G,D] -> [B,G,Hkv,D]` host reference and restore `[B,Hq,D]`; cover nontrivial head-specific values so a head/group transpose cannot pass accidentally. |
| Split heuristic | CUDA-free/helper vectors for `num_n_blocks`, SM count and B/Hkv combinations, including 1 and >1 splits; assert the exact upstream 80%/85% eligibility decisions and `<=128`. |
| Split and no-split runtime | Diagnostic hook/counter or trace assertion proves the expected main-only versus main+combine dispatch while outputs remain equivalent. |
| Same-binary fallback | Toggle `VT_FA2_DECODE=1/0` in process; ON selects FA2 only for exact eligibility, OFF selects `PagedAttentionDecodeOptKernel`; invalid dtype, D, qpk, mixed-prefill and window cases fall back without changing results. |
| Capture/replay | Cold eager → capture → at least two replays at each padded B; grow actual `seq_lens` without changing block-table columns and prove no allocation/free/sync/D2H event in capture/replay plus stable output. Change columns, prove graph invalidation/cold rewarm and no stale pointer. |
| Lifecycle/concurrency | Two queues and two padded sizes retain disjoint stable scratch; destroying a queue frees its FA2 storage. Strict no-pool compute-sanitizer reports zero errors and zero leaked bytes. |
| Model gate | Existing `test_qwen27_paged_engine` passes **235/235 + 16/16** with FA2 ON and OFF from the same frozen 64-plan map. Existing correctness-only 35B test proves ratio-8 dispatch inertness; it is not a 35B performance result. |

If a ported vector cannot run in CPU CI it remains compiled and reports the
existing `HasCuda()` hardware skip; it is mandatory in the immutable GB10 gate,
not silently dropped.

## Gates

### G0 — record and build

- clean warning-as-error CPU build and full CTest remain green;
- clean CUDA 13.0.88, GCC 13.3, sm_121a build with the exact FA2/CUTLASS pin;
- `scripts/check-agent-record.py`, mutation suite, staged documentation checker
  and `git diff --check` pass at every checkpoint.

### G1 — operator correctness and safety

- every ported ratio-2 and ratio-6 vector passes within the declared BF16
  tolerance; ON/OFF eligibility and layout tests pass;
- focused Release, ASan+UBSan and TSan pass locally where applicable;
- strict `compute-sanitizer` covers initialization, cold eager, capture and
  replay with zero errors and zero leaked bytes;
- no capture/replay `cudaMalloc*`, `cudaFree*`, D2H memcpy or
  `cudaStreamSynchronize` appears in the API trace.

### G2 — model correctness and inertness

- 27B default and `VT_FA2_DECODE=0` each pass the existing **235/235 + 16/16**
  frozen-plan gate, with all 64 plan IDs and metadata identical;
- a correctness-only 35B W4A16 run passes its existing assertions and proves
  zero FA2-decode selection. No 35B rate is measured or inferred;
- default and fallback controlled same-shape long-output comparison remains
  consistent with the W3-C3R production correctness contract.

### G3 — structural execution trace

Profile default and fallback with `nsys --cuda-graph-trace=node` under one
uncontended lock. Default must contain the exact FA2 split-KV main and combine
kernels in decode nodes and zero local `PagedAttentionDecodeOptKernel` for the
eligible 27B full-attention calls. Fallback must show the reverse. Prefill,
3,536 FP4 GEMMs/producers, eight CUTLASS identities, the frozen 64-plan map and
all non-attention kernel families must remain unchanged. Trace warmup/capture
is separated from the steady slice before wall-time attribution.

### G4 — bounded component

With G0-G3 green, run cache-off input-1,024/output-128 greedy c2 and c16
FA2-default versus `VT_FA2_DECODE=0` in AB/BA/AB order, three repetitions, one
immutable binary, one frozen 64-plan map and one uninterrupted `/tmp/gpu` lock.
Both arms rerun the model gate. All 40 timing and all 8 memory axes must pass;
any miss means W3-G receives no speed credit and the exact grid does not run.
Record requests/tokens, temperatures, cache drops, CVs, process/plan metadata,
raw hashes and idle GPU/port/lock exit.

### G5 — conditional exact vLLM grid

Only after **every G4 axis** passes, execute the full c1/c2/c4/c8/c16/c32 27B
online gate against the unchanged vLLM v0.25.0 `702f481` + FlashInfer 0.6.13
oracle, identical corpus/config and three interleaved repetitions. Correctness,
all 120 timing/throughput axes and all four memory axes must be at least 1.0
direction-normalized. Otherwise parity remains open and the scan loop resumes.
No 35B performance command is authorized until the 27B grid is 124/124.

### Reproduction commands

Read-only trace attribution:

```sh
ssh dgx.casa 'ROOT=~/work/vllm.cpp-online-gate/evidence/3f256abdbb558e162bf8a2196284deb119648560/trace/27; \
  sqlite3 -header -column "$ROOT/ours.sqlite" "SELECT s.value,COUNT(*),ROUND(SUM(k.end-k.start)/1e6,3),ROUND(AVG(k.end-k.start)/1e3,3) FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName WHERE instr(s.value,'"'"'PagedAttentionDecode'"'"')>0 GROUP BY s.value;"; \
  jq ".kernels[] | select((.name // \"\")|test(\"flash_fwd_splitkv\"))" "$ROOT/vllm-kernels.json"'
```

First immutable implementation handoff, from the pushed commit on `dgx.casa`:

```sh
SHA=$(git rev-parse HEAD)
BUILD="$HOME/work/vllm.cpp-fa2-decode/$SHA/build"
cmake -S . -B "$BUILD" -DVLLM_CPP_CUDA=ON -DVLLM_CPP_TRITON=ON \
  -DCMAKE_CUDA_ARCHITECTURES=121a \
  -DVLLM_CPP_CUTLASS_DIR="$HOME/venvs/vllm-oracle/lib/python3.12/site-packages/flashinfer/data/cutlass" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build "$BUILD" --target test_ops_paged_attn \
  test_qwen27_paged_engine test_qwen36_paged_engine server --parallel "$(nproc)"
flock /tmp/gpu sh -c '
  ctest --test-dir '"'"'"$BUILD"'"'"' -R "^(test_ops_paged_attn|test_qwen27_paged_engine)$" --output-on-failure &&
  VT_FA2_DECODE=0 ctest --test-dir '"'"'"$BUILD"'"'"' -R "^(test_ops_paged_attn|test_qwen27_paged_engine)$" --output-on-failure
'
```

The component and exact-grid driver are materialized under the immutable
evidence root only after G0-G3 pass; they must refuse overwrite and bind their
own SHA-256/provenance before execution.

## Dependencies

- rows `KERNEL-ATTN-FA2`, `SERVE-GATE-ONLINE` and
  `BACKEND-GATE-CUDA-VLLM`; claim `CLAIM-SERVE-GATE-1`;
- current W3-C frozen 64-plan reproduction control and W3-F structural evidence;
- vLLM v0.25.0 target `702f481`, executable oracle and FlashInfer 0.6.13;
- already-vendored FA2 `2c839c33`, CUTLASS 4.5 source used by the CUDA build,
  CUDA 13.0.88/GCC 13.3 and GB10 sm_121a;
- Qwen3.6-27B NVFP4 snapshot `890bdef7a42feba6d83b6e17a03315c694112f2a`
  and the existing 27B oracle goldens; 35B is needed only for inertness;
- uncontended `dgx.casa` and one `/tmp/gpu` lock for each trace/A-B series.

No external package, model download or dependency-version change is part of
this checkpoint.

## Work breakdown

| Work ID | Files/ownership | Result and handoff |
|---|---|---|
| W3-G0 | this spec; kernel/engine/backend/feature/quant matrices; roadmap, coordination, inventory, README, BENCHMARKS, ledger, state | Accept the full spike, rank FA2 first/FP4 producer second, claim `KERNEL-ATTN-FA2`, publish no rate. |
| W3-G1 | `cuda_flash_attn_fa2.cu`, optional FA2 internal lifecycle header, `cuda_backend.cu` | Exact swapped-stride params, split heuristic, stable scratch and queue cleanup. No model default change yet. |
| W3-G2 | `cuda_paged_attn.cu`, `qwen3_5.cpp` | Exact ratio-6 eligibility, default/fallback dispatch and cast-free BF16 model path; prefill and ratio-8 remain unchanged. |
| W3-G3 | `tests/vt/test_ops_paged_attn.cpp`, focused build registration only if needed | Port upstream paged pure-decode vectors, ratio-6, heuristic, fallback, capture, lifecycle and invalid-eligibility tests. |
| W3-G4 | immutable DGX evidence only; same-change public/canonical status updates | G0-G3 build/operator/sanitizer/model/inertness/trace closure. Fail closed before timing. |
| W3-G5 | immutable c2/c16 driver/evidence; same-change public/canonical status updates | Strict 40+8-axis component; either authorize G6 or record failure and rescan. |
| W3-G6 | existing exact online gate/evidence; same-change public/canonical status updates | Conditional 27B c1-c32 vLLM grid. Only 124/124 permits 35B performance. |

The work is intentionally serial because W3-G2 consumes W3-G1's ABI and W3-G4
must freeze exactly one implementation. The vector FP4 producer and host-memory
repair are separate future claims, never folded into this A/B.

## Risks/decisions

- **Stride assumption:** the vendored ABI exposes independent batch/row/head
  strides and both main/combine paths use them. Tests nevertheless adversarially
  validate direct swapped strides. If any specialization assumes contiguous
  layout, use explicit prewarmed pack/unpack rather than weakening correctness.
- **Captured pointer lifetime:** a grow-and-free singleton would invalidate
  another live graph. Scratch is keyed by capture-stable shape and released only
  with its queue; allocation in capture is a hard test failure.
- **Actual versus bounded K:** fixed graph geometry uses block-table capacity;
  `seqused_k` supplies actual lengths. Column-count change already invalidates
  the graph. Tests grow actual lengths within a slot and then force a column
  change.
- **BF16 changes arithmetic:** this is intentional because it mirrors the
  executed vLLM path, but token-exact model gates remain mandatory. No speed win
  may trade away the existing 16/16 oracle result.
- **Padded decode rows:** graph padding rows carry valid block-table row 0 and
  discarded outputs. FA2 must remain in-bounds for them and may not mutate real
  rows; B<S tests cover this explicitly.
- **Split selection and CUDA graphs:** exact upstream auto-selection depends on
  B, heads, K blocks and SM count. We freeze it from capture-stable bounds per
  graph slot; the structural trace must prove the intended split/combine path.
- **No modernization by deletion without trace:** fallback remains until the
  FA2 route passes all correctness/safety/performance gates and broader ratios
  are separately covered. vLLM's removal of legacy PagedAttention does not make
  our backend-neutral contract obsolete.
- **Performance uncertainty:** the 1.28 ms/forward estimate is a trace diagnostic
  with unequal windows, not promised speed. Only the controlled component can
  earn credit.
