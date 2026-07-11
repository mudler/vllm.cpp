# Spike: device-resident KV and indexed GDN state I/O

**Row:** `KV-DEVICE-RESIDENCY` · **state:** accepted; implementation is
`ACTIVE` under `CLAIM-KV-DEVICE-1`.

## Scope

Make persistent full-attention KV, GDN convolution state, and GDN recurrent
state true backend allocations for the lifetime of `GPUModelRunner`. Then
replace mixed-prefill row-by-row state transfers with indexed device work,
mirroring pinned vLLM's cache residency and selection semantics. CPU storage
stays host-resident. CUDA graph pointers must remain stable.

This row owns storage residency, allocation/free/zero lifecycle, state-index
upload, BF16↔F32 indexed gather/scatter, and direct indexed convolution-state
updates. It does not own prefix-hit policy (`KV-PREFIX-CACHE`), general Mamba
kernel breadth (`KERNEL-SSM-MAMBA`), or unrelated secondary GDN kernel tuning.

## Trace-grounded problem

At the representative 27B mixed shape T=2048 plus 11 decode requests, ours
measured 952.960 ms versus vLLM's 900.398 ms. Summed kernels account for
32.491 ms of the 52.562 ms gap; the remaining non-kernel gap is 20.071 ms.

Ours performs exactly 816 state-row copies per direction for 14 non-spec
sequences (11 decode + 3 prefill):

- convolution: `14 * 48 * 61,440` bytes;
- recurrent state: `3 * 48 * 1,572,864` bytes;
- total: 267,780,096 bytes = 255.375 MiB per direction.

The nsys slice records 817 D2H calls / 255.375015 MiB and 966 H2D calls /
255.842133 MiB, approximately 1,799 `cudaMemcpyAsync` calls per context. Their
API intervals overlap prior GPU work and are not added to wall time, but they
force host/GPU lockstep. Pure b16 decode is effectively tied, so this mixed
state path precedes generic scheduler-overlap tuning.

## Upstream chain

- `vllm/v1/worker/gpu/attn_utils.py:166-182,327-346` allocates raw cache tensors
  directly on `device` and creates attention/Mamba views over that storage.
- `vllm/v1/worker/gpu/model_runner.py:478-488` installs those device cache
  tensors into the runner and forward context.
- `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:1309-1375`
  passes device state indices to convolution update/prefill paths.
- The same file at `:1503-1532` gathers and writes recurrent state with device
  indexing and in-place final-state semantics.
- Runtime dependencies are the resolved causal-conv and FLA/FlashInfer GDN
  kernels already inventoried in `kernel-matrix.md`; no Python/Torch dependency
  enters vllm.cpp.
- Local backend allocation and stream-ordered zeroing are
  `include/vt/backend.h:16-18,70-71`, `src/vt/backend.cpp:58-74`, and the selected
  CUDA backend. Copy-kind behavior is verified from execution traces, not
  inferred from API names.

## Our baseline

**Binding pre-W0 evidence.**

Pushed `f065fce` completed the corrected 27B cache-off/closed-loop series. Its
pre-W0 CUDA cache storage is owned by host `std::vector<uint8_t>` objects whose
addresses are wrapped in CUDA-tagged tensors. Mean c1→c32 throughput ratios are
0.9616/0.9258/0.9364/0.9450/0.9775/0.9722×; only 3–5/20 timing and 2/4 memory
axes pass, with throughput CV below 0.7%. The model, lifecycle and corrected
trace contracts pass (trace-status SHA-256 `f8de5789…1c7e2a`).

Across the three c16 trace repetitions, nsys records 153,394
`cudaMemcpyAsync` calls. The two state-row sizes alone account for 66,096 H2D
and 66,094 D2H transfers: 20.809 GiB H2D plus 20.806 GiB D2H. This is the
binding before-state that every implementation slice must reproduce against.
Current W0 source at `7d29e0c` replaces the owners. Its clean CUDA
13.0.88/sm_121a build, default/fallback 35B+27B pointer/token gates, lifecycle,
trace, repeated A/B, and both-model memory-access checks pass. Full leak-check
remains open on inherited process-lifetime pools; W0's cache owners are absent
from both leak inventories.

**W1 indexed-I/O checkpoint.**

The W1 source fingerprint `bec1ff5fd5f3` implements persistent full non-spec
and prefill indices/query boundaries/i8 initial-state masks plus CPU/CUDA
indexed gather/scatter. The mixed convolution path gathers/scatters once per
layer, and recurrent prefill fuses BF16→F32 indexing with fresh-row zeroing;
pure decode still updates its persistent cache in place. A turnover regression
ports pinned `tests/v1/worker/test_mamba_utils.py:342-358` and proves that both
decode consumers take only the leading decode subset of a mixed
decode+prefill index vector.

The exact CUDA 13.0.88/sm_121a build, CUDA op suite, focused indexed-op memcheck
(7/7, zero errors), indexed/fallback 27B+35B model tests, and a 16-concurrent
exact-usage turnover smoke pass. Focused CPU plus leak-disabled ASan+UBSan
access suites pass. The indexed op passes strict LSan; the local model suite's
strict run reports 58,624 bytes in 153 process-lifetime scratch-pool
allocations, so zero-leak remains open. Two full plain CPU CTest attempts
reached 104/105: only the unrelated, timing-sensitive C API early-stop
callback-count case flaked (isolated execution passed and
`--repeat until-fail:20` first failed on iteration five); W1 does not touch C
API/async code. The current local serial rerun passes 105/105, so the failure is
retained as an intermittent-flake disclosure rather than a W1 regression.
Source/build/runtime manifest hashes are
`2013507a…ffc`, `2d9af7a2…471`, and `736842d6…41f`.

The exact cache-off c16/48 same-binary AB/BA/AB series averages **781.799
indexed versus 776.946 fallback tok/s = 1.006246×**, improves all 20 online
throughput/latency axes, has 0.846%/0.530% CV, and returns memory in all six
legs. Its manifest/summary hashes are `34285a91…a5b` / `4c68b1dc…033`.
Structural traces reduce `cudaMemcpyAsync` 163,540→7,508, D2D calls
142,717→1,231, and D2D volume 49,088.289→1,855.918 MB; indexed gather/scatter
each execute 9,394 times. nsys perturbs the two paths unequally and inverts
profiled throughput (768.249/794.593 tok/s), so the trace supplies only
copy/kernel evidence and the unprofiled interleaved series owns the speed
ratio. This is still a component result, not a fresh vLLM denominator.

## Port map

| Concern | Current local anchor | Required disposition |
|---|---|---|
| cache ownership | `include/vllm/v1/worker/gpu/runner.h:224-241`; `src/vllm/v1/worker/gpu/runner.cpp:360-412` | replace CUDA `std::vector<uint8_t>` owners with move-only `vt::Alloc`/`vt::Free` backend allocations; keep CPU vectors or a unified RAII owner; zero on `queue_`; preserve stable view pointers |
| full-attention KV | `runner.cpp:395-406`; `src/vt/cuda/cuda_paged_attn.cu:2440-2548`; `src/vt/cuda/cuda_flash_attn_fa2.cu:119-267` | make the existing view device-resident without changing layout/shape/dtype; first same-binary A/B isolates residency |
| state row helpers | `include/vt/ops.h:867-877`; `src/vt/{ops.cpp,cpu/cpu_ops.cpp,cuda/cuda_gdn.cu}` | W1 implemented: CPU reference plus one CUDA indexed BF16→F32 gather / F32→BF16 scatter launch |
| convolution state | `qwen3_5.cpp:1737-1777` | W1 fused indexed I/O implemented; W2 passes cache indices into prefill convolution for direct update like upstream |
| recurrent state | `qwen3_5.cpp:1847-1892` | W1 implemented: one indexed gather/scatter launch per GDN layer, fused fresh-request zeroing, untouched slots preserved |
| indices | persistent upload `qwen3_5.cpp:1565-1653` | W1 implemented full non-spec and prefill index/query/mask arrays; decode consumers explicitly subview their leading rows |

Fallbacks for same-binary attribution are `VT_DEVICE_KV_CACHE=0` for storage and
`VT_GDN_INDEXED_STATE_IO=0` for indexed I/O. They are diagnostic opt-outs, not
alternate supported defaults.

## Tests to port and extend

| Upstream executable spec | Local test/evidence |
|---|---|
| `tests/v1/worker/test_gpu_model_runner.py:968-1088` KV allocation/view ownership | extend `tests/vllm/v1/worker/test_runner.cpp`; CUDA assertions use `cudaPointerGetAttributes` for every full-attention, convolution, and recurrent cache pointer |
| `tests/v1/worker/test_gpu_model_runner.py:1265-1285` Mamba component layout | shape/stride/dtype/view assertions in runner tests and both paged-forward suites |
| `tests/v1/worker/test_mamba_utils.py:342-358,483-520` state copy/index behavior | extend `tests/vt/test_ops_gdn.cpp` with non-contiguous indexed gather/scatter, BF16 persistent/F32 working state, mixed decode/prefill, initial/no-initial state, and byte-identical untouched slots |
| `tests/v1/attention/test_mamba_update_block_table.py:176-212` persistent state-index capacity | extend input/metadata runner tests for max sequence capacity and stable CUDA-graph buffers |
| real-model behavior | preserve `tests/vllm/models/test_qwen27_paged_forward.cpp`, `test_qwen35_paged_forward.cpp`, and both 16/16 parity engine gates |

## Gates

1. CPU: full CTest and sanitizers remain green; CPU cache ownership and f32
   reference results are unchanged.
2. CUDA correctness: pointer attributes prove device residency; indexed op
   parity covers all cases above; compute-sanitizer reports zero errors/leaks.
3. Real models: both 27B and 35B 16/16 greedy token gates pass for default and
   each fallback A/B.
4. Trace: on the exact corrected mixed context, W0 makes state-scale H2D/D2H
   disappear and changes the row traffic to D2D. W1 then collapses the row-copy
   calls to indexed-kernel/metadata traffic. Pure-decode graph time may not
   regress.
5. Performance: same-binary W0 and W1 A/Bs each reproduce at least three times;
   then fresh exact matched-config vLLM direct-library and online gates run per
   `benchmark-protocol.md`. No historical mismatched denominator counts.
6. Memory/lifecycle: peak VRAM remains no worse than vLLM, all allocations are
   freed after runner shutdown, and graph-captured addresses never move.
7. Backend scope: GB10/sm_121a is binding. CPU is correctness-only. Other CUDA
   targets remain unclaimed until their backend matrix gates run.

## Work breakdown

| Work | Deliverable | State |
|---|---|---|
| W0 | move-only persistent backend allocation owner; device KV/GDN caches; queue zero; pointer/lifecycle tests; `VT_DEVICE_KV_CACHE=0` A/B | implemented; CPU/CI, clean sm_121a pointer/model, lifecycle, trace and repeated A/B pass at 1.021239×/20-of-20 axes; both models are access-clean. Zero-leak fails inherited pools (47.29 MB/36.82 GB), with W0 cache allocations absent; teardown closure remains required |
| W1 | persistent full non-spec/prefill index upload plus fused indexed BF16↔F32 GDN state gather/scatter; `VT_GDN_INDEXED_STATE_IO=0` A/B | implemented; focused CPU/sanitizer/CUDA/op/model/turnover gates and current local serial CTest 105/105 pass; repeated A/B is 1.006246× and 20/20 axes; structural trace collapses async copies 163,540→7,508. Earlier remote full attempts retain the unrelated intermittent C API timing flake disclosed above |
| W2 | convolution prefill consumes indexed persistent state directly; preserve initial-state/reset semantics | open after W1; next performance lever if the fresh 27B oracle gate remains below floor and still required before feature-row closure |
| W3 | corrected trace, both real-model gates, direct-library and online every-axis comparison | fresh 27B measurement begins after W1 and is rerun after W2 if needed; 35B follows only after 27B closes |
| W4 | only if FA remains slow after residency: A/B the oracle's sm80 virtual FA binary target; otherwise rank gated RMSNorm, post-conv prep, causal conv, then `chunk_o` as separate kernel rows/specs | conditional |

## Dependencies

- `SERVE-GATE-ONLINE` supplies the corrected trace shape and consumes W3.
  `KV-PREFIX-CACHE` must stay explicitly off for the parity trace.
- `vt::Backend` allocation/free/memset, the existing runner lifetime, fixed
  `StepDevInputs`, CUDA graph capture, and GDN/FA kernels are prerequisites.
- W1 depends on W0 residency; W2 depends on W1 indices. W3 first measures the
  W0+W1 stack, consumes W2 if any 27B axis remains below floor, and then closes
  the 35B comparison.

## Risks and decisions

- Allocation failure must unwind previously allocated layers without leaking.
  The RAII owner is move-only; tensor views never own or free storage.
- Freeing while queued work or a captured graph can still reference the pointer
  is forbidden. Runner shutdown must synchronize through the existing queue/
  executor lifecycle before ownership is released.
- A host pointer wrapped with a CUDA device tag is not device residency. Tests
  inspect pointer attributes and traces inspect transfer direction/counts.
- The FA2 d0-to-d13 crossover is treated as a cache-locality consequence first.
  Its binary-target difference is a conditional follow-up, not justification
  to skip the residency A/B.
- Individual kernel deltas are not additive: ours already wins some nvjet GEMMs
  and mixed GDN decode recurrence. Re-profile after each slice before ranking
  secondary work.
