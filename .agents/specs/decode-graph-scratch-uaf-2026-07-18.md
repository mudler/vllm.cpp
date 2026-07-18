# 35B online-serving decode-graph scratch use-after-free (SERVE-GATE-ONLINE) — 2026-07-18

**Row:** `SERVE-GATE-ONLINE` (35B grid blocker). **Claim:**
`CLAIM-35B-GRAPH-SCRATCH-1`. **Kind:** correctness bug root-cause + fix (CUDA
graph capture/replay use-after-free). **Verdict:** ROOT-CAUSED (cuda-gdb-pinned
faulting kernel) and FIXED; 35B c2+ online serving no longer dies with an illegal
memory access.

## The failure (grounded)

`nvidia/Qwen3.6-35B-A3B-NVFP4` (MoE) ONLINE serving crashed at concurrency > 1:
`engine-fatal: EngineCore busy loop threw: vt cuda: cudaEventSynchronize: an
illegal memory access was encountered`, after which every request 500s with
"request submitted to a stopped AsyncLLM". The c1 leg always completed; c2+ died.
The offline single-stream gate `test_qwen36_paged_engine` passed 315/315, so the
bug was concurrency-triggered, not a shape bug. Evidence root
`dgx:~/work/vllm.cpp-online-gate/evidence/bcf7df7…`.

## Investigation (differential isolation, then cuda-gdb)

The IMA surfaces at the next `cudaEventSynchronize`; the real fault is an earlier
kernel launch. compute-sanitizer memcheck could NOT reproduce it (its
serialization keeps the freed block from being reused/released before the replay),
even at c8 — so the exact kernel had to come from a differential sweep + cuda-gdb.

Reliable repro found: one server, concurrency sweep c1→c2→c2→c4→c8→c16 (each leg
`vllm bench serve`, in1024/out128). Differential results (each 3–5 trials):

| Arm | Result |
|---|---|
| graphs ON + async ON (production) | **5/5 CRASH** |
| graphs ON + async OFF (`VT_ASYNC_RUNNER=0 VT_ASYNC_SCHED=0`) | **5/5 CRASH** → async NOT the cause |
| graphs ON + `VT_NVFP4_WMMA=0` | **2/2 CRASH** → the cutlass WMMA-scatter path NOT the cause |
| graphs OFF (`VLLM_CPP_CUDAGRAPH=0`) | **CLEAN** → the decode CUDA graph IS required |
| eager, short prompts, c2 | CLEAN → long context (multi-KV-block) required |
| compute-sanitizer memcheck (c2 and c8) | 0 errors (masks the bug) |

Then **cuda-gdb** (which catches the device exception even inside a graph replay,
without memcheck's serialization) pinned the faulting kernel deterministically:

```
CUDA Exception: Warp Illegal Address
#0  marlin_moe_wna16::Marlin<…, 128, 1, 8, 4, false, 4, 1, false>(…)<<<(144,1,1),(128,1,1)>>>
```
`dgx:~/work/vllm.cpp-35b-gdb/gdb.log`.

## Root cause

The 35B MoE runs through the **Marlin** grouped-GEMM (`MoeBlockFusedMarlinCuda` →
`MoeGroupedGemmNvfp4MarlinKernelCuda` → `marlin_mm`). Its fp32 global-reduce
scratch `c_tmp` comes from **`EnsureCtmp`** (`src/vt/cuda/cuda_moe_marlin.cu:75`),
a per-stream **grow-on-demand** buffer that, on growth, did
`cudaFreeAsync(old); cudaMallocAsync(bigger)`.

The pure-decode forward is captured into a CUDA graph whose kernel nodes BAKE the
device pointers returned at capture time — including `c_tmp` passed to `marlin_mm`.
`c_tmp`'s size is `size_n * sorted_token_ids` (scales with the token count). A
LATER, LARGER forward — a bigger co-scheduled prefill or a larger decode batch,
only reachable at concurrency > 1 — grows `c_tmp`; the `cudaFreeAsync` FREES the
block the captured decode graph still references, so the next graph replay's
`marlin_mm` reads freed memory → Warp Illegal Address (surfaced at the next
`cudaEventSynchronize`). This is exactly why **c1 never crashes** (after its one
prefill nothing ever grows `c_tmp` again → the baked pointer stays valid) while
c2+ does, why it is independent of async scheduling and of the cutlass WMMA path,
why long context (multi-KV-block → the crash appears once the graphed step does
real work over many blocks) is required, and why memcheck cannot reproduce it.

`EnsureCtmp` is one of a FAMILY of grow-on-free scratch allocators reached inside
decode graphs; the identical `cudaFree/cudaFreeAsync`-on-grow hazard exists in
`EnsureMoeScratch` (`cuda_matmul_nvfp4.cu`, the MoE WMMA-scatter index scratch),
the cutlass NVFP4 GEMM workspace (`cuda_matmul_nvfp4_cutlass.cu`) and the cutlass
FP8 GEMM workspace (`cuda_matmul_fp8_cutlass.cu`) — reached by other model/quant
decode paths (27B dense cutlass GEMMs, the nvfp4 WMMA path). The 35B/Marlin
instance is the cuda-gdb-proven one; the siblings are the same pattern.

## Fix

New portable, CPU-unit-testable helper `src/vt/cuda/graph_safe_scratch.h`:
`RetireGraphScratch(void*)` keeps the old block RESIDENT (retires, never frees) on
growth, so any pointer a captured graph baked stays valid for the graph's lifetime
— the graph only ever needs the size it had at capture, which is ≤ the retired
block's capacity. Growth is O(log(max/min)) so retired memory is bounded and
negligible (mirrors the pre-existing "scratch leaks at process exit" discipline of
the DevicePool and the cublasLt/Marlin workspaces). All four grow-on-free
allocators reached in the decode graph replace their `cudaFree`/`cudaFreeAsync`
on growth with `RetireGraphScratch`:

- `src/vt/cuda/cuda_moe_marlin.cu` `EnsureCtmp` — **the 35B fix (proven)**;
- `src/vt/cuda/cuda_matmul_nvfp4.cu` `EnsureMoeScratch`;
- `src/vt/cuda/cuda_matmul_nvfp4_cutlass.cu` `EnsureScratch`;
- `src/vt/cuda/cuda_matmul_fp8_cutlass.cu` `EnsureScratch`.

Capture behaviour is unchanged (growth only ever happens OUTSIDE capture — the
nvfp4/cutlass path even asserts "high-water miss during CUDA graph capture"); only
the free-on-grow becomes a retire-on-grow.

## Tests to port / add

- CPU tier: `tests/vt/test_graph_safe_scratch.cpp` pins the never-free retire
  bookkeeping (structural: the helper exposes no free path).
- Device proof (DGX serving harness): the concurrency-sweep serving harness at
  async-ON production defaults must go 0-crash over ≥5 trials (the pre-fix repro
  was 5/5 crash). Token-exactness `test_qwen27_paged_engine` 235/235 +
  `test_qwen36_paged_engine` 315/315. memcheck c2 35B 0 errors.

## Gates

- 35B concurrency-sweep serving harness (async ON, graphs ON): 0 crashes / ≥5.
- Token-exact: `test_qwen27_paged_engine` 235/235 + `test_qwen36_paged_engine`
  315/315.
- compute-sanitizer memcheck 35B c2: 0 errors.
- Clean `-Werror` CUDA build (0 warnings), CUTLASS sm120a + FA2 sm_121a
  hard-verified; CPU ctest + tools unittest 164 + record/doc checkers green.

## Dependencies / risks

No product calls. Behaviour otherwise unchanged (bit-identical compute; only the
lifetime of already-baked scratch blocks changes). The retire list is
process-lifetime resident memory bounded by O(log) growths per allocator.
