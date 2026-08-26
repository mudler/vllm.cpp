# FIX-CUBLASLT-CAPTURE-1732 — cuBLASLt heuristic query inside decode graph capture

- Issue: [#1732](https://github.com/mudler/vllm.cpp/issues/1732)
- Base: `08c81a89218906cac08209a63c6301f03fdc8ec7`
- Pull request shape: one pull request for spec and implementation
  (recorded in `.agents/developer-preferences.md`, `## Git integration`).

## Scope

The default decode path dies on the first CUDA graph capture on CUDA 13.3 +
sm_80. `cublasLtMatmulAlgoGetHeuristic` returns `CUBLAS_STATUS_INTERNAL_ERROR`
(status 14) when it runs while a created stream is in capture. The bf16/f32
GEMM paths in `src/vt/cuda/cuda_matmul.cu` query the heuristic on every call,
so the first in-capture call throws.

Fix: cache the heuristic result per full call key, default on. The eager warm
step that precedes every capture populates the cache; the in-capture call hits
the cache and never queries cuBLASLt inside capture.

Out of scope:

- The fp8 heuristic paths. Their feature table cells resolve empty for sm_80,
  so they are unreachable on this board. See `## Owed`.
- Any change to algo selection policy. The cache returns the same algo the
  uncached path selects, because cuBLASLt selection is process-deterministic
  per shape.

## The defect, grounded

Board: CMP 170HX 64GB (GA100, sm_80, 70 SMs), driver 610.57.04, CUDA 13.3
toolkit V13.3.73. Build `-DVLLM_CPP_CUDA_ARCHITECTURES=80`, FA2 enabled,
1546/1546 targets, zero errors.

Default configuration (graphs on), `Qwen3.8-27B-UD-Q8_K_XL.gguf`:

```
engine-fatal: EngineCore busy loop threw: vt cuda: matmul: bt
cublasLtMatmulAlgoGetHeuristic: cublas status 14
(CUBLAS_STATUS_INTERNAL_ERROR)
```

`VLLM_CPP_CUDAGRAPH=0` on the identical workload generates the expected text,
24.3 tok/s warm, deterministic across three runs.

cuBLASLt's own trace names the internal cause (`CUBLASLT_LOG_LEVEL=4`):

```
[Error][cublasLtMatmulAlgoGetHeuristic] Could not obtain green context information
```

Minimal probe, no vllm.cpp code, CUDA 13.3 nvcc, `-arch=sm_80`, bf16 TN GEMV:

| Call site state | Result |
|---|---|
| Eager | SUCCESS |
| Capture active on a created stream | status 14 |
| Capture active on the legacy null stream | SUCCESS |
| After capture | SUCCESS |

Call sites that query the heuristic per call in the bf16/f32 lane:
`src/vt/cuda/cuda_matmul.cu:266` (row-major NN), `:346` (TN `bt`),
`:432` (strided-batched NN). The observed failure is `bt`; the other two are
the same defect class and can be the next in-capture call after `bt` is fixed.

Why the eager warm step guarantees a cache hit at capture: every decode graph
slot runs its forward eagerly first (`s.warm` state machine,
`src/vllm/model_executor/models/qwen3_5.cpp` at the `~GraphCaptureScope`
comment: an inert scope runs the region eagerly and the slot re-warms), and
the batch is padded before the slot key is chosen, so the warm step and the
capture step run the same GEMM shapes.

Upstream vLLM does not hit this because torch caches the selected cuBLASLt
algo per shape and vLLM runs eager warmup steps before capture.

## Design

New header `src/vt/cuda/gemm_plan_cache.h`, mirroring `fp8_plan_cache.h`:

- `GemmPlanKey`: device, op kind (NN / TN / batched), m, n, k, every leading
  dimension and batch stride passed to a layout create, batch count, ab type,
  out type. Equality and hash cover every field. A field that selects an algo
  but is absent from the key is the defect class the fp8 key documents
  (`out_type` is part of `Fp8PlanKey`'s `==` and its hash).
- `GemmPlanCacheEnabled()`: default ON, `VT_GEMM_PLAN_CACHE=0` disables.
  Unlike the fp8 cache, this one is not a performance knob: on CUDA 13.3 the
  uncached path is wrong under capture, so on is the only safe default. The
  variable exists as an escape hatch and for A/B measurement, not as a
  supported configuration.
- The plan map itself stays in `cuda_matmul.cu`
  (`GetOrQueryGemmHeuristic`, mutex-guarded, values leak by design) exactly as
  `GetOrBuildCachedFp8Plan` does. Only the key, hash, and flag live in the
  header, so the CPU test can reach them.

Cache value: the `cublasLtMatmulHeuristicResult_t` alone. Descriptor and
layout creation are host-side and legal under capture; only the heuristic
query reads green-context state and fails. The cached `workspaceSize` is
bounded by the same `kWorkspaceBytes` preference the query ran with.

### Alternatives rejected

- Flip `VT_FP8_PLAN_CACHE` default and reuse `Fp8PlanKey`. Wrong key: it
  carries fp8 fields (scale mode, pointer mode) and lacks ld/batch strides.
  Two caches with different keys on one file is clearer than one key that
  half-fits both.
- Query the heuristic before `BeginCapture` and thread the algo in. Needs
  shape knowledge at every capture site in eight model files; the cache gets
  the same result at the vt layer with no model change.
- Detect capture and refuse the query. A cold cache inside capture has no
  safe fallback; the warm step makes the miss unreachable, and the cache makes
  that true without a capture predicate.

## Tests

1. `tests/vt/test_gemm_plan_cache.cpp`, CPU-only, red first: key equality
   separates shape, dtype, ld, op, and batch; equal keys hash equal; the flag
   parses default-on and `0`-off. Mirrors `test_fp8_plan_cache.cpp`.
2. Board gate, this host, under the file mutex `${GPU_LOCK:-$HOME/gpu.lock}`
   (the card is not a fleet device): default configuration, graphs on, 24
   tokens, three runs, expected continuation, deterministic; token stream
   identical to a `VLLM_CPP_CUDAGRAPH=0` run.
3. Focused suites: the new test, `test_fp8_plan_cache`, and the existing
   matmul suites under `build-cuda`.

Reviewer mutation for reachability: delete the cache-hit return in the `bt`
path; the board gate must fail again with status 14.

## Gates

- `scripts/agent-preflight.sh` green before every push.
- `ctest` focused set green.
- Board gate green, recorded here with command lines and output.
- No public document owes a change: `docs/BUILD.md` is not a triggered
  surface, and the build-verified to run-verified wording for sm_80 rides in
  this row's `## Outcome`, not in a projection.

### Board gate record (2026-08-22)

Board: CMP 170HX 64GB (GA100, sm_80, 70 SMs), driver 610.57.04, CUDA 13.3
V13.3.73. Build dir `build-cuda-wt` with `-DVLLM_CPP_CUDA_ARCHITECTURES=80`.
Every GPU run held `flock ${GPU_LOCK:-$HOME/gpu.lock}`.

Command:

```sh
build-cuda-wt/examples/vllm-cli --model /home/models/Qwen3.8-27B-UD-Q8_K_XL.gguf --prompt 'The quick brown fox' --max-tokens 24
```

Three default (graphs-on) runs exit 0 with the identical 24-token output
(" jumps over the lazy dog." continuation), `tok_s` 1.018/1.022/1.022. One
`VLLM_CPP_CUDAGRAPH=0` run exits 0 with the identical token stream. One
`VT_GEMM_PLAN_CACHE=0` run (graphs on) exits 1 with:

```text
vt cuda: matmul: bt cublasLtMatmulAlgoGetHeuristic: cublas status 14
(CUBLAS_STATUS_INTERNAL_ERROR)
```

That disabled-cache run is the reachability proof: the cache is the fix
path. Reviewer mutations, tree restored byte-for-byte after each: M1 deleted
the `bt` cache-hit return and the board gate went red with status 14; M2
dropped `ld` from the key and the CPU test went red; M3 flipped the flag
default and the CPU test went red. Verdict: PASS at `06f905912`.

## Stop conditions

- If CUDA 13.3 also fails `cublasLtMatmul` itself inside capture, the cache is
  insufficient; stop and report the probe result on the issue.
- If the board becomes unavailable, the board gate stays PENDING and the row
  does not merge.

## Owed

- Extend the same mechanism to the fp8 heuristic call sites
  (`cuda_matmul.cu` fp8 GEMM, fp8 block dispatch) or flip
  `VT_FP8_PLAN_CACHE` default, measured on a Blackwell host with CUDA 13.x.
  Unreachable on sm_80 because those feature cells resolve empty; the latent
  risk is a future CUDA 13.x capture on sm_12x. Tracked here, not filed
  separately: this row's cache header is the vehicle.

## Now

- 2026-08-22: SPEC committed (`8dc827cbc`).
- 2026-08-22: IMPLEMENTED and reviewed PASS (`06f905912`); board gate green;
  awaiting merge.
