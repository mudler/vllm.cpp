# ROCm: launch SharedK WMMA from the host (#785)

Row: `BACKEND-ROCM`. Issue:
[#785](https://github.com/mudler/vllm.cpp/issues/785).

## Defect

`PagedAttnPrefillSharedKWmma` launches were behind
`#if defined(VT_ROCWMMA_OK)`. That macro is defined only on HIP's
**device** pass (`__gfx1200__` / `__gfx1201__`). The host pass never
defines it, so both launch sites were deleted. Every d=256/d=512
prefill silently ran scalar `PagedAttnPrefillSharedK`.

## P0 (this head)

Repairs **d=256 production dispatch only**. d=512 remains scalar.

- Host launch uses `hipDeviceProp_t.gcnArchName` prefix-match
  `gfx1200` / `gfx1201` (not substring). `gfx1201:xnack-` matches;
  `foogfx1201` and `gfx12010` do not.
- Per-device decision cached once (`std::call_once`). Not per-build.
  Not getenv.
- Device kernel-body `#if !defined(VT_ROCWMMA_OK)` stub is unchanged.
- `VT_ATTN_PREFILL_SHAREDK_WMMA` still forces scalar when `=0`.
- Host path launches only `PagedAttnPrefillSharedKWmma<2,8,16,32,false>`
  (d=256). The d=512 WMMA launch/stub is removed, not hidden.
- d=512 keeps the existing scalar `PagedAttnPrefillSharedK` fallthrough.
  Shipping f58b d=512 WMMA is VGPR 192 / spill 52 / private 212 — that
  violates the 0/0 compile gate and is owed as a separate kernel repair.
- Shipping f58b HIP compile of `rocm_paged_attn.hip` (`clang++` roc-7.2.4
  `f58b06dce1f9`, `--offload-arch=gfx1201`). Device body unchanged.

Measured KD on that object (fields after `.name`):

| Instantiation | path | vgpr | spill | private | LDS |
|---|---|---|---|---|---|
| `<2,8,16,32,false>` | d=256 | 151 | **0** | **0** | 4880 |

P1 (this package; GPU HOLD):

Witness the product `vt::PagedAttention` seam, not a direct WMMA kernel call.

- Fixture: BF16 Q/K/V/out, d=256, qg=2 (hq=2,hk=1), T=64, one request,
  causal, sliding window left=32 right=0, scale=1/sqrt(256),
  seeds 78525601/02/03. Tensor SHA-256 frozen in
  `tests/vt/test_ops_paged_attn_sharedk_wmma_p1.cpp`.
- A/B same binary, separate processes (`VT_ATTN_PREFILL_SHAREDK_WMMA` is
  process-static). A = default/on, exact kernel
  `PagedAttnPrefillSharedKWmma<2,8,16,32,false>`. B = `=0`, that kernel
  absent, scalar `PagedAttnPrefillSharedK<2,8,...>` present.
- Both vs the same host f32 oracle. Preregistered BF16 bar:
  `abs(got-ref) <= 1.5e-2 + 1.0e-2*|ref|`, `corr>=0.999`, no nonfinite.
  Candidate-vs-scalar distance is reported; bit identity is not required.
- Fail closed on skip, missing trace, wrong kernel identity, nonfinite,
  oracle miss, or non-zero status. No silent retry. No timing. No d=512.
- Runner: `tests/scripts/run-785-p1.sh` (exit 78 without `VT_785_P1_GPU_GO=1`).
  Trace: `rocprofv3 --kernel-trace` parsed by `tests/scripts/parse_785_p1_trace.py`.
  Classifier separates family from exact specialization: A = exact
  WMMA `<2,8,16,32,false>` and no SharedK family and no other WMMA;
  B = exact scalar `<2,8,32,32>` and no WMMA family and no other scalar;
  wrong BM/BN, wrong qg/d, mixed, or none = UNKNOWN. GPU binary is an
  `add_executable` only (not ordinary CTest). Shared fixture header
  `tests/vt/sharedk_wmma_p1_fixture.h` is used by host and GPU; GPU prints
  frozen Q/K/V SHA-256 and the runner fails on mismatch. Arms pin
  `SHAREDK_WMMA=1|0`, SharedK=1, decode-opt=1, decode-GQA=1, CPU-ref=0.

## Owed

- Researcher review of this P1 package, then GPU GO/HOLD
- Separate kernel repair before any d=512 WMMA launch
- PR body (never-ran consequence + expected d=256 uplift)
