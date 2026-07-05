# CUTLASS sm120a NVFP4 GEMM drop-in — feasibility investigation

Grounded feasibility basis for reusing vLLM's cutlass block-scaled NVFP4 GEMM
(the near-peak kernel) behind a `vt::MatmulNvfp4Cutlass` op, to close the ~5×
residual speed gap on GB10 (sm_121). All vLLM cites are `file:line` relative to
the pinned checkout `/home/mudler/_git/vllm` @ `e24d1b24`. Build probe run on
`dgx.casa` (GB10, compute_cap 12.1, CUDA 13.0.88) 2026-07-05.

**Verdict: GO.** The minimal cutlass sm120a nvfp4 GEMM — both the dense and the
grouped/MoE variants — compiles AND runs AND passes on GB10 sm_121a with nvcc
13.0. This was the load-bearing unknown; it is resolved positively.

---

## 1. Where the kernel lives (cited)

vLLM keeps two csrc trees. The *live, torch-stable-ABI* NVFP4 kernels (the ones
built for the wheels) are under `csrc/libtorch_stable/quantization/fp4/`:

- **Dense**: `nvfp4_scaled_mm_sm120_kernels.cu` — defines
  `cutlass_scaled_fp4_mm_sm120a(D, A, B, A_sf, B_sf, alpha)`
  (`:230`). This is the sm120/sm121 dispatch target.
- **Entry / dispatch**: `nvfp4_scaled_mm_entry.cu:41` `cutlass_scaled_fp4_mm(...)`
  routes by SM: `sm>=120 && sm<130 → cutlass_scaled_fp4_mm_sm120a` (`:60-62`).
  **GB10 sm_121 falls in this window** — the sm120a kernel is the GB10 path.
  `cutlass_scaled_mm_supports_fp4()` (`:71`) reports support only when the
  SM-specific kernel was compiled in.
- **MoE grouped**: `nvfp4_blockwise_moe_kernel.cu` — `cutlass_fp4_group_mm(...)`
  (`:652`) → `run_fp4_blockwise_scaled_group_mm<OutType>` (`:589`) →
  **`run_fp4_blockwise_scaled_group_mm_sm120` (`:406`)** for `sm∈[120,130)`
  (`:613`). So there IS a cutlass grouped fp4 GEMM for MoE, sm120-native.
- (SM100 siblings: `nvfp4_scaled_mm_kernels.cu`; not needed on GB10.)

CUTLASS version: **v4.4.2**, header-only, FetchContent'd
(`CMakeLists.txt:430` `set(CUTLASS_REVISION "v4.4.2")`, `:445-455`,
`CUTLASS_ENABLE_HEADERS_ONLY ON` `:426`). Overridable via `VLLM_CUTLASS_SRC_DIR`.

### The dense collective it instantiates (`nvfp4_scaled_mm_sm120_kernels.cu:76-127`)
```
ElementA = ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>   // W4A4: fp4×fp4
LayoutA = RowMajor (align 32), LayoutB = ColumnMajor (align 32)
ElementSFA = ElementSFB = cutlass::float_ue4m3_t                    // ue4m3 group scales
ElementC = ElementD = OutType (bf16 or fp16), RowMajor
ElementAccumulator = float
ArchTag = cutlass::arch::Sm120
OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp
CollectiveBuilder mainloop + epilogue, GemmUniversal, GemmUniversalAdapter
Configs: default MmaTile 256×128×128 (PersistentScheduler); M≤256 → 128×128×128
Scales layout: Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA/SFB (swizzled)
alpha via epilogue.thread.alpha_ptr (per-tensor global-scale product)
```
Grouped/MoE variant (`:406-598`) is the same element/arch config with
`GroupProblemShape`, `GemmUniversalMode::kGrouped`, per-expert pointer arrays,
`MmaTile 128×128×128`, output hardcoded bf16 (`:422` "templating the output type
is not supported" on SM120).

### Host-side entry signature (dense, `:230`)
```cpp
void cutlass_scaled_fp4_mm_sm120a(
    torch::stable::Tensor& D,          // [M,N] out, bf16/fp16, RowMajor
    torch::stable::Tensor const& A,    // [M, K/2] packed fp4 (Byte=e2m1x2), RowMajor
    torch::stable::Tensor const& B,    // [N, K/2] packed fp4, ColumnMajor
    torch::stable::Tensor const& A_sf, // [round_up(M,128), round_up(K/16,4)] ue4m3, SWIZZLED
    torch::stable::Tensor const& B_sf, // [round_up(N,128), round_up(K/16,4)] ue4m3, SWIZZLED
    torch::stable::Tensor const& alpha // [1] float = 1/(gscaleA*gscaleB)
);
```
K is passed as `A.size(1)*2` (fp4 packed 2/byte). Alignment: k,n % 32 == 0.
Body: CHECK macros → shape/swizzle asserts → `cutlass_fp4_gemm_dispatch<bf16|half>`
→ `runGemm<Gemm>` = `args_from_options` + `get_workspace_size` +
`can_implement`/`initialize`/`run` (`:170-196`).

---

## 2. Dependency posture — torch-free-compilable? YES

**CUTLASS is pure header-only CUDA/C++ templates — no PyTorch, no ggml, no
runtime lib.** Adding it is consistent with the no-pytorch/no-ggml rule (same
category as the header-only cpp-httplib transport dep already vendored).

**Cost of adding cutlass v4.4.2:**
- Headers only: `include/` + `tools/util/include/`. Shallow clone = **202 MB**
  on disk (measured); vendor just `include/**` + a slice of `tools/util` and
  it shrinks. No compiled artifact, no link dep.
- Build time: **one dense TU ≈ 34 s** on the GB10 (measured, `-O2`,
  single arch sm_121a). Grouped TU similar. These are heavy template TUs;
  isolate them (own `.cu` files, own arch flag) so they don't inflate the
  general build. Binary-size add is the two GEMM kernels only.

**The torch coupling is TINY and mechanical.** The entire torch surface used by
the dense kernel (measured by grep over `nvfp4_scaled_mm_sm120_kernels.cu`):
`.size()` (×27, shape checks), `.data_ptr()` (×9), `.dim()` (×4),
`.scalar_type()` (×2), `.get_device_index()` (×2), `.is_cuda()`,
`.is_contiguous()`, `.device()`, `torch::stable::empty` (×1, workspace),
`DeviceGuard` (×1), `get_current_cuda_stream` (torch_utils.h). **The GEMM math
itself has ZERO torch** — it is 100% cutlass templates. `torch::stable::Tensor`
is already a thin ABI shim (not full ATen); every method above maps 1:1 onto our
`vt::Tensor` POD (`include/vt/tensor.h`: `void* data; DType dtype; Device device;
int64_t shape[4]/stride[4]`). No blocker to a pure-C++/CUDA build.

---

## 3. Build-probe result (LOAD-BEARING) — PASS

Cloned cutlass **v4.4.2** on GB10, compiled the two matching upstream examples
with **nvcc 13.0.88, `-std=c++17 -arch=sm_121a -O2 --expt-relaxed-constexpr`**:

| Example (matches vLLM's) | Compile | Run on GB10 | Result |
|---|---|---|---|
| `79b_blackwell_geforce_nvfp4_nvfp4_gemm.cu` (dense, = `cutlass_scaled_fp4_mm_sm120a`) | OK, 34 s | 1024³ | **Passed**, 103 TFLOPS |
| `79d_blackwell_geforce_nvfp4_grouped_gemm.cu` (grouped, = MoE `run_..._group_mm_sm120`) | OK | 10 groups | **Passed**, 125 TFLOPS |

Both are `ElementA=ElementB=nv_float4_t<float_e2m1_t>`, `ArchTag=Sm120`,
`OpClassBlockScaledTensorOp`, `ElementSF=float_ue4m3_t` — i.e. the *exact*
collective the vLLM kernels instantiate. **cutlass's sm120a block-scaled fp4
collective builds cleanly and runs correctly for `sm_121a` on CUDA 13** — the
`-arch=sm_121a` (family target) is accepted; no `sm_120a`-only blocker. This
directly de-risks the compile, which was the primary open question.

Environment note: `nvcc` lives at `/usr/local/cuda/bin` (not in login PATH);
GB10 disk is at 97% (126 G free) — vendor headers, don't leave full clones.

---

## 4. Effort estimate + scope

**Files (new):**
- `src/vt/cuda/cuda_matmul_nvfp4_cutlass.cu` — dense: lift
  `Fp4GemmSm120` + `args_from_options` + `runGemm` verbatim; replace the
  `torch::stable::Tensor` params with `vt::Tensor`, `.data_ptr()`→`.data`,
  `torch::stable::empty(workspace)`→ a `vt` device-buffer / cudaMalloc, stream
  from our backend. ~1 TU, own arch flag (already have `VT_FP4_MMA_SM120A`).
- `src/vt/cuda/cuda_moe_nvfp4_cutlass.cu` — grouped: lift
  `run_fp4_blockwise_scaled_group_mm_sm120` + `__get_group_gemm_starts`; same
  glue swap; feed our MoE per-expert layout (we already build expert offsets /
  problem sizes in `cuda_moe.cu`).
- `third_party/cutlass/include/**` (+ minimal `tools/util`) vendored headers,
  gated `.cu`-only include dir so it never touches the general build.
- Wire `vt::MatmulNvfp4Cutlass` (+ grouped) into `include/vt/ops.h` and route
  from the qwen3_5 dense/MoE forward behind a flag, replacing the current
  W4A16 `cuda_matmul_nvfp4.cu` path for the fp4 GEMMs.

**The real integration work (NOT the kernel):** our current fp4 path is **W4A16**
(bf16 activations × fp4 weights, `cuda_matmul_nvfp4.cu` — dequant-and-bf16-GEMM),
whereas the cutlass kernel is **W4A4** (fp4×fp4). To use it we must:
1. **Activation quant to fp4 + per-group ue4m3 scales at runtime** (per-token
   dynamic quant). We already have the CPU W4A4 activation-quant *reference*
   (`nvfp4_emulation.h`, unit-tested — see `.agents/qwen27b-w4a4-notes.md §3.4`);
   needs a GPU kernel.
2. **Scale swizzle**: cutlass wants `A_sf`/`B_sf` in the padded+swizzled atom
   layout (`round_up(M,128) × round_up(K/16,4)`, `Sm1xxBlkScaledConfig`). Our
   weight scales are stored **LINEAR on disk** (w4a4-notes §3, table). Need a
   swizzle for `B_sf` at load time and `A_sf` at runtime. This is the
   FlashInfer/cutlass block-scale "to_blocked" transform — a real but bounded
   piece (~one kernel + host layout math the kernel already exposes via
   `tile_atom_to_shape_SFA/SFB`).
3. `alpha = 1/(global_scale_A * global_scale_B)` per GEMM.

Dense projections (attention QKV/O, MLP up/gate/down): straightforward. MoE:
cutlass grouped kernel exists and passed the probe; our expert-offset plumbing
already exists, so mapping is tractable. **Estimate: kernel lift is small
(~2 TUs, ~1–2 days); the W4A4 activation-quant + scale-swizzle GPU kernels and
parity harness are the bulk (~3–5 days), most of the *math* already exists as
our tested CPU reference.**

---

## 5. Expected parity / gap-closure — GO recommendation

This IS the same kernel family that produced vLLM's near-peak numbers: vLLM
selects `CutlassNvFp4LinearKernel → cutlass_scaled_fp4_mm → sm120a` (grounded in
`qwen27b-w4a4-notes.md §7`), i.e. exactly `cutlass_scaled_fp4_mm_sm120a`, the
kernel probed above.

Gap arithmetic (honest Amdahl, our runtime): fp4 GEMMs ≈ 41% decode + 41%
prefill, currently ~15–16% of peak; cutlass ran ~near-peak on the probe (a
~5–6× per-op speedup). Bringing the fp4 portion to parity:
- Decode-only: `1 / (0.59 + 0.41/5.6) ≈ 1.5×` overall.
- Prefill+decode fp4 together (~82% of combined work): up to `≈ 2.2–2.4×`.

So dropping cutlass in for the fp4 GEMMs plausibly closes **~1.5–2.4× of the
5.18× residual** (35B), i.e. a large fraction but not all — the remainder is
non-fp4 (attention, GDN, RMSNorm, sampling, launch overhead) and must be chased
separately. The fp4 GEMM is nonetheless the single biggest lever and the kernel
is proven to build+run on our exact target.

**Recommendation: GO.** No compile blocker (proven), no torch/ggml dependency
blocker (cutlass is header-only templates, torch glue is ~50 lines of trivial
`vt::Tensor` mapping). The gating work is the W4A4 activation-quant + scale-
swizzle GPU kernels — already specced and CPU-referenced in the w4a4 notes — not
the cutlass kernel itself. Proceed with the dense path first (attention+MLP
projections), validate parity vs our W4A16 output, then wire the grouped MoE
kernel.
