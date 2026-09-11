ID: ISSUE-GH-1590
Title: Qwen3.5/3.6 MoE GGUFs cannot run on discrete ROCm: kSharedExpertGate is CPU-only and the reference tier is ineligible on non-host-addressable memory
Row: BACKEND-ROCM
State: CLOSED
Kind: UNKNOWN
GitHub: 1590
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-21
Updated: 2026-08-24
Closed: 2026-08-24

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> A Qwen3.5/3.6 MoE GGUF loads on a discrete ROCm device and then dies on the first
> forward:
>
> ```
> engine-fatal: EngineCore busy loop threw: vt: no kernel for op SharedExpertGate
> (id 67) on device rocm (type 5), and the portable CPU reference tier is NOT
> eligible: this backend does not report its device memory host-addressable, so a
> host kernel may not dereference what it allocated (unified memory is false,
> which is a DIFFERENT property).
> ```
>
> Reproduced on gfx1200 / ROCm 7.2.3 with
> `Qwen3.6-14B-A3B-VibeForged-v2-Q4_K_M.gguf` (arch `qwen35moe`, 8.5 GB, fits the
> 15.9 GiB card), `--device auto`, and identically with `VT_GGUF_KEEP_QUANT=0`, so
> it is independent of the keep-quant residency route.
>
> ## Why it only bites ROCm
>
> `kSharedExpertGate` is registered on **`kCPU` and nowhere else**
> (`src/vt/cpu/cpu_ops.cpp:3732`). CUDA survives this because GB10 is unified and
> host-addressable, so the portable reference tier legitimately runs the CPU
> kernel over device pointers. A discrete AMD card is neither, and since
> `VT-REFTIER-HOST-ADDRESSABLE` the tier refuses by name instead of dereferencing
> what the host cannot address. That refusal is correct. The gap it exposes is
> that the ROCm MoE path was never covered by a native kernel.
>
> The only caller is `qwen3_5.cpp` (`:6356`, `:6902`), so this is the Qwen3.5/3.6
> MoE family specifically, not every MoE architecture.
>
> ## It is unlikely to be the only one
>
> `RegisterOp` counts today: **ROCm 45 ops, CPU 84, with 45 CPU-only.** Landing
> `kSharedExpertGate` alone may simply reveal the next one in the same forward.
> The full CPU-only set:
>
> ```
> kAttention kAttentionCross kAttentionDenseFa2 kAttentionDenseFast
> kAttentionDenseFlash kAttentionRelPos kAttnGateSplit kBatchedMatmul
> kCausalConv1dSpecUpdate kConv1d kConv2d kConv3d kConvTranspose1d
> kDepthwiseConv1d kDflash2SelectorEdges kDFlashBlockAttention kDFlashGroupedConv
> kDFlashPagedBlockAttention kFusedChain kGatherMlaCache kGdnConvSplit kGdnGBeta
> kGreedyRejectionSample kIndexCopy kIndexSelect kL2Norm kLtx2
> kMatmulFp8BlockScaled kMatmulFp8Cutlass kMatmulNvfp4Fp4 kMergeAttnStates
> kMiniMaxH3 kMlaPrefillAttention kMoeCombine kMoeGateUpSwiGLUGrouped kMoeRelu2
> kMulColVecF32 kQuantFp8Group kQuantFp8Static kRmsNormQuantFp8 kScaledFp4Quant
> kSharedExpertGate kSigmoidGateFp4Quant kSiluMulFp4Quant kTopKValuesIndices
> ```
>
> `kMoeCombine` and `kMoeGateUpSwiGLUGrouped` are in that list and are on the same
> MoE path, so scoping a fix should probably walk the whole Qwen3.5 MoE forward
> rather than chase one op at a time.
>
> ## How it was found
>
> Gating #1506 (the ROCm keep-quant GEMM). That row's `## Tests` step 6 is an
> MoE-model arm, to exercise the grouped `kMatmulBTQuantGrouped` op in a real
> checkpoint rather than only in a unit test. **That arm cannot run**, and this
> issue is why. Filed with an owning row rather than left as a silent hole in the
> other row's evidence.
>
> Also worth recording for whoever picks this up: #1506's justification for
> shipping the grouped op alongside the dense one said a dense-only registration
> would break "models that load today". On ROCm that is **overstated** and has
> been corrected there. Qwen3.5 MoE GGUFs do not run on ROCm today at all; they
> die here first. The grouped op is still required the moment this issue is
> fixed, which is why it shipped, but it protects a path that is currently
> unreachable rather than one that currently works.
>
> Owned by `BACKEND-ROCM`.
>

## Resolution

The binding comment https://github.com/mudler/vllm.cpp/issues/1590#issuecomment-5399705977 dated 2026-08-24 verifies the exact reproduction succeeds on `4b1154bc5` and identifies fix commit `1e97567ce` from pull request #509.
