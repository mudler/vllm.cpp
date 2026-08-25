#!/usr/bin/env python3
"""Validate the canonical .agents tables and lifecycle contracts."""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGENTS = ROOT / ".agents"

MATRICES = {
    # 358 since 2026-08-05: +31 architectures vLLM's registry defines that we had
    # NEVER inventoried (found by scripts/upstream-inventory.py). All INVENTORIED;
    # inventorying is not committing.
    # 360 since 2026-08-07: +`MODEL-AUDIO-PARAKEET-ENCODER`, the first row of the
    # new `MODEL-AUDIO` section (Parakeet / FastConformer encoder + CTC head,
    # spike parakeet-conformer-encoder.md work item P4). It is NOT one of the 328
    # registry architectures: vLLM ships it as the audio COMPONENT of
    # `nano_nemotron_vl.py` and delegates the encoder itself to transformers, so
    # there is no `registry.py` entry to inventory. A genuinely new row, never a
    # count relaxed to make a transition pass.
    # 361 since 2026-08-07: +`MODEL-AUDIO-PARAKEET-TRANSDUCER` (the Parakeet RNN-T
    # and TDT heads over that same encoder: `ParakeetForRNNT` / `ParakeetForTDT`,
    # spike work item P6). A SEPARATE row rather than an advance of the encoder
    # row, because it is a different upstream model class with its own state dict,
    # its own decode and its own checkpoints. Like the encoder row it is not one
    # of the 328 registry architectures: vLLM has no transducer call site at all
    #: so there is nothing in `registry.py` to inventory. Bumped because a new
    # row EXISTS, never to make a transition pass.
    # 362 since 2026-08-10: +`MODEL-MM-muse-glimmer-muse-glimmer-for-conditional-generation`
    # (Meta's Muse Glimmer 30B, released 2026-08-08). A THIRD beyond-pin row: it is
    # not one of the 326 registry architectures at `555967922`, and unlike the two
    # Parakeet rows it is absent because it did not exist yet, not because vLLM
    # delegates it. Its only upstream implementation is the still-OPEN
    # vllm#51655; see porting-inventory.md §9 deviation 16. Bumped because a new
    # row EXISTS, never to make a transition pass.
    # 369 since 2026-08-13: +7 rows for the architectures behind official
    # `vllm-project/recipes` models that had no row at all (#609, #610). One is
    # pin-lag — `BailingMoeV3ForCausalLM` is registered on vLLM `main` and
    # absent only at `555967922`. Six are out-of-repo: `MossTTSDelayModel`,
    # `MossTTSRealtime`, `Qwen3TTSForConditionalGeneration` and
    # `HiggsMultimodalQwen3ForConditionalGeneration` are registered by
    # `vllm-project/vllm-omni`, and `VoxtralRealtimeForConditionalGeneration`
    # and `BailingMMNativeForConditionalGeneration` are target-pending — their
    # exact `config.json` architecture strings are registered in neither core
    # vLLM `main` nor `vllm-omni`, so the rows record what was searched instead
    # of an invented anchor. SEVEN, not eight: the audit's eighth architecture
    # `Qwen3_5MoeForCausalLM` is rowed by #490 / PR #601, which registers it
    # rather than only inventorying it. Two branches ADDING the same keyed row
    # merge without a conflict and define it twice, so the row is left to its
    # owner. None of the seven touches the at-the-pin model inventory below
    # (324/373/356/310/261 is unchanged), because like the MuseGlimmer, KimiK3
    # and MiniMaxH3DiT rows they carry no pinned-registry target. Bumped
    # because seven new rows EXIST, never to make a transition pass.
    # 370 since 2026-08-13: +`MODEL-DIFFUSION-ltx-2-5-ltx2-video-transformer-3d-model`
    # (Lightricks LTX-2.5, 21.00B joint video+audio DiT, released 2026-08). A FOURTH
    # beyond-pin row, and like Muse Glimmer it is absent from `555967922` because it
    # did not exist yet. Unlike the others it is also out-of-repo: the architecture
    # reference is Lightricks' own `LTX-2` (`ltx-core`), and vLLM-Omni's `ltx2` module
    # stops at 2.3 (`ltx2_recipes.py:162-166`), with 2.5 still OPEN upstream at
    # vllm-omni#6066. Same lane as the MiniMax-H3 diffusion row. Bumped because a new
    # row EXISTS, never to make a transition pass.
    #
    # This entry READ `363 since 2026-08-11` until #651. Both halves were wrong,
    # and 363 is a value this pin has never held at any commit in its history —
    # so the entry described a transition that never happened, in a log whose
    # whole job is to say why each bump was legitimate. Re-derived from git
    # rather than carried forward, which is the only way any number in this
    # block is ever allowed to move: `git log -S` on the row id finds exactly
    # one commit, `cefacd2d0` (2026-08-13), and the pin reads 369 at
    # `cefacd2d0~1` and 370 at `cefacd2d0`. What makes that checkable rather
    # than plausible is the block itself — 358, 360, 361, 362, 369, 370, 372,
    # 373, 375, 377 is the sequence of values this pin has actually held, in the
    # order it held them, and an append-log that ran 369, 363, 372 was
    # self-evidently not a history. `test_model_pin_log_records_only_transitions_that_happened`
    # is what ties this entry to that sequence.
    # 372 since 2026-08-13: +2 for IndexTTS-2.5, which vLLM-Omni registers as TWO
    # architectures (`IndexTTS2TalkerForConditionalGeneration` stage 0 and
    # `IndexTTS2S2MelDecoder` stage 1), so a port described in prose as "a model"
    # moves this pin by two. Both land `INVENTORIED`, unclaimed and blocked on the
    # absent vllm-omni pin (#633). Bumped because two rows EXIST, never to make a
    # transition pass. #634.
    # 373 since 2026-08-13: +1 for `MiniMaxMusic3ForConditionalGeneration`, landing
    # `SPIKE` with its spec committed (#672). Two independent rows moved this pin on
    # the same day and BOTH branches read 371, so an auto-merge taking either side
    # would have left the matrix internally consistent while short a real
    # architecture. Re-derived, which is the only way this pin is ever allowed to
    # move. test_music3_and_indextts_rows_both_survive_their_collision names all
    # three rows, because a count alone cannot see that failure.
    # 375 since 2026-08-14: +`MODEL-TEXT-qwen3-5-qwen3-5-for-causal-lm` and
    # +`MODEL-TEXT-qwen3-5-qwen3-5-moe-for-causal-lm` (issue #490), the TEXT-ONLY
    # arms of the Qwen3.5 backbone — the eighth architecture the #609/#610 audit
    # found and deliberately left to its owner, plus its dense sibling. Both are
    # beyond-pin: they are not among the 355 registry architectures at
    # `555967922` because they landed upstream afterwards (PR vllm#50210 @
    # `ad5d29db7`), exactly like the Muse Glimmer row above. Their Upstream cells
    # deliberately carry no pinned module/class target, so the pin-derived static
    # invariants in check_model_invariants are UNCHANGED (324/373/356/310/261) —
    # this is the row-EXISTS count only, bumped because two new rows exist, never
    # to make a transition pass. This row was authored against 362 -> 364, then
    # re-derived to 370 -> 372, and is now RE-DERIVED AGAIN to 373 -> 375: the
    # #609/#610 backfill, LTX-2.5, IndexTTS-2.5 and MiniMax-Music3 all landed
    # while it was in review, and every one of them moved this pin. The number is
    # counted off the matrix as it stands after the merge, never carried forward
    # from the branch — a justification framed against a number this file no
    # longer carries would be false about the file it sits in, and
    # `Qwen35TextOnlyRowsAreCounted` is what ties this value to the two rows the
    # matrix actually holds.
    # 377 since 2026-08-14, and RE-DERIVED rather than carried forward: +2 for
    # dots3-note, which vLLM registers as TWO architectures
    # (`Dots3NoteForCausalLM` and its speculative head `Dots3NoteMTPModel`),
    # landing `SPIKE` and `INVENTORIED` respectively with the spec committed
    # (#699). This is the collision the Music3/IndexTTS comment above warns
    # about, happening again on the same day: the #490 branch took 373 -> 375
    # for the Qwen3.5 text-only arms while the dots3 branch took 373 -> 375 for
    # its own two rows. BOTH read 375 and neither was right -- the merged tree
    # holds four new rows, so it is 377. An auto-merge keeping either side would
    # have left this file internally consistent while silently short two real
    # architectures, which is why the number is counted off the matrix AFTER the
    # merge and why `test_dots3_rows_are_inside_the_model_ratchet` names the rows
    # instead of trusting the count. dots3-note is beyond-pin (vLLM `main` only,
    # vllm#51255, still being patched), carries no pinned-registry target, and
    # leaves the at-the-pin inventory (324/373/356/310/261) unchanged. Bumped
    # because two rows EXIST, never to make a transition pass.
    "MODEL": (AGENTS / "model-matrix.md", 377),
    # 82 since 2026-07-21: +`QUANT-NVFP4-CT-W4A16` (compressed-tensors NVFP4A16 /
    # W4A16 — NVFP4 weights with BF16 activations, distinct from the existing
    # `QUANT-NVFP4-CT-W4A4` and `QUANT-NVFP4-MO-W4A16` rows in both scheme
    # discovery and kernel selection). This count is the inventory size, so it is
    # bumped when a genuinely new scheme is inventoried — never to make a failing
    # state transition pass.
    # 2026-07-29: the pre-existing INVENTORIED `QUANT-GGUF-IQ3_XXS` (id 18) row was
    # ADVANCED to `ACTIVE` (keep-quant compute landed) by `CLAIM-DEEPSEEK-V4-W8` —
    # the `UD-IQ2_XXS` down-projection routed experts (`ffn_down_exps`) are IQ3_XXS;
    # no row count change (an in-place advance, not a new row).
    # 84 since 2026-08-18: +`QUANT-QWEN38-27B-GGUF-ARM` and
    # +`QUANT-QWEN38-27B-NVFP4-ARM`, the two quantized arms of Qwen3.8-27B whose bf16
    # arm is already gated (#915) and which #821 has owned with no row of its own.
    # Two rows and not one: they share nothing but a model name -- different file
    # format, different loader translation unit, different oracle (llama.cpp for the
    # GGUF arm, because vLLM has no in-tree GGUF at the pin and SGLang's alias table
    # does not reach `qwen3_5`; pinned vLLM for NVFP4, which it runs), different
    # external blockers (#857 vs #1185), and even different tokenizers on disk.
    # Merging them would let one external blocker hold the other's work. Neither is
    # expressible by the per-encoding rows in sections 1 and 2, which are keyed on the
    # encoding rather than on a checkpoint. Both `READY`, spec
    # `specs/qwen38-27b-quant-arms.md`, issue #821.
    "QUANT": (AGENTS / "quantization-matrix.md", 84),
    # 34 since 2026-07-22: +`KERNEL-GEMM-CPU-ELEM` (the elementwise f32/f16/bf16 CPU
    # GEMM — a genuinely separate family from `QUANT-GGUF-CIQ-GEMM`'s block-quantized
    # `kMatmulBTQuant`: it serves every safetensors CPU path and every non-block
    # tensor of a mixed GGUF). 33 since 2026-07-22: +`KERNEL-ACCEL-PROVIDER-SELECT`
    # (which IMPLEMENTATION of an op runs when a device has more than one — the
    # selection layer above every kernel family here, distinct from
    # `KERNEL-CUDA-DISPATCH-AOT`, which selects an ARCH for one implementation).
    # 35 since 2026-07-26: +`KERNEL-ATTN-DFLASH-BLOCK` (the DFlash draft's in-block
    # attention — the project's FIRST non-causal / bidirectional attention primitive,
    # a genuinely separate op from the causal `kAttention`/`kPagedAttention`; SPEC-DFLASH
    # D2, `CLAIM-DFLASH-D2`).
    # 36 since 2026-07-27: +`KERNEL-ATTN-DFLASH-PAGED-BLOCK` (the CAPTURE-SAFE paged
    # variant of the DFlash in-block attention — a genuinely separate op
    # `kDFlashPagedBlockAttention` reading the growing context from a paged K/V cache +
    # persistent device block_table instead of a materialized combined buffer, the
    # CUDA-graph draft-attention primitive; SPEC-DFLASH D12 Part B, `CLAIM-DFLASH-D12`).
    # 37 since 2026-07-28: +`KERNEL-ATTN-DENSE-FLASH` (the SHARED-MEMORY-TILED flash
    # variant of `AttentionDenseFast` for long non-causal contexts — a genuinely
    # separate op `kAttentionDenseFlash` that tiles K/V across a block of query-warps
    # in shared memory, for the Whisper AUDIO encoder; multimodal-speed §14,
    # `CLAIM-MM-SPEED-AUDIO-ENC-KERNEL`).
    # 38 since 2026-07-28: +`KERNEL-ATTN-DSA-SPARSE-INDEX` (the DeepSeek-V4 DSA
    # "Lightning Indexer" sparse-attention SELECTION op — weighted MQA logits with a
    # per-head ReLU + causal top-k token selection, the project's FIRST sparse-
    # attention candidate-selection primitive, distinct from every dense/paged/MLA
    # attention family above which score ALL keys. W3 landed a portable host
    # reference + unit gate; the device kernel is a W7 residual. `SPIKE`,
    # `CLAIM-DEEPSEEK-V4-W3`, spec specs/deepseek-v4-flash.md).
    # 39 since 2026-07-28: +`KERNEL-KDA-DELTA` (the Kimi Delta Attention gated-
    # linear-attention delta vs plain GDN — the per-channel [H,D] low-rank decay
    # (f_a/f_b bottleneck), the sigmoid-gated output norm (FusedRMSNormGated), the
    # 3 q/k/v short convs + q/k L2-norm. A genuinely new gated-linear-attention
    # family distinct from GDN, which has only a per-head scalar decay and no gated
    # output norm; SUBCLASSES GDN so its recurrence is REUSED, not re-ported. W1
    # landed a portable host reference + unit gate; the device kernel + the
    # Kimi-Linear-48B proxy e2e gate are named residuals. Shared unblocker for
    # Kimi-Linear-48B and Kimi-K3 (W4). `SPIKE`, `CLAIM-KDA-KERNEL`, spec
    # specs/kda-kernel-delta.md).
    # 40 since 2026-07-29: +`KERNEL-ATTN-DSA-COMPRESSOR` (the DeepSeek-V4 DSA
    # COMPRESSOR — the softmax-weighted window POOL that compresses
    # `(1+overlap)*compress_ratio` KV-state rows into one compressed latent (per
    # head-dim-column softmax, then RMSNorm) + the fused save-time APE add + the
    # fp8_ds_mla KV-cache STATE layout (448 fp8 UE8M0 per-64 block scales + 64
    # bf16 rope, 576B token stride, 7+1 scale bytes). A genuinely separate op from
    # `KERNEL-ATTN-DSA-SPARSE-INDEX` (which SELECTS keys): this one POOLS +
    # QUANTIZES the selected/windowed KV into the latent the MLA reads and how it
    # is cached across steps. W4 landed a portable host reference + unit gate; the
    # device kernel (the fused `_fused_kv_compress_norm_rope_insert_sparse_attn`)
    # is a W7 residual. `SPIKE`, `CLAIM-DEEPSEEK-V4-W4`, spec
    # specs/deepseek-v4-flash.md).
    # 41 since 2026-07-29: +`KERNEL-MHC-SINKHORN` (the DeepSeek-V4 Manifold/Markov
    # Hyper-Connections topology — the `[tokens, hc_mult, hidden]` residual
    # manifold mixed by a 20-iteration Sinkhorn-normalized doubly-stochastic
    # matrix, the mHC pre/post mixes with the FOLDED attn/ffn RMSNorms, and the
    # hc_head collapse). A genuinely new residual-stream topology distinct from the
    # plain residual+RMSNorm every other family uses. W5 landed a portable host
    # reference + unit gate (ported from the vLLM eager reference mhc/torch.py —
    # correcting the W0 "no eager reference" premise — and gated against a from-
    # first-principles double-precision Sinkhorn derivation); the device kernel +
    # DeepseekV4Model::Forward assembly are W7 residuals. `SPIKE`,
    # `CLAIM-DEEPSEEK-V4-W5`, spec specs/deepseek-v4-flash.md).
    # 42 since 2026-07-29: +`KERNEL-MOE-SQRTSOFTPLUS-HASH` (the DeepSeek-V4 MoE
    # router + clamped-SwiGLU deltas — the `sqrt(softplus(·))` score function, the
    # noaux_tc bias-for-selection top-k with weights gathered from the UNBIASED
    # scores, the hash `tid2eid` token-id→expert route that BYPASSES top-k, and the
    # asymmetric clamped SwiGLU expert activation `SiluAndMulWithClamp`). The three
    # genuinely-new-vs-V2/V3 MoE pieces; the shared grouped-GEMM / expert / shared-
    # expert machinery is REUSED, not re-ported. W6 landed a portable host reference
    # + unit gate; the device kernels + DeepseekV4Model::Forward assembly are W7
    # residuals. `SPIKE`, `CLAIM-DEEPSEEK-V4-W6`, spec specs/deepseek-v4-flash.md).
    # 43 since 2026-07-29: +`KERNEL-DSV4-W7-DEVICE` (the DeepSeek-V4-Flash W7-DEVICE
    # CUDA kernels — the four NEW V4 op families' device kernels: MHC Sinkhorn+pre/
    # post+head, DSA indexer weight-fold/MQA-logits/causal-topk + sink softmax +
    # grouped output-LoRA, compressor pool+norm + fp8_ds_mla KV encode/decode, and
    # the sqrtsoftplus/hash router + clamped SwiGLU. Each a 1:1 device port of the
    # landed host reference (KERNEL-{MHC-SINKHORN,ATTN-DSA-SPARSE-INDEX,ATTN-DSA-
    # COMPRESSOR,MOE-SQRTSOFTPLUS-HASH}) registered through the OpProvider seam and
    # RUNTIME-VERIFIED on the DGX GB10 at small shape vs its host-ref oracle (11/11
    # cases · 153 assertions, compute-sanitizer memcheck 0 errors, RED-first proven).
    # The 512-wide MLA attn + expert grouped-GEMM REUSE the existing NVFP4/FP8
    # kernels — NOT re-ported. `IMPL`, `CLAIM-DEEPSEEK-V4-W7-DEVICE`, spec
    # specs/deepseek-v4-flash.md; real-checkpoint e2e stays W8, multi-Spark).
    # 44 since 2026-07-29: +`KERNEL-QUANT-CIQ-IQUANT` (the DeepSeek-V4 W8 keep-quant
    # `vec_dot` for the ~2-3-bit codebook encodings IQ2_XXS/IQ3_XXS/Q2_K — the
    # single-Spark GGUF memory enabler; extends `QUANT-GGUF-COMPUTE`'s six-type
    # `kMatmulBTQuant` so the 158 B routed experts stay COMPRESSED instead of
    # OOM-expanding to bf16. CPU tier only, like the six existing k-quants. 1:1
    # ports of ggml `vec_dot_q2_K/iq2_xxs/iq3_xxs_q8_K_generic`; gated 19 cases /
    # 130444 assertions, RED-first proven. `SPIKE`, `CLAIM-DEEPSEEK-V4-W8`, spec
    # specs/deepseek-v4-flash.md §W8).
    # 45 since 2026-07-29: +`KERNEL-QUANT-CIQ-GEMM-CUDA` (the FIRST CUDA keep-quant
    # GGUF k-quant GEMM — the kCUDA provider for `kMatmulBTQuant`. MMVQ-style
    # dequant-in-kernel dot: quantize the activation to Q8_K on-GPU, integer-dot
    # against the compressed Q8_K-family weight blocks (IQ2_XXS/IQ3_XXS/Q2_K +
    # Q3_K/Q4_K/Q5_K/Q6_K) kept COMPRESSED in the unified pool. Registering it flips
    # GgufQuantComputeAvailable TRUE on kCUDA so DeepSeek-V4's routed experts run on
    # the GPU instead of the 20 ARM cores. A NEW kernel family — the CPU
    # KERNEL-QUANT-CIQ-IQUANT vec_dot is a separate row/impl. RUNTIME-VERIFIED on
    # the DGX GB10: 2/2 cases · 92401 assertions vs the CPU oracle + f64 dequant,
    # memcheck 0, RED-first proven. `ACTIVE`, `CLAIM-CUDA-KEEPQUANT-GEMM`, spec
    # specs/deepseek-v4-flash.md §W8.)
    # 46 since 2026-08-06: +`KERNEL-CPU-A76-Q8-DOT`, a separately gateable
    # Cortex-A76 Q8_0 x Q8_0 DotProd/assembly family. The physical-Pi trace
    # proves the portable dot is reached at 20.10% of Qwen3.5-2B user cycles;
    # the row owns exact-order C++ SDOT vs scheduled AAPCS64, independent of
    # the broad CPU-backend row.
    # Inventory size, bumped for a genuinely new family — never to make a failing
    # state transition pass.
    # 47 since 2026-08-06: +`KERNEL-GEMM-CPU-ELEM-X86WIDE` (the AVX2/AVX-512
    # elementwise tiers; our x86 tier is SSE2 while the box has avx512f, a
    # measured 3.5x) and +`KERNEL-GEMM-CPU-TILED` (the tinyBLAS-style tiled
    # sgemm; controls proved our NEON kernel is at ggml-stock parity and the
    # whole 16-bit deficit is llamafile, ~1.9x Arm / ~2.4x x86).
    # 50 since 2026-08-06: +`KERNEL-CPU-CONV2D-SUBSAMPLE`, +`KERNEL-DEPTHWISE-CONV1D`
    # and +`KERNEL-ATTN-RELPOS` — the three conformer/FastConformer audio-encoder
    # primitives the tree had no device op for at all (Conv2d existed only as a host
    # std::vector loop; the only depthwise conv1d was the CAUSAL Mamba/GDN one; every
    # attention path was RoPE + paged/flash KV). Spike specs/parakeet-conformer-encoder.md.
    # 51 since 2026-08-06 (PR #79): +`KERNEL-CPU-A76-Q8-DOT`, a separately gateable
    # Cortex-A76 Q8_0 x Q8_0 DotProd/assembly family. The physical-Pi trace
    # proves the portable dot is reached at 20.10% of Qwen3.5-2B user cycles;
    # the row owns exact-order C++ SDOT vs scheduled AAPCS64, independent of
    # the broad CPU-backend row.
    # 52 since 2026-08-18 (#1171): +`KERNEL-GDN-REPLAYSSM`, the ReplaySSM buffered
    # output-only GDN decode. A genuinely new family, not a variant of the packed
    # decode row: it changes WHEN the state is written (every L steps, from a ring
    # of rank-1 factors) rather than how one step is tiled, and it adds three cache
    # tensors to the MambaSpec. vLLM ships the algorithm for Mamba2 only and cannot
    # reach GDN (four walls, spec §Upstream chain); SGLang ships the GDN arm.
    # 53 since 2026-08-19 (#1314): +`KERNEL-DFLASH2-GROUPED-CONV`, the DFlash2
    # draft's grouped DYNAMIC depthwise convolution. A genuinely new family and
    # not a variant of `KERNEL-DEPTHWISE-CONV1D`, on all three axes that decide
    # a kernel's shape: the weights are DYNAMIC (a per-position delta projected
    # from the sublayer input, added to a static per-channel base) rather than
    # static, they are GROUPED (one delta per group of channels against one base
    # per channel) rather than per-channel, and the tap mask is over the QUERY
    # BLOCK (`i mod (1+k)`) rather than causal over the sequence. It also carries
    # a SIDE axis no other convolution here has: one projection of the sublayer
    # input produces both the prepare-side and the finish-side coefficients.
    # Bumped because the row EXISTS, never to make a state transition pass; the
    # row is `ACTIVE` rather than `DONE` because its CUDA arm has never compiled
    # (spec `## Owed` O6, no `nvcc` on the authoring host).
    # 54 since 2026-08-20 (#1007): +`KERNEL-CONV3D`, the general 3-D convolution
    # `vt` had on NO device. It is not a variant of `KERNEL-CPU-CONV2D-SUBSAMPLE`:
    # the ACCUMULATION ORDER differs and is part of the contract (one f32 partial
    # per input channel with the bias seeded first, against kConv2d's single flat
    # accumulator with the bias last), which is the same sibling relationship
    # kConv1d has with kDepthwiseConv1d. It is also the only conv family with a
    # CUDA arm and a CPU arm landing together, and the reason the LTX-2.5 video
    # VAE decode had no device path at all. Spec specs/ltx25-device-residency.md.
    # 56 since 2026-08-20 (#1314): +`KERNEL-DFLASH2-SELECTOR-EDGES` and
    # +`KERNEL-TOPK-PAIRS`, the DFlash2 candidate selector's two kernels. TWO
    # rows and not one because they are two kernels with different shapes and
    # different gates: the first is a small dense contraction over two
    # per-token codebooks, whose difficulty is the PREDECESSOR indexing (step 0
    # is the verified anchor, every later step is the previous step's candidate)
    # and the bf16 ROUNDING PLACEMENT; the second is a sort-free selection over a
    # 248320 vocabulary, whose difficulty is the TIE-BREAK, because the
    # pivot-bracket search converges to an exact array VALUE and therefore keeps
    # whole tie groups. `KERNEL-TOPK-PAIRS` is also a distinct family from the
    # shipped sampling threshold search rather than a variant of it: that kernel
    # masks below the k-th largest IN PLACE and returns no indices, this one
    # compacts the survivors, orders them and emits (id, value) pairs. Bumped
    # because the rows EXIST, never to make a state transition pass; both are
    # `ACTIVE` rather than `DONE` because neither CUDA arm has ever compiled
    # (spec `## Owed` O10, no `nvcc` on the authoring host).
    # 57 since 2026-08-20 (#1314): +`KERNEL-DFLASH2-PATH-WALK`, the DFlash2
    # candidate selector's PATH WALK. A separate family from
    # `KERNEL-DFLASH2-SELECTOR-EDGES` rather than a second entry point into it,
    # on the axis that decides kernel families here: the lattice op is a dense
    # CONTRACTION whose difficulty is a reduction (and which is therefore gated
    # within an f32 envelope), while the walk performs no arithmetic at all --
    # only comparisons and one gather -- and is specified BIT-EXACT across
    # backends. Their grids follow from that: one block per (request, step,
    # predecessor slot) against one block per REQUEST with the step loop INSIDE
    # it, which is spec `## Risks/decisions` D3's requirement and upstream's own
    # `(num_reqs,)` / `num_warps=1` shape. Bumped because the row EXISTS, never
    # to make a state transition pass; it is `ACTIVE` rather than `DONE` because
    # its CUDA arm has never compiled on the authoring host (spec `## Owed` O11).
    "KERNEL": (AGENTS / "kernel-matrix.md", 57),
    # 56 since 2026-07-22: +`BACKEND-ACCEL-PROVIDER` (the acceleration-provider seam
    # itself, which is a cross-backend platform concern rather than a platform).
    # 57 since 2026-07-22: +`BACKEND-SEAM-AUDIT` (the accelerator-seam AUDIT — does
    # our MLX/Vulkan/Metal architecture actually port vLLM's CUDA-path strategy, and
    # what is the shared layer's device-leakage budget. A cross-backend structural
    # concern that owns no kernel and no platform, hence its own row).
    # 60 since 2026-07-27: +`BACKEND-CUDA-SM060` + `BACKEND-CUDA-SM061` (the Pascal
    # targets for the beyond-vLLM fp16/non-tensor-core breadth lane) + the
    # `BACKEND-GATE-CUDA-LLAMACPP-LEGACY` competitor floor (llama.cpp on the same
    # old card, since vLLM has no entry on Pascal/Volta/Turing). See
    # specs/cuda-arch-breadth-fp16.md.
    # 65 since 2026-07-28: +the distributed / scale-out family
    # `BACKEND-DISTRIBUTED-COMM`/`-TP`/`-PP`/`-MULTINODE-SPARK`/`-MLX-RING` (the NEW
    # scale-out capability dimension — one `vt::` collective abstraction with
    # nccl/RDMA/MLX-ring transports; single-GPU today). All `SPIKE`,
    # `CLAIM-SCALE-OUT-SPIKE`, spec specs/scale-out-distributed.md.
    # 68 since 2026-07-28: +the parallelism-MODE rows `BACKEND-DISTRIBUTED-DP`/
    # `-EP`/`-SP` (the enumeration of vLLM's parallelism modes onto our transport
    # abstraction — DP engine-replica scale-out, EP whole-expert all-to-all, SP the
    # TP-mode reduce-scatter compilation pass; TP/PP already existed, context/CP
    # rides the same abstraction and takes no row). All `SPIKE`,
    # `CLAIM-PARALLELISM-MODES-SPIKE`, spec specs/parallelism-modes.md.
    # 79 since 2026-08-05: +11 BACKEND-GGML-* rows, the llama.cpp ggml
    # backends folded into scope (user-directed). All INVENTORIED and
    # spike-gated; inventorying is not committing.
    # 80 since 2026-08-09: +`BACKEND-TENSTORRENT`, an extension platform
    # proposal (Tenstorrent Blackhole, ttnn C++ adapter) in the same class as
    # Metal/Vulkan. INVENTORIED; spec-only, not yet reviewed or accepted.
    # 81 since 2026-08-11: +`BACKEND-TENSTORRENT-RESIDUAL-GOLDEN`, the child
    # row owing residual-RMS numerics evidence at the device boundary (rows>=32
    # bf16 device path vs CPU f32 oracle). Bot-flagged on #289; READY once the
    # RED-first probe lands.
    # 82 since 2026-08-11: +`BACKEND-TENSTORRENT-MISTRAL`, allowlist
    # MistralForCausalLM on TT + device-aware SACRED gate. Reuses Qwen3-dense
    # forward; no new kernel. Pending 7B checkpoint + vLLM oracle for the e2e
    # gate.
     # 83 since 2026-08-16: +`BACKEND-GATE-CUDA-LLAMACPP` (#979), the llama.cpp
     # floor on a CURRENT CUDA card. Neither existing llama.cpp row covers it:
     # `BACKEND-GATE-CPU-LLAMACPP` is the CPU floor and
     # `BACKEND-GATE-CUDA-LLAMACPP-LEGACY` is scoped to Pascal/Volta/Turing,
     # where vLLM has no entry at all. The four-way Qwen3.8-27B campaign needs
     # it because llama.cpp is the ONLY comparator that runs the Q4_K_M arm:
     # vLLM removed GGUF from its tree at our pin. INVENTORIED, no run.
     # 84 since 2026-08-12: +`BACKEND-TENSTORRENT-TRACE-RUNNER`, feasibility
     # spike for wiring the landed #354 graph-capture foundation into a
     # capturable forward region (decode host-free region? capture tok/s cost?
     # ttnn program-cache warm-up?). No code; decision record only.
     # 85 since 2026-08-13: +`BACKEND-TENSTORRENT-HOST-FREE-FORWARD`, the plan
     # row decomposing the host-free decode forward (R1 RmsNorm+RoPE, R2
     # QkvSplit+RAC, R3 PA decode, R4 capture wire) that the trace-runner
     # spike revealed as the real prerequisite for decode capture.
     # 86 since 2026-08-22: +`BACKEND-TENSTORRENT-GDN` (#1715), the GDN
     # linear-attention op chain as native TT kernels — the hard prerequisite
     # for every Qwen3.5/3.8 arch on Tenstorrent. ACTIVE, spec-first; no
     # implementation yet.
    "BACKEND": (AGENTS / "backend-matrix.md", 86),
}

ENGINE_MATRIX = AGENTS / "engine-matrix.md"
ENGINE_PREFIXES = (
    "ENG",
    "KV",
    "PAR",
    "SAMPLE",
    "TOOLS",
    "SPEC",
    "SERVE",
    "LORA",
    "ATTN",
    "LOAD",
)
# 117 since 2026-07-25: +`ENG-MM-AUDIO-ENCODER` (the Whisper-class AUDIO encoder
# tower, audio-track A2 — the encoder half of audio understanding, proven faithful
# in isolation; a genuinely-new engine capability, distinct from the A0/A1
# `ENG-MM-AUDIO-PIPELINE` INPUT row). Bumped for a real new row, never to make a
# failing state transition pass.
# 120 since 2026-07-27: +`KV-SGLANG-RADIX-CACHE` (SGLang RadixAttention
# behavior-parity scope — verdict: already fused into our block-hash APC, the
# `--enable-radix-attention` flag is an alias) and +`ENG-SGLANG-BEHAVIOR-FLAG`
# (the SGLang-alike runtime survey + the enable/disable control: cache-aware LPM
# scheduling is the one genuinely-distinct flag-worthy behavior; overlap ==
# `ENG-ASYNC-SCHED`, jump-forward deferred). Both `SPIKE`,
# `CLAIM-SGLANG-RADIX-SCOPE`, spec `specs/sglang-radixattention.md`.
# 121 since 2026-07-28: +`SAMPLE-N` (parallel sampling — a request with n>1 fans
# out into n prompt-sharing child sequences aggregated into one RequestOutput /
# n OpenAI choices; the n==1 default path stays byte-identical). A real new engine
# capability, `ACTIVE`, `CLAIM-C7-N-SAMPLING`.
# 122 since 2026-07-28: +`SAMPLE-BEST-OF` (the OpenAI `best_of` endpoint control —
# generate best_of children via the SAMPLE-N fan-out, rank by cumulative logprob,
# return the top-n; a distinct request-surface capability from raw n-sampling). The
# best_of==n default path is byte-identical. `ACTIVE`, `CLAIM-C7-BESTOF-BEAM-API`.
# 123 since 2026-07-28: +`SPEC-MTP-GGUF` (MTP speculative decoding from a GGUF
# TARGET). Distinct from `SPEC-MTP` (`DONE`, safetensors): the engine currently
# REFUSES mtp+GGUF on the assumption that GGUF exports carry no head, but
# llama.cpp's Qwen3.5 converter does emit it and `HfConfigFromGguf` already reads
# the announcing metadata key. `READY`, spiked.
# 124 since 2026-07-28: +`SPEC-DFLASH-GGUF` (DFlash draft, then target, from GGUF).
# Distinct again: a separate draft checkpoint with its own llama.cpp `dflash` arch
# contract, plus a shared-head coupling to the target that `SPEC-MTP-GGUF` does not
# have. `READY`, spiked.
# 125 since 2026-07-28: +`ENG-POOLER-SEQ` (the non-generative POOLER OP — the
# sequence pooling methods CLS/LAST/MEAN + the normalize/classify activation heads
# that turn hidden states into a pooled embedding/logit row instead of a sampled
# token). HIGH-priority feature-gap #2 (pooling task class); a genuinely-new
# engine capability, W1 CPU brick landed + unit-gated. `ACTIVE`, `CLAIM-POOLING`,
# spec `specs/pooling-task-class.md`. Bumped for a real new row, never to make a
# failing state transition pass. (`SERVE-POOLING-ENDPOINTS` INVENTORIED→SPIKE in
# the same change is a state move on the existing row, not a new row.)
# 126 since 2026-07-28: +`KV-PREFIX-MATCH-UNIT` (`--prefix-match-unit` / config
# `prefix_match_unit`, NEW in vLLM 0.26). A distinct capability: the fine-grained
# prefix-cache matching unit (`resolve_kv_cache_block_sizes` -> `hash_block_size`),
# not covered by the `KV-PREFIX-CACHE` block-hash row (which hashes at
# `block_size`). W0 spike + W1 resolver landed; `PARTIAL`.
# 127 since 2026-07-29: +`ENG-PLUGIN-SYSTEM` (the out-of-core PLUGIN system —
# `LoadGeneralPlugins()` + the general-plugin registration seam that lets an
# external TU / shared object register a model factory / platform / quant method
# through the existing `REGISTER_VLLM_MODEL`-style seams WITHOUT editing engine
# core; the RECORDS-GAP the feature-gap analysis named for row creation). A real
# new engine capability, W0 spike + W1 CPU brick landed + unit-gated RED-first;
# `ACTIVE`, `CLAIM-PLUGIN-SYSTEM`, spec `specs/plugin-system.md`. Bumped for a
# real new row, never to make a failing state transition pass.
# 128 since 2026-07-29: +`SERVE-BATCH-API` (the offline OpenAI Batch API runner —
# read a JSONL of BatchRequestInput, dispatch each line to the matching serving
# handler, write a BatchRequestOutput JSONL; the RECORDS-GAP the feature-gap
# analysis named for row creation, recommending SERVE-BATCH-API). A real new
# serving capability, W0 spike + W1 CPU brick (chat dispatch orchestrator over
# the existing OpenAIServingChat, NO reimplemented generation) landed +
# unit-gated RED-first; `ACTIVE`, `CLAIM-BATCH-API`, spec `specs/batch-api.md`.
# Bumped for a real new row, never to make a failing state transition pass.
# 130 since 2026-07-29: +`SPEC-DRAFT-MODEL` (the classic model-agnostic SEPARATE
# draft-model speculator — a full smaller standalone LM runs K autoregressive
# greedy steps to propose K drafts, target verifies in one forward, longest
# accepted prefix emitted; distinct from MTP/EAGLE/DFlash: no target-hidden tap,
# no shared embed/lm_head) and +`SPEC-MEDUSA` (Medusa N-head single-pass
# speculator). The two RECORDS-GAP items the feature-gap analysis named for row
# creation (lines 82-83). `SPEC-DRAFT-MODEL` ACTIVE (W0 spike + W1 CPU greedy
# propose brick landed + unit-gated RED-first, reusing the landed SPEC-REJECTION
# verify); `SPEC-MEDUSA` SPIKE (W0 spike only, proposer deferred to W2).
# `CLAIM-SPEC-DRAFT-MEDUSA`, spec `specs/draft-model-medusa-spec.md`. Bumped for
# two real new rows, never to make a failing state transition pass.
# 131 since 2026-07-29: +`ENG-POOLING-RUNNER` (the pooling RUNNER path — where the
# generation runner samples a token, `PoolingRunner` applies the model's `Pooler`
# to the last hidden state and returns the POOLED DATA / embedding vector; the W3
# brick of the pooling task class). A real new engine capability, CPU brick landed
# + structurally cosine-gated RED-first; `ACTIVE`, `CLAIM-POOLING`, spec
# `specs/pooling-task-class.md`. (Its sibling `ENG-POOLER-SEQ` advanced W1→W2 in
# the same change — the heads/`SequencePooler`/`DispatchPooler` composite — not a
# new row.) Bumped for a real new row, never to make a failing state transition
# pass.
# 141 since 2026-08-06: +`SERVE-VIDEOS-OAI` (the `/v1/videos` request surface in
# OpenAI's Sora shape plus `GET /v1/videos/{id}/content`) — a real new serving
# capability, not a restatement of the MiniMax-H3 model row: an OpenAI video
# client works unmodified, the MP4 is fetchable over HTTP at all, and the
# fl2va/ref2va exclusivity is enforced at the request boundary. CPU-landed +
# gated, `PARTIAL`, `CLAIM-SERVE-VIDEOS-OAI`, spec `specs/minimax-h3.md` §9.
# Bumped for a real new row, never to make a failing state transition pass.
# 142 since 2026-08-06: +`SERVE-VIDEOS-REFS` (reference CONDITIONING over
# `/v1/videos`: `input_reference` -> fl2va, plus the two `metadata` ref2va
# modalities) — a real new capability stacked on `SERVE-VIDEOS-OAI`, not a
# restatement of it: that row made an OpenAI body PARSE, this one makes its
# references reach the pipeline. Before it no reference modality was reachable
# over HTTP at all. CPU-landed + gated, `CLAIM-SERVE-VIDEOS-REFS`, spec
# `specs/minimax-h3.md` §10.
# Bumped for a real new row, never to make a failing state transition pass.
# 143 since 2026-08-07: +`ENG-RELEASE-BINARIES` (downloadable, backend-specific
# server bundles and their static/runtime dependency contract) — a real
# distribution capability requested in issue #117, not a restatement of the
# server implementation. Inventoried while its release-matrix spike is written;
# no packaging support is claimed by the count bump.
# Bumped for a real new row, never to make a failing state transition pass.
# 144 since 2026-08-08: +`ENG-RELEASE-CONTAINERS` (published GHCR container
# images built by GitHub Actions — a distribution channel distinct from the
# downloadable archives in `ENG-RELEASE-BINARIES`: different artifact format,
# registry, tag contract, multi-arch manifest and publish flow, sharing only the
# staged bundle. User-directed, issue #170; inventoried while its spike is
# written, and no image, workflow or registry package is claimed by the bump.
# Bumped for a real new row, never to make a failing state transition pass.
# 145 since 2026-08-09: +`ENG-DOCS-SITE` (publish `docs/` as a GitHub Pages site
# that mounts the existing markdown read-only rather than copying it — a real
# distribution surface for the documentation, distinct from the binary and
# container channels above and from the docs themselves, which it does not
# modify). User-directed, issue #224; `READY` on its committed spec, and no
# site, workflow or published page is claimed by the bump.
# 147 since 2026-08-10: +`KV-MOONCAKE-STORE` (`MooncakeStoreConnector`, Mooncake's
# distributed KV object store as an external cache pool). Split out of
# `KV-CONNECTORS`, whose blanket "Mooncake NOT SCHEDULED" verdict conflated the
# P2P `MooncakeConnector` with the store connector; the P2P half keeps that
# verdict. User-directed, issue #287; `SPIKE` on its committed spec. No client,
# no connector, no build flag and no gate result is claimed by the bump.
# 148 since 2026-08-11: +`ENG-RECORD-CONFLICT-SURFACES` (retire the shared record
# surfaces that make concurrent PRs conflict by construction — the `STATUS_RATCHET`
# global, the `NOW.md` byte budget, and the insert-at-one-anchor claims table).
# MEASURED at `origin/main` `d928e2c3`: 16 of 29 open PRs conflict and 13 of those
# 16 conflict in bookkeeping only. User-directed, issue #364; `READY` on its
# committed spec. No checker semantic, no doc content and no gate result is
# changed by the bump — this row is the record of the work, not the work.
# 149 since 2026-08-11: +`ENG-NOW-DERIVED` (the live position is DERIVED and the
# freshness obligation moves to the row's own spec, so `.agents/NOW.md` stops
# being a surface every row-advancing PR must write). Follow-up to #364, which
# removed the file's byte budget but not the doc-checkpoint requirement that
# marched every PR into it. User-directed, issue #374; `ACTIVE` on its committed
# spec. No checker semantic beyond the row's own scope and no product source is
# changed by the bump.
# 150 since 2026-08-11: +`ENG-TRAILER-MERGE-ARTIFACTS` (the trailer gate rejects
# correct commits because GitHub appends `Co-authored-by:` as a separate
# paragraph, which hides the block from `git interpret-trailers --parse`; 13 of
# the last 30 commits on main failed the check, unnoticed because those runs were
# cancelled). User-directed, issue #406; `ACTIVE` on its committed spec. No rule
# in that checker is relaxed and no product source changes.
# 151 since 2026-08-11: +`ENG-FORGE-COAUTHOR` (the forbidden-AI-trailer rule was
# catching GitHub's auto-generated `Co-authored-by`, which attributes the ACCOUNT
# that opened the PR rather than claiming a model wrote the code; most PRs here
# are bot-opened, so nearly every squash red main). Developer-approved,
# issue #418; `ACTIVE` on its committed spec. Sign-off keeps its rule with no
# exemption and no product source changes.
# 152 since 2026-08-11: +`ENG-RELEASE-WINDOWS` (native Windows x86_64 CPU and
# Vulkan pre-alpha release extension). User-directed, issue #117; `INVENTORIED`
# while its committed specification awaits implementation and hosted evidence.
# No build, artifact, runtime evidence, workflow, or publication is claimed by
# this row-count bump.
# 153 since 2026-08-13: +`SERVE-RECIPE-ARGS` (accepted-and-inert serve arguments).
# `vllm-serve` aborts on any unrecognized flag, so `--enable-auto-tool-choice`
# (89 of 157 official vLLM recipes) and `--trust-remote-code` (82 of 157) stop the
# server before model load even though neither means anything to this engine —
# including for models we ship token-exact and gated. Found by the 2026-08-13
# recipe-surface sweep, issue #606. The row was `SPIKE` when this bump was first
# written against the spec-only commit; it lands `ACTIVE`, because the squash that
# carries this line also carries the seam (`kAcceptedInertArgs` in
# `server_main.cpp`), its test (`test_serve_recipe_args.cpp`) and the
# `docs/USAGE.md` entry. Stated as of THIS tree rather than as of the spec commit:
# `main` is squash-only, so a justification framed at an intermediate commit would
# ship as a comment that is false about the file it sits in.
# Bumped for a real new row, never to make a failing state transition pass.
# 154 since 2026-08-13: +`ENG-UPSTREAM-OMNI-PIN` (a parity pin for the separate
# `vllm-project/vllm-omni` repository). A genuinely-new protocol capability, not a
# restatement of the vLLM pin: it is a SECOND pin that may legitimately disagree
# with the first, because vllm-omni requires vLLM 0.27.0+ against our 0.26.0.dev0
# core pin. Landed the same day as the 153 bump above and merged against it: both
# rows are real and neither replaces the other, which is why this line reads 154
# rather than restating 153. `READY`, spec `specs/upstream-omni-pin.md`, issue #633.
# Bumped for a real new row, never to make a failing state transition pass.
# 155 since 2026-08-14: +`ENG-HYBRID-PLACEMENT` (per-tensor-group device placement,
# delivering routed-MoE expert COMPUTE on the CPU backend while the rest of the
# model stays on GPU). Genuinely new, not a restatement of either offload row it
# sits beside: `ENG-WEIGHT-OFFLOAD` and `ENG-EXPERT-STREAM` both move weights
# toward the compute, and this row moves compute toward the weights, so no
# existing row can express it. Surpass-track — vLLM ships CPU MoE kernels but
# selects them platform-wide via `current_platform.is_cpu()`, so hybrid placement
# is absent at the pin and the gate runs against llama.cpp `237ad9b96`.
# `READY`, spec `specs/hybrid-placement.md`, issue #149.
# 156 since 2026-08-14: +`ENG-RECORD-ANCHOR-RATCHET` (the record's own `path:line`
# citations were checked for RANGE but never for CONTENT, and a failing check was
# silently DROPPED. `local_line_anchors` in THIS file parses both citation forms
# -- markdown links, and (since ee511ca8a) bare `file.cpp:123` under the
# `RAW_LOCAL_ANCHOR_RE` prefixes -- but on a missing file or an out-of-range line
# it `continue`s, so the bad anchor is omitted from the list and swallowed by
# `is_code_anchor`'s `any()`. There was no symbol test and no report. 32 of the
# 38 offenders this row banks are IN RANGE, so range-checking alone could never
# have found them. Found by three stale anchors that humans caught by reading
# during the 2026-08-13/14 campaign, all of them IN RANGE. Issue #632.
# It lands `ACTIVE`, not `SPIKE`: the same change carries the parser, the
# STALE/BROKEN classifier, `scripts/record-anchor-baseline.json` and the
# `RecordAnchorRatchet` suite, so a comment framed at the spec-only commit would
# be false about the file it sits in. The COUNT is unchanged by that -- the row
# already existed at 156 and this is not a bump.
# 157 since 2026-08-17: +`ENG-RESIDENCY-CONFIG` (the host-RAM->DISK weight-residency
# tier as a CONFIG surface -- a `vllm_cpp` extension key inside the existing
# `--offload-config` document, reaching the loader through
# `EngineParams::weight_residency`). Genuinely new, and not expressible by either
# offload row it sits beside: `ENG-WEIGHT-OFFLOAD` owns the MIRRORED device->host
# tier and may not grow a disk arm without breaking a 1:1 transcription of
# `vllm/config/offload.py`, and `ENG-EXPERT-STREAM` owns the streaming MECHANISM
# rather than its configuration -- this row changes where a value comes from and
# nothing about what it does. `ACTIVE`, spec `specs/weight-residency-config.md`,
# issue #1110 (also fixes #1109 in flow).
# 158 since 2026-08-18: +`SPEC-DSPARK-QWEN3-ROUTING` (the DSpark draft-ARCHITECTURE
# route: `architectures=["DSparkDraftModel"]` + `model_type` `qwen3` must resolve to
# the landed Qwen3 DSpark lane, and `IsDsparkDraft` must be reached from the loader).
# Genuinely new and not expressible by `SPEC-DSPARK` beside it: that row owns the
# DSpark MECHANISM -- the Markov head, the block draft, the sequential sample -- and
# its W1-W8 all landed, while this row changes only which lane a draft config is
# classified into before any of that runs. BEYOND-PIN on vLLM PR 52197 (merged
# 2026-08-17 at `7075ddac`); the pinned behavior at `speculative.py:934-944` was never
# ported here, so this row records a divergence that already exists rather than
# introducing one. `READY`, spec `specs/dspark-qwen3-routing.md`, issue #1193.
# 164 since 2026-08-18: +`LOAD-GGUF-MMPROJ` (a SECOND, `clip`-architecture GGUF
# projector file beside the language file, and the Qwen3-VL vision tower loaded out
# of it). Genuinely new and not expressible by `LOAD-GGUF` beside it: that row owns
# the reader, the dequantization and the Qwen name transforms for ONE file, and the
# single-file assumption it was built on is structural rather than incidental --
# `ModelSource` carries a vector of safetensors shards and exactly one `GgufFile*`
# (`model_registry.h:98`), and `EngineParams` has no projector field, so an mmproj
# has nowhere to arrive. Sharding is already handled and is not this: `DetectSplit`
# merges shards of ONE split, never a second, differently-architected file. Nothing
# in the tree loads a `clip` projector today; MuseGlimmer's mmproj path is a refusal
# whose only caller is a test, and that refusal becomes reachable production code the
# moment this lands. `READY`, spec `specs/qwen38-27b-quant-arms.md`, issue #821.
# 165 since 2026-08-19: +`SPEC-DFLASH2` (the `DFlash2DraftModel` architecture: a
# grouped dynamic depthwise convolution inside each draft block, and a candidate
# selector that replaces the per-slot argmax with a scored path walk over the target
# head's top-K). Genuinely new and not expressible by `SPEC-DFLASH` beside it: that
# row owns the DFlash mechanism and reached `DONE`, and upstream itself carries
# DFlash2 as a SECOND architecture rather than as a change to the first -- DFlash1
# gains two subclass seams and keeps every behaviour, so a `DFlashDraftModel`
# checkpoint resolves exactly as it does today. BEYOND-PIN on vLLM PR 52816 (OPEN at
# head `19c93519`, base `9842d701`); the parity pin `555967922` does not carry the
# architecture at all, so this row does not advance the pin. `READY`, spec
# `specs/dflash2-spec-decode.md`, issue #1314.
# 163 since 2026-08-18: +`ENG-EXPERT-STREAM-DEVICE` (the DESTINATION half of expert
# streaming -- where a streamed slice lives and which platform may read it). Genuinely
# new and not expressible by `ENG-EXPERT-STREAM` beside it: that row owns the streaming
# MECHANISM -- the slot cache, the streamer, the `pread` filler and the host store --
# and all of it landed and runs on `--device cpu`, while this row changes only the
# destination those bytes are written to and the predicate that decides who may read
# them. The two have different hardware requirements and different lifecycle states:
# the parent is `READY` with a live CPU lane, and this one cannot reach its own
# discrete-GPU gate on any box the project owns. Surpass-track, no oracle: inference-time
# disk expert paging is absent in pinned vLLM (`offloader/uva.py:21`,
# `offloader/prefetch.py:557-560`) and no secondary oracle implements it either.
# `READY`, spec `specs/expert-stream-device-slots.md`, issue #1124.
# 167: `ENG-HF-MODEL-DOWNLOAD`. `--model` takes a local path only, so no shipped
# container image and no release archive can obtain a checkpoint: the runtime
# stage carries no Python and no `curl`, while `docker/Dockerfile:188-192`
# already sets `HF_HOME=/cache` and declares the `/cache` volume for a fetch
# that does not exist. The row is not a duplicate of `LOAD-SAFETENSORS` or
# `LOAD-GGUF`, which both start from bytes already on disk, and it is not
# `LOAD-CONFIG-SURFACE`, which parses a flag it never resolves. The one adjacent
# implementation, `model_loader.cpp:279-303`, reads an existing cache for the
# DFlash draft alone and never downloads. `READY`, spec
# `specs/hf-model-download.md`, issue #1280.
# 168 since 2026-08-19: +`SPEC-BPE-QUADRATIC-MERGE` (the BPE merge loop is O(n^2)
# in pretoken length, on the request path, before `ValidatePromptLen`). Genuinely
# new and not expressible by the two tokenizer rows beside it: `LOAD-HF-BPE` and
# `LOAD-SENTENCEPIECE` both own a FORMAT -- which `tokenizer.json` shapes parse and
# which token identifiers come out -- and both are token-exact against HF goldens
# today and stay that way. This row changes no identifier at all. It replaces the
# algorithm underneath both of them, and its gate is a COST bound, which is the one
# thing a token gate provably cannot see. It is also not a benchmark row: the encode
# runs synchronously on the HTTP worker five lines before the only length check, so
# `max_model_len` bounds none of it and `/tokenize` reaches it with no engine.
# MEASURED, and stated as the two SESSION-INVARIANT quantities only: over 1 KB to
# 64 KB of ordinary English prose the fit through the committed Mistral golden has
# exponent 2.01, and at 64 KB our cost is 2,507x HF `tokenizers` 0.22.2's on the
# same file for byte-identical identifiers. Both are ratios taken inside one
# session, so contention cancels. The ABSOLUTE milliseconds are deliberately not
# repeated here: they moved 54% between two runs of one binary on one input, so a
# constant copied into this comment would be a fourth place for a number nobody
# can reproduce to drift. They live in the spec's tables, each beside its own load
# average, and `## Gates` owes the idle-host re-measure. The exponent, not the
# constant, is what makes this a row.
# `READY`, spec `specs/bpe-quadratic-merge.md`, issue #1365.
# 169 since 2026-08-21: +`SPEC-DRAFTER-CHAIN` (a preference-ordered chain of
# speculators: try the first, and if it yields no draft for a sequence, try the
# next). Genuinely new and not expressible by the per-method rows beside it: each
# of `SPEC-MTP`, `SPEC-DFLASH`, `SPEC-DSPARK` and `SPEC-NGRAM` owns ONE
# speculator's mechanism, and every one of them assumes it is the only speculator
# resolved for a step. This row owns the composition -- a new optional field on
# `--speculative-config` that is inert when absent, per-sequence resolution, and
# the per-drafter attribution none of those rows has any reason to carry. vLLM
# implements no composition at all (`SpeculativeMethod` is a single `Literal` at
# the pin AND at `origin/main` `c20572610`), so this is a DIVERGENCE with
# llama.cpp as a secondary oracle for semantics only, not a port. `READY`, spec
# `specs/drafter-chain.md`, issue #1522.
# Bumped for a real new row, never to make a failing state transition pass.
# 170 since 2026-08-22: +`SERVE-REQUEST-LENGTH-GUARD` (the REFUSING byte bound at
# the request boundary, #1541). A genuinely-new serving capability rather than a
# state move: `67823aee2` removed the quadratic term from the BPE merge loop and
# added no bound, and nothing between an unauthenticated body and the tokenizer
# limited its size except httplib's 100 MB default. `GATING`, spec
# `specs/serve-request-length-guard.md`. Bumped for a real new row, never to make
# a failing state transition pass.
# 171 since 2026-08-24: +`ENG-UPSTREAM-LTX2-PIN` (pin `Lightricks/LTX-2`, the
# LTX-2.5 lane's actual reference and a third repository, #1433). A genuinely-new
# row rather than a state move: the oracle registry named nine upstreams and this
# lane's own was not one of them, so `check-oracle-pins.py` reported
# `oracle-pins ok (9 oracles pinned)` while the repository every LTX anchor
# resolves in had no file, no table row and no pin. `READY`, spec
# `specs/oracle-ltx-2-pin.md`. Bumped for a real new row, never to make a failing
# state transition pass.
ENGINE_ROWS = 171

ENGINE_SUMMARY_SECTIONS = (
    ("Engine and scheduling", "Engine core and scheduling"),
    ("KV cache and memory", "KV cache and memory"),
    ("Parallelism", "Parallelism and scale-out"),
    ("Sampling and generation", "Sampling and generation controls"),
    ("Structured output and tools", "Structured outputs and tool calling"),
    ("Speculative decoding", "Speculative decoding"),
    ("Serving, API, CLI, library", "Serving surface, CLI, and library"),
    ("LoRA and adapters", "LoRA and adapters"),
    ("Long context and attention", "Long context and attention breadth"),
    ("Loading, tokenizer, config", "Loading, tokenizer, and config"),
)

MATRIX_PATHS = [ENGINE_MATRIX, *(path for path, _ in MATRICES.values())]
REQUIRED = [
    ROOT / "AGENTS.md",
    ROOT / "README.md",
    ROOT / "docs/BENCHMARKS.md",
    AGENTS / "roadmap_v1.md",
    AGENTS / "issue-index.md",
    AGENTS / "coordination.md",
    AGENTS / "feature-matrix.md",
    AGENTS / "specs/model-family-inventory.md",
    AGENTS / "specs/feature-anchor-backfill.md",
    *MATRIX_PATHS,
]

STATES = {
    "INVENTORIED",
    "SPIKE",
    "READY",
    "ACTIVE",
    "GATING",
    "PARTIAL",
    "DONE",
    "BLOCKED",
    "ANCHOR-BACKFILL",
    "BUILD-ONLY",
    "UNTRACED",
}
READY_STATES = {"READY", "ACTIVE", "GATING", "DONE", "BLOCKED"}
EVIDENCED_STATES = {
    "PARTIAL",
    "ANCHOR-BACKFILL",
    "GATING",
    "DONE",
    "BUILD-ONLY",
    "UNTRACED",
}

ALL_PREFIXES = (*ENGINE_PREFIXES, *MATRICES)
ID_RE = re.compile(
    rf"(?:{'|'.join(re.escape(prefix) for prefix in ALL_PREFIXES)})-"
    r"[A-Za-z0-9_.-]+"
)
STATE_RE = re.compile(r"`(" + "|".join(re.escape(state) for state in STATES) + r")`")
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
# Group 1 is the run of fence characters, group 2 everything after it, which is
# the INFO STRING on an opening fence and must be empty on a closing one. Both
# groups are load-bearing: see strip_code_spans for the pairing rule and for the
# live file that mis-paired without it.
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})(.*)$")
INLINE_CODE_RE = re.compile(r"`+[^`\n]*`+")
CLAIM_RE = re.compile(r"CLAIM-[A-Za-z0-9_.-]+")
LINE_FRAGMENT_RE = re.compile(r"L(\d+)(?:-L?(\d+))?")
COMMIT_RE = re.compile(r"[0-9a-f]{7,40}")
RAW_LOCAL_ANCHOR_RE = re.compile(
    r"(?<![A-Za-z0-9_./-])"
    r"((?:src|include|tests|examples|cmake|scripts|tools|\.github/workflows)/"
    r"[A-Za-z0-9_./-]+|CMakeLists\.txt):(\d+)(?:-(\d+))?"
)

CODE_ANCHOR_PREFIXES = (
    "src/",
    "include/",
    "examples/",
    "cmake/",
    "scripts/",
    "tools/",
    ".github/workflows/",
)
TEST_ANCHOR_PREFIXES = (
    "tests/",
    "scripts/",
    ".github/workflows/",
)
CODE_ANCHOR_FILES = {"CMakeLists.txt"}
EVIDENCE_ANCHOR_FILES = {
    ".agents/parity-ledger.md",
}

SPEC_REQUIREMENTS = {
    "Scope": ("scope",),
    "Upstream chain": ("upstream chain",),
    "Our baseline": ("our baseline",),
    "Port map": ("port map", "port and harness map"),
    "Tests to port": ("tests to port",),
    "Gates": ("gates",),
    "Dependencies": ("dependencies",),
    "Work breakdown": (
        "work breakdown",
        "non overlapping work breakdown",
        "work decomposition",
    ),
    "Risks/decisions": ("risks", "risk /", "risk and", "risks/decisions"),
}


@dataclass(frozen=True)
class ClaimRow:
    path: Path
    line_no: int
    item_id: str
    state: str
    header: tuple[str, ...]
    cells: tuple[str, ...]
    raw: str

    def field(self, name: str) -> str:
        index = field_index(self.header, name)
        return self.cells[index] if index is not None and index < len(self.cells) else ""


def markdown_files() -> list[Path]:
    return [
        ROOT / "AGENTS.md",
        ROOT / "README.md",
        ROOT / "docs/BENCHMARKS.md",
        *sorted(AGENTS.rglob("*.md")),
    ]


def split_cells(line: str) -> list[str]:
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|"):
        body = body[:-1]
    return [cell.strip() for cell in re.split(r"(?<!\\)\|", body)]


def normalize_header(value: str) -> str:
    value = value.replace("`", "").replace("*", "").lower()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value).split())


def field_index(header: tuple[str, ...], field: str) -> int | None:
    for index, value in enumerate(header):
        if field == "id" and value == "id":
            return index
        if field == "item" and (
            value.startswith("item")
            or value == "encoding"
            or "method scheme" in value
            or "cache dtype mode" in value
            or value == "scope"
            or value == "surface"
        ):
            return index
        if field == "upstream" and "upstream" in value:
            return index
        if field == "code" and ("our code" in value or "local evidence" in value):
            return index
        if field == "tests" and (
            "tests evidence" in value
            or "test evidence" in value
            or "local evidence" in value
            or value == "evidence"
        ):
            return index
        if field == "spec" and "spike" in value:
            return index
        if field == "state" and (value == "state" or value.endswith(" state")):
            return index
        if field == "owner" and "owner" in value:
            return index
    return None


def is_separator(cells: list[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def parse_claim_rows(path: Path, errors: list[str]) -> list[ClaimRow]:
    rows: list[ClaimRow] = []
    header: tuple[str, ...] | None = None
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.startswith("|"):
            header = None
            continue
        cells = split_cells(line)
        normalized = tuple(normalize_header(cell) for cell in cells)
        if "id" in normalized and any(value == "state" or value.endswith(" state") for value in normalized):
            header = normalized
            continue
        if is_separator(cells):
            continue
        if header is None or not cells:
            continue

        item_id = cells[0].strip().strip("`")
        if not ID_RE.fullmatch(item_id):
            continue
        if len(cells) != len(header):
            errors.append(
                f"{path.relative_to(ROOT)}:{line_no}: {item_id} has {len(cells)} cells; "
                f"header has {len(header)}"
            )
            continue
        state_index = field_index(header, "state")
        state_cell = cells[state_index] if state_index is not None else ""
        state_matches = STATE_RE.findall(state_cell)
        if len(state_matches) != 1:
            errors.append(
                f"{path.relative_to(ROOT)}:{line_no}: {item_id} must have exactly one canonical state"
            )
            continue
        rows.append(
            ClaimRow(path, line_no, item_id, state_matches[0], header, tuple(cells), line)
        )
    return rows


def strip_code_spans(text: str) -> str:
    """Blank out fenced blocks and inline code, preserving line and column count.

    A target inside a code span is NOT a link: CommonMark renders it as literal
    text, so no reader can follow it and there is nothing for "every link
    resolves" to be about. Before 2026-08-12 this checker validated them anyway
    (#460), which meant no document in the tree could SHOW a link in sample
    output, and, worse, that a docs/BENCHMARKS.md row quoting its evidence link
    could not be archived into .agents/ byte-for-byte.

    THE PAIRING RULE IS COMMONMARK'S, not "the next line that looks like a
    fence". A closing fence must use the OPENER'S character, be at least as
    long, and carry nothing but whitespace after the marker; a line with an info
    string opens a block and never closes one. Getting that wrong does not fail
    safe, it INVERTS fence phase for the rest of the file: with the one
    unbalanced fence this tree already has, a bare ``` at
    .agents/completed/state-events/0000-00/STATE-LEGACY-000001.md:17697 was
    "closed" by the ```sh at :17948, and ordinary prose at :18297 was blanked,
    so a live reader-followable link stopped being validated. Measured
    tree-wide, the loose rule dropped 5 targets and the CommonMark rule drops 4.

    Blanking rather than deleting preserves every line and column offset. Note
    that check_links reports no line numbers today, so this buys nothing yet; it
    is kept so that a caller that does report positions cannot be broken by this
    function, and test_stripping_preserves_line_and_column_positions holds it.
    """
    out: list[str] = []
    fence: str | None = None
    fence_len = 0
    for line in text.splitlines():
        marker = FENCE_RE.match(line)
        if fence is None:
            # CommonMark: a backtick opening fence's info string may not contain
            # a backtick, which is what keeps `` `a` and `b` `` from opening one.
            if marker is not None and not (
                marker.group(1)[0] == "`" and "`" in marker.group(2)
            ):
                fence = marker.group(1)[0]
                fence_len = len(marker.group(1))
                out.append(" " * len(line))
                continue
            out.append(INLINE_CODE_RE.sub(lambda m: " " * len(m.group(0)), line))
            continue
        if (
            marker is not None
            and marker.group(1)[0] == fence
            and len(marker.group(1)) >= fence_len
            and not marker.group(2).strip()
        ):
            fence = None
        out.append(" " * len(line))
    return "\n".join(out)


def extract_links(text: str) -> list[str]:
    """Return every link target a READER could follow in this document."""
    return LINK_RE.findall(strip_code_spans(text))


def link_bases(source: Path, text: str) -> tuple[Path, ...]:
    """Return every directory a relative link in this file may resolve from.

    Normally exactly one, the file's own directory. Two files are archives that
    hold content written somewhere else and moved here verbatim, so a target in
    them was authored against the ORIGINAL directory: migrated legacy
    state-event payloads came from .agents/, and .agents/benchmark-record.md is
    the declared archive of docs/BENCHMARKS.md (#460).

    BE PRECISE ABOUT WHAT THE SECOND BASE BUYS. It does NOT make the archived
    copy clickable: a reader opening .agents/benchmark-record.md on GitHub and
    clicking a docs/-relative target such as USAGE.md, BUILD.md or
    bench-evidence/... gets a 404, because the browser resolves it against
    .agents/. What it enforces is that the EVIDENCE STILL EXISTS in the tree
    under one of the two declared bases, so archiving a row byte-for-byte cannot
    silently orphan the file it points at. That is the property the archive
    exists to give, and it is weaker than followability. Making the archived
    copy followable means rewriting the target or recording its origin, which is
    spec W5 and is deliberately not done here: rewriting a target would break
    the byte-for-byte guarantee, and W5 records the origin instead.
    """
    if (
        source.is_relative_to(AGENTS / "completed/state-events")
        and "<!-- legacy-payload:begin -->" in text
    ):
        return (AGENTS,)
    if source == AGENTS / "benchmark-record.md":
        return (source.parent, ROOT / "docs")
    return (source.parent,)


def check_links(errors: list[str]) -> None:
    for source in markdown_files():
        text = source.read_text(encoding="utf-8")
        bases = link_bases(source, text)
        for raw_target in extract_links(text):
            target = raw_target.strip().strip("<>")
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            target_path, _, fragment = target.partition("#")
            candidates = [(base / target_path).resolve() for base in bases]
            resolved = next((c for c in candidates if c.exists()), None)
            if resolved is None:
                errors.append(f"{source.relative_to(ROOT)}: dangling link {raw_target}")
                continue
            line_match = LINE_FRAGMENT_RE.fullmatch(fragment)
            if line_match is None or not resolved.is_file():
                continue
            line_count = len(resolved.read_text(encoding="utf-8", errors="replace").splitlines())
            start = int(line_match.group(1))
            end = int(line_match.group(2) or start)
            if start < 1 or end < start or end > line_count:
                errors.append(
                    f"{source.relative_to(ROOT)}: out-of-range line anchor {raw_target} "
                    f"(file has {line_count} lines)"
                )


def table_ids(prefix: str, path: Path, errors: list[str]) -> list[str]:
    return [
        row.item_id
        for row in parse_claim_rows(path, errors)
        if row.item_id.startswith(prefix + "-")
    ]


def check_matrices(errors: list[str]) -> tuple[list[ClaimRow], dict[str, ClaimRow]]:
    rows: list[ClaimRow] = []
    for path in MATRIX_PATHS:
        rows.extend(parse_claim_rows(path, errors))

    by_id: dict[str, ClaimRow] = {}
    for row in rows:
        prior = by_id.get(row.item_id)
        if prior is not None:
            errors.append(
                f"{row.path.relative_to(ROOT)}:{row.line_no}: duplicate ID {row.item_id}; "
                f"first at {prior.path.relative_to(ROOT)}:{prior.line_no}"
            )
        else:
            by_id[row.item_id] = row

    for prefix, (path, expected) in MATRICES.items():
        count = sum(row.item_id.startswith(prefix + "-") for row in rows if row.path == path)
        if count != expected:
            errors.append(f"{path.relative_to(ROOT)}: {count} {prefix} rows; expected {expected}")

    engine_count = sum(
        any(row.item_id.startswith(prefix + "-") for prefix in ENGINE_PREFIXES)
        for row in rows
        if row.path == ENGINE_MATRIX
    )
    if engine_count != ENGINE_ROWS:
        errors.append(
            f"{ENGINE_MATRIX.relative_to(ROOT)}: {engine_count} engine rows; expected {ENGINE_ROWS}"
        )
    return rows, by_id


def check_engine_summary(rows: list[ClaimRow], errors: list[str]) -> None:
    lines = ENGINE_MATRIX.read_text(encoding="utf-8").splitlines()
    header: list[str] | None = None
    total: list[str] | None = None
    summaries: dict[str, list[str]] = {}
    section_lines: dict[str, int] = {}
    for line_no, line in enumerate(lines, 1):
        if line.startswith("## "):
            section_lines[line.removeprefix("## ").strip()] = line_no
        if line.startswith("| Area | Rows |"):
            header = [normalize_header(cell) for cell in split_cells(line)]
        elif header is not None and total is None and line.startswith("|"):
            cells = [cell.replace("*", "").strip() for cell in split_cells(line)]
            if is_separator(cells):
                continue
            if cells[0] == "Total":
                total = cells
            else:
                summaries[cells[0]] = cells
    if header is None or total is None or len(header) != len(total):
        errors.append(f"{ENGINE_MATRIX.relative_to(ROOT)}: missing or malformed lifecycle summary")
        return

    actual_rows = [row for row in rows if row.path == ENGINE_MATRIX]

    def check_counts(label: str, recorded_cells: list[str], scoped_rows: list[ClaimRow]) -> None:
        if len(recorded_cells) != len(header):
            errors.append(f"{ENGINE_MATRIX.relative_to(ROOT)}: malformed {label} lifecycle summary")
            return
        expected = {"rows": len(scoped_rows)}
        expected.update(
            {normalize_header(state): sum(row.state == state for row in scoped_rows) for state in STATES}
        )
        for index, name in enumerate(header[1:], 1):
            if name not in expected:
                continue
            try:
                recorded = int(recorded_cells[index])
            except ValueError:
                errors.append(
                    f"{ENGINE_MATRIX.relative_to(ROOT)}: non-numeric {label} summary for {name}"
                )
                continue
            if recorded != expected[name]:
                errors.append(
                    f"{ENGINE_MATRIX.relative_to(ROOT)}: {label} summary {name}={recorded}; "
                    f"actual {expected[name]}"
                )

    check_counts("total", total, actual_rows)
    for area, section in ENGINE_SUMMARY_SECTIONS:
        recorded_cells = summaries.get(area)
        section_line = section_lines.get(section)
        if recorded_cells is None or section_line is None:
            errors.append(f"{ENGINE_MATRIX.relative_to(ROOT)}: missing {area} lifecycle summary")
            continue
        next_section_line = min(
            (line_no for line_no in section_lines.values() if line_no > section_line),
            default=len(lines) + 1,
        )
        scoped_rows = [
            row for row in actual_rows if section_line < row.line_no < next_section_line
        ]
        check_counts(area, recorded_cells, scoped_rows)


def is_placeholder(value: str) -> bool:
    normalized = value.strip().strip("`").lower()
    return not normalized or normalized in {"-", "none", "unassigned", "open", "leaf open"}


def local_line_anchors(value: str, source: Path) -> list[str]:
    anchors: list[str] = []
    for raw_target in LINK_RE.findall(value):
        target = raw_target.strip().strip("<>")
        if not target or target.startswith(("http://", "https://", "mailto:")):
            continue
        target_path, _, fragment = target.partition("#")
        line_match = LINE_FRAGMENT_RE.fullmatch(fragment)
        if line_match is None:
            continue
        resolved = (source.parent / target_path).resolve()
        if not resolved.is_file():
            continue
        line_count = len(resolved.read_text(encoding="utf-8", errors="replace").splitlines())
        start = int(line_match.group(1))
        end = int(line_match.group(2) or start)
        if start < 1 or end < start or end > line_count:
            continue
        try:
            anchors.append(resolved.relative_to(ROOT).as_posix())
        except ValueError:
            continue
    for match in RAW_LOCAL_ANCHOR_RE.finditer(value):
        resolved = (ROOT / match.group(1)).resolve()
        if not resolved.is_file():
            continue
        line_count = len(resolved.read_text(encoding="utf-8", errors="replace").splitlines())
        start = int(match.group(2))
        end = int(match.group(3) or start)
        if start < 1 or end < start or end > line_count:
            continue
        try:
            anchors.append(resolved.relative_to(ROOT).as_posix())
        except ValueError:
            continue
    return anchors


def is_code_anchor(value: str, source: Path) -> bool:
    if is_placeholder(value):
        return False
    return any(
        path in CODE_ANCHOR_FILES or path.startswith(CODE_ANCHOR_PREFIXES)
        for path in local_line_anchors(value, source)
    )


def is_test_anchor(value: str, source: Path) -> bool:
    if is_placeholder(value):
        return False
    return any(
        path in EVIDENCE_ANCHOR_FILES or path.startswith(TEST_ANCHOR_PREFIXES)
        for path in local_line_anchors(value, source)
    )


def ledger_line_anchors(value: str, source: Path) -> list[str]:
    return [
        path
        for path in local_line_anchors(value, source)
        if path == ".agents/parity-ledger.md"
    ]


# --- record-anchor ratchet (ENG-RECORD-ANCHOR-RATCHET, #632) ------------------
#
# Records cite code as `server_main.cpp:505`. Code moves; the citation does not.
# The checker above LOOKS like it catches that and does not:
#
#   1. NO REPORT, not "no parser". `local_line_anchors` reads BOTH citation
#      forms: the markdown link through `LINK_RE`, and the bare
#      `file.cpp:123` form through `RAW_LOCAL_ANCHOR_RE` since ee511ca8a. It
#      range-checks each one. What it does not do is REPORT: on a missing file
#      and on an out-of-range line the loop runs `continue`, so the bad anchor
#      never enters the returned list, and `is_code_anchor`'s `any()` swallows
#      what is left. There was no symbol test either, which is the gap that
#      matters: 32 of the 38 offenders recorded here are IN RANGE, so a range
#      check could not have found them.
#   2. `any`, NOT `all`. `is_code_anchor` returns true if ONE anchor in a cell
#      qualifies, so a single good link covers arbitrarily many rotted citations
#      beside it. That is not a bug in the STATE gate -- a row IS evidenced by
#      one good anchor, and the `any` stays -- but it is why nothing counted the
#      others.
#   3. STATE. `EVIDENCED_STATES` omits `ACTIVE` and `READY`, so 92 live rows got
#      no anchor check at all.
#
# TWO POPULATIONS, TWO RATIOS, and quoting one without its denominator is what
# produced the false "the bare form was never parsed" claim this comment
# replaces. Measured at `8daa67b39`, the head before this branch merged `main` a
# second time, counting a citation only where it is the WHOLE of a backtick
# span, which is what the parser requires:
#
#   * ALL citation forms in the five matrices, ours and upstream: 492 links and
#     1708 bare, 2200 in total. 525 of the bare forms sit under a
#     `RAW_LOCAL_ANCHOR_RE` prefix, so 1017 of 2200 (46.2%) were already parsed.
#     Most of the remainder are upstream paths that reach no local checker.
#   * The citations this ratchet CLASSIFIES, which is the population that
#     matters: 867. Of those 832 (96.0%) were already parsed AND range-checked
#     before this row, and 35 (4.0%) are genuinely new to parsing, under
#     `.agents/`, `docs/` and `website/`. Every offender recorded then sat in
#     the 96%. The value this row adds is the symbol test and the report, not the
#     parser.
#
# A range check is not the fix: every stale anchor found
# by hand during the 2026-08-13/14 campaign was IN RANGE --
# `docs/USAGE.md:902` (the count line had moved to :1126), `multimodal.py:17-43`
# (the block ends at :45) and `server_main.cpp:308` (a different table entry
# after a 4-line comment landed above). Only "does this line contain the symbol
# named beside it" separates those from a live citation.
#
# THE RATCHET. Enforcing correctness over the whole backlog in one landing would
# surface an unknown amount of unrelated rot, so this mirrors the DSR ratchet in
# scripts/check-device-leakage.py, which this repo already trusts:
# scripts/record-anchor-baseline.json holds the accepted STALE + BROKEN counts,
# a bucket ABOVE its baseline fails, and a bucket BELOW it fails too, with the
# instruction to lower the baseline in the SAME commit as the repair. The number
# only ever moves down, and only deliberately. `--report` names every offender
# so the backlog is legible rather than a number.
#
# WHERE THE CONSERVATIVE LINE IS DRAWN, and why each side of it is where it is.
# A checker that cries wolf gets disabled, and this one has to survive a
# four-figure backlog, so every rule below prefers a missed rot to a false one:
#
#   * ONLY the `code` and `tests` cells of rows in RECORD_ANCHOR_STATES. The
#     `upstream` column is never read. That is what keeps the upstream
#     references (`vllm/model_executor/...py:123`, `csrc/...cu:44`) out of the
#     count structurally, rather than by a path heuristic.
#   * A bare citation must be the WHOLE of a backtick span, so running prose can
#     never be parsed as a path.
#   * It must resolve to a file in the tree, or be a near miss: at least two path
#     separators AND an existing parent directory, i.e. "a directory we own with
#     a filename we do not" -- a rename or a deletion. `vllm/utils/hashing.py`
#     and `tests/v1/core/test_scheduler.py` have no such parent here and are
#     silently skipped, which is correct: they are upstream, and we cannot
#     validate an anchor into a tree we do not have.
#   * A symbol is inferred ONLY from an immediately adjacent backtick span --
#     whitespace between them, or whitespace and one `(` for the trailing
#     `path:line (`Symbol`)` form. Nothing else, and never from prose.
#   * That span must LOOK like a symbol: an identifier, at least 4 characters,
#     carrying `_`, `::`, `()` or an uppercase letter. `bf16` and `nvfp4` sit
#     next to citations constantly and are not symbols; `MoeAuxStream`,
#     `evict_blocks` and `Scheduler::shutdown()` are.
#   * With no inferable symbol the citation is OK by construction. 801 of the
#     867 in-scope citations land there, which is 92.4%, or about 13 in 14.
#     That is the intended polarity: this gate exists to be believed when it
#     does fire.
#   * The symbol test asks whether the cited LINES CONTAIN the name. A comment
#     that mentions the symbol therefore reads OK. That is a measured limit and
#     not a defect: `KERNEL-ATTN-MLA-SPARSE` cites
#     `include/vllm/v1/attention/backend.h:271` for `get_kv_cache_shape`, whose
#     real declaration is at :341, and drift on `main` moved a ROCm comment
#     naming the symbol onto :271. Tightening this would need a parser per
#     language, which is the cry-wolf trade this whole block refuses.
RECORD_ANCHOR_BASELINE = ROOT / "scripts/record-anchor-baseline.json"
RECORD_ANCHOR_VERDICTS = ("ok", "stale", "broken")
# The BUDGET is the rot only. `ok` is counted and printed but deliberately NOT
# stored: a baseline that pinned it would make every PR that adds or removes any
# citation rewrite this file, which is exactly the shared-file lock AGENTS.md
# forbids. Per-bucket rather than one total, so a repaired BROKEN cannot pay for
# a new STALE.
RECORD_ANCHOR_BUCKETS = ("stale", "broken")
# Gap 3. `EVIDENCED_STATES` itself is deliberately NOT widened. Making ACTIVE and
# READY *require* an anchor raises 85 errors across 53 rows that carry prose
# evidence today, which is the bulk cleanup this row exists to avoid. (The unit
# is errors, not rows: the contract check emits one per missing anchor field, so
# a row can raise more than one -- 32 rows raise two here and 21 raise one.)
# They join the COUNT instead, and the ratchet absorbs what that surfaces.
RECORD_ANCHOR_STATES = EVIDENCED_STATES | {"ACTIVE", "READY"}
RECORD_ANCHOR_FIELDS = ("code", "tests")
BARE_CITATION_RE = re.compile(
    r"([A-Za-z0-9_./+-]*[A-Za-z0-9_+-]\.[A-Za-z0-9_+-]+):(\d+)(?:-(\d+))?"
)
BACKTICK_SPAN_RE = re.compile(r"`([^`\n]+)`")
SYMBOL_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)*(?:\(\))?")


@dataclass(frozen=True)
class Citation:
    # No matrix field: row IDs are unique across the matrices (check_matrices
    # enforces it), so the item id already locates the offender.
    item_id: str
    path: str
    start: int
    end: int
    symbol: str | None
    verdict: str

    def describe(self) -> str:
        span = f"{self.start}" if self.end == self.start else f"{self.start}-{self.end}"
        want = f" expected `{self.symbol}`" if self.symbol else ""
        return f"{self.verdict:<6} {self.item_id} -> {self.path}:{span}{want}"


@dataclass
class RecordAnchorResult:
    counts: dict[str, int] = dataclasses.field(
        default_factory=lambda: dict.fromkeys(RECORD_ANCHOR_VERDICTS, 0)
    )
    citations: list[Citation] = dataclasses.field(default_factory=list)

    @property
    def offenders(self) -> list[Citation]:
        return [c for c in self.citations if c.verdict in {"STALE", "BROKEN"}]

    @property
    def total(self) -> int:
        """STALE + BROKEN. `ok` is reported but is not part of the budget."""
        return sum(self.counts[bucket] for bucket in RECORD_ANCHOR_BUCKETS)


def looks_like_symbol(text: str) -> bool:
    text = text.strip()
    if len(text) < 4 or SYMBOL_RE.fullmatch(text) is None:
        return False
    # A LEADING underscore in a record is nearly always an abbreviated suffix --
    # "the text-only `_ModelInfo`" naming the `ModelInfo` beside it, not a
    # symbol spelled `_ModelInfo`. Searching for it literally produced the one
    # false STALE this rule was measured against, on an anchor that was right.
    if text.startswith("_"):
        return False
    return "_" in text or "::" in text or text.endswith("()") or any(c.isupper() for c in text)


def cell_citations(cell: str, source: Path, root: Path) -> list[tuple[Path, int, int, str | None]]:
    """Every citation of THIS tree in one record cell, with its expected symbol.

    Returns `(resolved_path, start, end, symbol_or_None)`. Both citation forms
    are recognised -- the markdown link `[label](path#L12)` the old parser saw,
    and the bare `` `path:12` `` / `` `path:12-20` `` span that is six times
    more common here and was never parsed as a citation at all.
    """
    tokens: list[tuple[int, int, str, object]] = []
    links: list[tuple[int, int]] = []
    for match in LINK_RE.finditer(cell):
        target = match.group(1).strip().strip("<>")
        links.append((match.start(), match.end()))
        if not target or target.startswith(("http://", "https://", "mailto:")):
            continue
        target_path, _, fragment = target.partition("#")
        line_match = LINE_FRAGMENT_RE.fullmatch(fragment)
        if line_match is None:
            continue
        start = int(line_match.group(1))
        end = int(line_match.group(2) or start)
        resolved = (source.parent / target_path).resolve()
        tokens.append((match.start(), match.end(), "citation", (resolved, start, end)))
    for match in BACKTICK_SPAN_RE.finditer(cell):
        # A backtick span inside a link is that link's LABEL, not a token beside
        # it: `[`foo.cpp`](../src/foo.cpp#L4)` must not be read as a neighbour.
        if any(lo <= match.start() < hi for lo, hi in links):
            continue
        inner = match.group(1).strip()
        bare = BARE_CITATION_RE.fullmatch(inner)
        if bare is not None:
            start = int(bare.group(2))
            end = int(bare.group(3) or start)
            resolved = (root / bare.group(1)).resolve()
            tokens.append((match.start(), match.end(), "citation", (resolved, start, end)))
        elif looks_like_symbol(inner):
            tokens.append((match.start(), match.end(), "symbol", inner))
        else:
            tokens.append((match.start(), match.end(), "other", inner))
    tokens.sort(key=lambda t: t[0])

    found: list[tuple[Path, int, int, str | None]] = []
    for index, (lo, hi, kind, payload) in enumerate(tokens):
        if kind != "citation":
            continue
        resolved, start, end = payload  # type: ignore[misc]
        symbol: str | None = None
        if index > 0 and tokens[index - 1][2] == "symbol":
            if not cell[tokens[index - 1][1]:lo].strip():
                symbol = str(tokens[index - 1][3]).strip()
        if symbol is None and index + 1 < len(tokens) and tokens[index + 1][2] == "symbol":
            if cell[hi:tokens[index + 1][0]].strip() in {"", "("}:
                symbol = str(tokens[index + 1][3]).strip()
        found.append((resolved, start, end, symbol))
    return found


def classify_citation(
    resolved: Path, start: int, end: int, symbol: str | None, root: Path
) -> str | None:
    """OK / STALE / BROKEN, or None when the citation is not about this tree."""
    if not resolved.is_file():
        # "A directory we own with a filename we do not" -- a rename or a
        # deletion, and the one missing-file case worth calling BROKEN. Require
        # THREE path components so a one-segment upstream name whose top-level
        # directory happens to match ours (llama.cpp's `src/llama-model.cpp`,
        # vLLM's `cmake/utils.cmake`) is skipped rather than blamed on us.
        try:
            depth = len(resolved.relative_to(root).parts)
        except ValueError:
            return None
        if depth >= 3 and resolved.parent.is_dir():
            return "BROKEN"
        return None
    lines = resolved.read_text(encoding="utf-8", errors="replace").splitlines()
    if start < 1 or end < start or end > len(lines):
        return "BROKEN"
    if symbol is None:
        return "OK"
    base = symbol.removesuffix("()").split("::")[-1]
    body = "\n".join(lines[start - 1:end])
    return "OK" if re.search(r"\b" + re.escape(base) + r"\b", body) else "STALE"


def scan_record_anchors(
    rows: list[ClaimRow] | None = None, root: Path | None = None
) -> RecordAnchorResult:
    root = ROOT if root is None else root
    if rows is None:
        rows, _ = check_matrices([])
    result = RecordAnchorResult()
    for row in rows:
        if row.state not in RECORD_ANCHOR_STATES:
            continue
        # BY INDEX, not by name: several matrices carry one `Local evidence`
        # column that field_index resolves for BOTH `code` and `tests`, and
        # reading it twice would count every citation in it twice.
        indices = {field_index(row.header, name) for name in RECORD_ANCHOR_FIELDS}
        for index in sorted(i for i in indices if i is not None):
            cell = row.cells[index] if index < len(row.cells) else ""
            if is_placeholder(cell):
                continue
            for resolved, start, end, symbol in cell_citations(cell, row.path, root):
                verdict = classify_citation(resolved, start, end, symbol, root)
                if verdict is None:
                    continue
                try:
                    shown = resolved.relative_to(root).as_posix()
                except ValueError:
                    shown = resolved.as_posix()
                result.counts[verdict.lower()] += 1
                result.citations.append(
                    Citation(
                        item_id=row.item_id,
                        path=shown,
                        start=start,
                        end=end,
                        symbol=symbol,
                        verdict=verdict,
                    )
                )
    return result


def load_record_anchor_baseline() -> dict[str, int]:
    if not RECORD_ANCHOR_BASELINE.is_file():
        return {}
    data = json.loads(RECORD_ANCHOR_BASELINE.read_text(encoding="utf-8"))
    return {bucket: int(data["buckets"][bucket]) for bucket in RECORD_ANCHOR_BUCKETS}


def write_record_anchor_baseline(result: RecordAnchorResult) -> int:
    previous = load_record_anchor_baseline()
    if previous and result.total > sum(previous.values()):
        print(
            "REFUSING to write a HIGHER record-anchor baseline "
            f"({sum(previous.values())} -> {result.total}). The ratchet only turns one "
            "way: repair the anchors instead of banking the rot.",
            file=sys.stderr,
        )
        return 1
    payload = {
        "_comment": [
            "Record-anchor baseline for scripts/check-agent-record.py",
            "(ENG-RECORD-ANCHOR-RATCHET, .agents/specs/record-anchor-ratchet.md).",
            "STALE = the cited line exists but does not contain the symbol named",
            "beside it. BROKEN = the line is out of range, or the file is gone.",
            "THESE NUMBERS MAY ONLY EVER GO DOWN. Lower them in the SAME commit as",
            "the repair that earned it, by running:",
            "  python3 scripts/check-agent-record.py --write-baseline",
            "It is a rot budget, never to be raised to make a failing check pass.",
            "Only the rot is stored. The OK count is printed by --report but kept out",
            "of this file on purpose: pinning it would make every change that adds or",
            "removes a citation rewrite this file, which is a lock, not a ratchet.",
        ],
        "total": result.total,
        "buckets": {bucket: result.counts[bucket] for bucket in RECORD_ANCHOR_BUCKETS},
    }
    RECORD_ANCHOR_BASELINE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"baseline written: {RECORD_ANCHOR_BASELINE.name} -> {result.total}")
    return 0


def record_anchor_report(result: RecordAnchorResult) -> str:
    lines = [
        "Record anchors in the `code` / `tests` cells of "
        f"{'/'.join(sorted(RECORD_ANCHOR_STATES))} rows:",
        "",
    ]
    lines.extend(f"  {c.describe()}" for c in result.offenders)
    if result.offenders:
        lines.append("")
    lines.append(
        "record anchors: "
        + ", ".join(f"{b}={result.counts[b]}" for b in RECORD_ANCHOR_VERDICTS)
        + f"  -> rot {result.total}"
    )
    return "\n".join(lines)


def check_record_anchors(result: RecordAnchorResult, errors: list[str]) -> None:
    baseline = load_record_anchor_baseline()
    if not baseline:
        errors.append(
            f"no record-anchor baseline at {RECORD_ANCHOR_BASELINE.relative_to(ROOT)}; "
            "run --write-baseline to establish one"
        )
        return
    for bucket in RECORD_ANCHOR_BUCKETS:
        got, want = result.counts[bucket], baseline[bucket]
        if got > want:
            errors.append(
                f"RECORD ANCHOR REGRESSION in bucket '{bucket}': {got} > baseline {want}. "
                "A citation names a line that no longer holds what the prose says it "
                "does. Run `python3 scripts/check-agent-record.py --report` for the "
                "offenders and repair the anchor. NEVER raise the baseline to pass."
            )
        elif got < want:
            errors.append(
                f"record-anchor baseline STALE in bucket '{bucket}': {got} < baseline "
                f"{want}. A repair must lower the baseline in the SAME commit: run "
                "`python3 scripts/check-agent-record.py --write-baseline` and commit it."
            )


def commit_exists(commit: str) -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def local_spec_paths(row: ClaimRow) -> list[Path]:
    paths: list[Path] = []
    spec_root = (AGENTS / "specs").resolve()
    for raw_target in LINK_RE.findall(row.field("spec")):
        target = raw_target.strip().strip("<>").split("#", 1)[0]
        if not target or target.startswith(("http://", "https://", "mailto:")):
            continue
        resolved = (row.path.parent / target).resolve()
        try:
            resolved.relative_to(spec_root)
        except ValueError:
            continue
        if resolved.suffix == ".md" and resolved.is_file():
            paths.append(resolved)
    return paths


def has_substantive_spec_content(lines: list[str]) -> bool:
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or re.match(r"^#{1,6}\s+", stripped) or stripped.startswith("```"):
            continue
        if stripped.startswith("|"):
            cells = split_cells(stripped)
            if is_separator(cells):
                continue
            next_nonempty = next(
                (candidate.strip() for candidate in lines[index + 1 :] if candidate.strip()),
                "",
            )
            if next_nonempty.startswith("|") and is_separator(split_cells(next_nonempty)):
                continue
            if len(cells) >= 2 and any(not is_placeholder(cell) for cell in cells[1:]):
                return True
            continue
        if not is_placeholder(stripped):
            return True
    return False


def structured_spec_fields(text: str) -> set[str]:
    lines = text.splitlines()
    fields: set[str] = set()
    for index, line in enumerate(lines):
        heading = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
        if heading is not None:
            level = len(line) - len(line.lstrip("#"))
            end = len(lines)
            for candidate in range(index + 1, len(lines)):
                next_heading = re.match(r"^(#{1,6})\s+", lines[candidate])
                if next_heading is not None and len(next_heading.group(1)) <= level:
                    end = candidate
                    break
            if has_substantive_spec_content(lines[index + 1 : end]):
                fields.add(normalize_header(heading.group(1)))
            continue
        if line.startswith("|"):
            cells = split_cells(line)
            next_nonempty = next(
                (
                    candidate.strip()
                    for candidate in lines[index + 1 :]
                    if candidate.strip()
                ),
                "",
            )
            is_header = next_nonempty.startswith("|") and is_separator(
                split_cells(next_nonempty)
            )
            if (
                len(cells) >= 2
                and not is_header
                and not is_separator(cells)
                and any(not is_placeholder(cell) for cell in cells[1:])
            ):
                fields.add(normalize_header(cells[0]))
    return fields


def missing_spec_requirements(text: str) -> list[str]:
    fields = structured_spec_fields(text)
    missing: list[str] = []
    for label, alternatives in SPEC_REQUIREMENTS.items():
        normalized = [normalize_header(alternative) for alternative in alternatives]
        if not any(
            field == alternative or field.startswith(alternative + " ")
            for field in fields
            for alternative in normalized
        ):
            missing.append(label)
    return missing


def check_spec(row: ClaimRow, errors: list[str]) -> None:
    specs = local_spec_paths(row)
    location = f"{row.path.relative_to(ROOT)}:{row.line_no}"
    if not specs:
        errors.append(f"{location}: {row.item_id} {row.state} has no real .agents/specs link")
        return
    token = f"`{row.item_id}`"
    matching = [path for path in specs if token in path.read_text(encoding="utf-8")]
    if not matching:
        errors.append(f"{location}: no linked spec names exact stable token `{row.item_id}`")
        return
    text = matching[0].read_text(encoding="utf-8")
    for label in missing_spec_requirements(text):
        errors.append(
            f"{matching[0].relative_to(ROOT)}: spec for {row.item_id} lacks structured {label}"
        )


def claim_sources() -> list[Path]:
    """Every file that may carry an active claim row.

    One file per claim in .agents/claims/ (ENG-RECORD-CONFLICT-SURFACES, #364),
    plus the legacy table in coordination.md. Both are read, so a claim is
    equally valid in either and no existing row had to be migrated -- the table
    empties as its claims close.

    The per-claim file exists because the table is insert-at-one-anchor: every
    concurrent claim appended at the same line, which made coordination.md the
    largest single conflict source in the repository (8 of the 16 conflicting
    open PRs at origin/main d928e2c3, six of them one author's sequential ROCm
    stack whose only conflict was this). A file with one writer cannot collide.
    """
    sources = [AGENTS / "coordination.md"]
    claims_dir = AGENTS / "claims"
    if claims_dir.is_dir():
        sources.extend(sorted(claims_dir.glob("CLAIM-*.md")))
    return sources


def parse_active_claims(errors: list[str]) -> dict[str, set[str]]:
    claims: dict[str, set[str]] = {}
    origin: dict[str, str] = {}
    for path in claim_sources():
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.startswith("| `CLAIM-"):
                continue
            cells = split_cells(line)
            claim_match = CLAIM_RE.search(cells[0])
            if claim_match is None:
                continue
            claim = claim_match.group(0)
            if claim in claims:
                errors.append(
                    f"{path.relative_to(ROOT)}:{line_no}: duplicate active claim "
                    f"{claim} (already declared in {origin[claim]})"
                )
                continue
            origin[claim] = str(path.relative_to(ROOT))
            claims[claim] = set(ID_RE.findall(cells[1])) if len(cells) > 1 else set()
    return claims


def check_row_contracts(
    rows: list[ClaimRow], by_id: dict[str, ClaimRow], errors: list[str]
) -> None:
    active_claims = parse_active_claims(errors)
    required_fields = ("id", "item", "upstream", "code", "tests", "spec", "state", "owner")

    for row in rows:
        location = f"{row.path.relative_to(ROOT)}:{row.line_no}"
        for field in required_fields:
            if field_index(row.header, field) is None:
                errors.append(f"{location}: {row.item_id} table lacks semantic {field} column")

        if is_placeholder(row.field("item")):
            errors.append(f"{location}: {row.item_id} has no item description")
        if is_placeholder(row.field("upstream")):
            errors.append(f"{location}: {row.item_id} has no upstream anchor/target")

        if row.state in EVIDENCED_STATES:
            if not is_code_anchor(row.field("code"), row.path):
                errors.append(f"{location}: {row.item_id} {row.state} lacks exact local code anchor")
            if not is_test_anchor(row.field("tests"), row.path):
                errors.append(f"{location}: {row.item_id} {row.state} lacks exact test/evidence anchor")

        if row.state in READY_STATES:
            check_spec(row, errors)

        if row.state in {"SPIKE", "ACTIVE"}:
            claim_match = CLAIM_RE.search(row.field("owner"))
            if claim_match is None:
                errors.append(f"{location}: {row.state} row {row.item_id} has no CLAIM-* owner")
            else:
                claim = claim_match.group(0)
                if row.item_id not in active_claims.get(claim, set()):
                    errors.append(
                        f"{location}: owner {claim} does not claim active row {row.item_id} in any claim source"
                    )

        if row.state == "DONE":
            if not ledger_line_anchors(row.field("tests"), row.path):
                errors.append(f"{location}: DONE row {row.item_id} lacks exact parity-ledger link")
            owner = row.field("owner").strip().strip("`")
            if COMMIT_RE.fullmatch(owner) is None:
                errors.append(
                    f"{location}: DONE row {row.item_id} owner is not the hexadecimal closing commit"
                )
            elif not commit_exists(owner):
                errors.append(
                    f"{location}: DONE row {row.item_id} closing commit {owner} does not exist"
                )

    for claim, item_ids in active_claims.items():
        if not item_ids:
            errors.append(f"active claim {claim} has no stable row IDs")
        for item_id in item_ids:
            row = by_id.get(item_id)
            if row is None:
                errors.append(f"active claim {claim} references unknown row {item_id}")
            elif row.state not in {"SPIKE", "ACTIVE"}:
                errors.append(
                    f"active claim {claim} references {item_id} in state {row.state}, not SPIKE/ACTIVE"
                )


def check_model_invariants(errors: list[str]) -> None:
    path = AGENTS / "model-matrix.md"
    rows: list[tuple[list[str], str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| `MODEL-"):
            continue
        cells = split_cells(line)
        if len(cells) < 3:
            continue
        aliases = re.findall(r"`([^`]+)`", cells[1])
        targets = [value for value in re.findall(r"`([^`]+)`", cells[2]) if "::" in value]
        if aliases and targets:
            rows.append((aliases, targets[-1]))

    actual = {
        "rows": len(rows),
        "memberships": sum(len(aliases) for aliases, _ in rows),
        "architectures": len({alias for aliases, _ in rows for alias in aliases}),
        "targets": len({target for _, target in rows}),
        "modules": len({target.split("::", 1)[0] for _, target in rows}),
    }
    expected = {
        "rows": 324,
        "memberships": 373,
        "architectures": 356,
        "targets": 310,
        "modules": 261,
    }
    if actual != expected:
        errors.append(f"{path.relative_to(ROOT)}: model inventory {actual}, expected {expected}")


def check_table_shapes(paths: list[Path], errors: list[str]) -> None:
    for path in sorted(set(paths)):
        expected_pipes: int | None = None
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.startswith("|"):
                expected_pipes = None
                continue
            pipes = len(re.findall(r"(?<!\\)\|", line))
            if expected_pipes is None:
                expected_pipes = pipes
            elif pipes != expected_pipes:
                errors.append(
                    f"{path.relative_to(ROOT)}:{line_no}: table has {pipes} pipes; expected {expected_pipes}"
                )


def check_spec_location(errors: list[str]) -> None:
    misplaced = re.compile(r"(?:spec|scoping|semantics|feasibility|notes)", re.I)
    allowed = {"benchmark-protocol.md"}
    for path in AGENTS.glob("*.md"):
        if path.name not in allowed and misplaced.search(path.stem):
            errors.append(f"{path.relative_to(ROOT)}: feature spec/scoping file belongs in .agents/specs/")


ISSUE_ROW = re.compile(
    r"^\|\s*\[#(\d+)\]\((https://github\.com/[^)]+/issues/(\d+))\)\s*\|"
    r"\s*(?:`([A-Z0-9][A-Za-z0-9_.-]*)`|—)\s*\|"
)


ISSUE_INDEX = AGENTS / "issue-index.md"

# The index carries `merge=union`, so an edited preamble line is DUPLICATED
# rather than merged. Holding the expected text here means the preamble cannot
# drift without a deliberate edit to both sides.
INDEX_PREAMBLE = """# Issue index

**No work without an open issue.** Before claiming a row or writing code,
confirm an issue tracks the work; open one if it does not. The number is linked
from here, from the row's spec, and from the PR.

This file is append-only. Add a row at the end. Never edit a row and never
delete one. GitHub holds the open and closed state, so closing an issue costs no
edit here. `Row` is the owning roadmap block or area-matrix row, or a dash when
a spec lists the issue under `## Owed`.

The path carries `merge=union` in `.gitattributes`, so two branches that each
append a row merge without a conflict. That driver is only safe while the rule
above holds. An edited row and an edited line of this preamble are duplicated
rather than merged. `scripts/check-agent-record.py` gates both.

| Issue | Row | Title | Kind |
|---:|---|---|---|
"""

# Rows naming no owning row AND not listed under a spec's `## Owed`. A RATCHET:
# it may only fall, and a fall must lower this number in the same change.
# Measured 2026-08-14 on 186 rows. Raising it is a checker semantic change and
# needs a spec plus a red-before test, which is the point.
UNOWNED_HIGH_WATER = 33


def owed_issues() -> set[str]:
    """Issue numbers a spec claims under `## Owed`, read with a glob.

    Per-row surface by construction: one file per spec, so filing an owed issue
    never makes two branches write the same line.
    """

    owed: set[str] = set()
    for path in sorted((AGENTS / "specs").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        if "\n## Owed" not in text:
            continue
        body = text.split("\n## Owed", 1)[1].split("\n## ", 1)[0]
        owed |= set(re.findall(r"#(\d+)", body))
        owed |= set(re.findall(r"issues/(\d+)", body))
    return owed


def check_issue_index(
    errors: list[str],
    text: str | None = None,
    owed: set[str] | None = None,
    high_water: int | None = None,
) -> None:
    """Every tracked issue is well-formed, consistent, and owned.

    Deliberately NETWORK-FREE. Querying GitHub would make this gate fail on
    connectivity, which is exactly the class of flake this protocol exists to
    remove. It checks the FORM, the internal consistency, and whether an owner
    is named; whether the issue is still open is GitHub's record, not a CI
    blocker.

    `text`, `owed` and `high_water` are injectable so a test can build a small
    index. Without an injectable mark every fixture would be red for having the
    wrong number of unowned rows, which would hide whatever the fixture is
    actually about.
    """

    label = ISSUE_INDEX.relative_to(ROOT)
    if text is None:
        if not ISSUE_INDEX.is_file():
            errors.append(f"{label}: missing the issue index")
            return
        text = ISSUE_INDEX.read_text(encoding="utf-8")
    if owed is None:
        owed = owed_issues()
    if high_water is None:
        high_water = UNOWNED_HIGH_WATER

    if not text.startswith(INDEX_PREAMBLE):
        actual = text.split("| [#", 1)[0]
        expected_lines = INDEX_PREAMBLE.splitlines()
        for number, line in enumerate(actual.splitlines(), 1):
            if number > len(expected_lines) or line != expected_lines[number - 1]:
                errors.append(
                    f"{label}:{number}: preamble drifted from the checker's copy; "
                    f"got {line[:60]!r}. A union merge DUPLICATES an edited "
                    "preamble line instead of merging it"
                )
                break
        else:
            errors.append(f"{label}: preamble is shorter than the checker's copy")

    seen: set[str] = set()
    rows = 0
    unowned: list[str] = []
    for line in text.splitlines():
        # Any table line that is not the header or the separator. Matching only
        # `| [#` would make a row that LOST its link invisible instead of
        # malformed, which is the failure this loop exists to report.
        if not line.startswith("|") or line.startswith("| Issue") or set(line) <= set("|-: "):
            continue
        match = ISSUE_ROW.match(line)
        if not match:
            errors.append(
                f"{label}: malformed issue row {line[:60]!r}; "
                "expected | [#N](https://github.com/.../issues/N) | `ROW-ID` or — | title | kind |"
            )
            continue
        rows += 1
        number, url, url_number, row_id = match.group(1), match.group(2), match.group(3), match.group(4)
        if number != url_number:
            errors.append(f"{label}: issue #{number} links to {url}, a different issue")
        if number in seen:
            errors.append(
                f"{label}: issue #{number} listed twice. Under `merge=union` a "
                "duplicate is what two branches appending the same issue look like"
            )
        seen.add(number)
        if row_id is None and number not in owed:
            unowned.append(number)

    if rows == 0:
        errors.append(f"{label}: the issue index has no rows")

    if len(unowned) > high_water:
        fresh = unowned[high_water:]
        errors.append(
            f"{label}: {len(unowned)} rows name no owner, above the recorded "
            f"{high_water}: {', '.join('#' + n for n in fresh[:5])}. "
            "Name an owning row ID, or list the issue under `## Owed` in the "
            "spec that owes it. Filing an issue does not defer the fix"
        )
    elif len(unowned) < high_water:
        errors.append(
            f"{label}: {len(unowned)} rows name no owner, below the recorded "
            f"{high_water}. Lower UNOWNED_HIGH_WATER to {len(unowned)} "
            "in the same change, so the ratchet cannot slip back"
        )


def check_roadmap(by_id: dict[str, ClaimRow], errors: list[str]) -> None:
    path = AGENTS / "roadmap_v1.md"
    expected_blocks = [
        "ROAD-V1-A",
        "ROAD-V1-C1",
        "ROAD-V1-C2",
        "ROAD-V1-C3",
        "ROAD-V1-C4",
        "ROAD-V1-C5",
        "ROAD-V1-C6",
        "ROAD-V1-C7",
        "ROAD-V1-C8",
        "ROAD-V1-C9",
        "ROAD-V1-D1",
        "ROAD-V1-D2",
        "ROAD-V1-D3",
        "ROAD-V1-D4",
        "ROAD-V1-D5",
        # +D6 2026-08-05: llama.cpp device breadth folded into scope (user-directed).
        "ROAD-V1-D6",
    ]
    seen: list[tuple[int, str]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.startswith("|"):
            continue
        cells = split_cells(line)
        if len(cells) < 6 or not cells[0].isdigit():
            continue
        block_id = cells[1].strip().strip("`")
        if not re.fullmatch(r"ROAD-V1-[A-Z0-9]+", block_id):
            continue
        seen.append((int(cells[0]), block_id))
        if STATE_RE.fullmatch(cells[5].strip()) is None:
            errors.append(
                f"{path.relative_to(ROOT)}:{line_no}: portfolio State cell needs exactly one state"
            )
    if seen != list(enumerate(expected_blocks)):
        errors.append(f"{path.relative_to(ROOT)}: portfolio order {seen}; expected {list(enumerate(expected_blocks))}")

    for source in (path, AGENTS / "coordination.md"):
        text = source.read_text(encoding="utf-8")
        for item_id in set(re.findall(r"`(" + ID_RE.pattern + r")`", text)):
            if item_id not in by_id:
                errors.append(f"{source.relative_to(ROOT)}: references unknown stable row {item_id}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        action="store_true",
        help="print every STALE/BROKEN record anchor with the symbol expected",
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="rewrite scripts/record-anchor-baseline.json (only ever DOWNWARD)",
    )
    args = parser.parse_args(argv)

    errors: list[str] = []
    for path in REQUIRED:
        if not path.is_file():
            errors.append(f"missing canonical record: {path.relative_to(ROOT)}")

    rows: list[ClaimRow] = []
    by_id: dict[str, ClaimRow] = {}
    if not errors:
        check_links(errors)
        check_issue_index(errors)
        rows, by_id = check_matrices(errors)
        check_engine_summary(rows, errors)
        check_row_contracts(rows, by_id, errors)
        check_model_invariants(errors)
        spec_paths = [path for row in rows if row.state in READY_STATES for path in local_spec_paths(row)]
        # ISSUE_INDEX is here for the same reason every other record table is:
        # nothing else counts its cells. It was the ONE record surface every
        # change must write and the only markdown table in the set with no shape
        # gate, so a row that lost its trailing pipe, or carried an unescaped one
        # inside a code span, mis-rendered on GitHub while every gate stayed
        # green (#1033). The constant is reused rather than respelled so this
        # gate and check_issue_index cannot drift onto different files.
        check_table_shapes(
            [
                AGENTS / "roadmap_v1.md",
                AGENTS / "coordination.md",
                ISSUE_INDEX,
                *MATRIX_PATHS,
                *spec_paths,
            ],
            errors,
        )
        check_spec_location(errors)
        check_roadmap(by_id, errors)

        anchors = scan_record_anchors(rows)
        if args.report:
            print(record_anchor_report(anchors))
        if not args.write_baseline:
            # Writing is a MODE, not a step: it must not also report the gate it
            # is about to move, or a run that lowered the baseline would print a
            # regression against the value it just replaced.
            check_record_anchors(anchors, errors)
    elif args.write_baseline:
        print(
            "REFUSING to write a baseline from a tree whose record does not parse.",
            file=sys.stderr,
        )

    if errors:
        for error in dict.fromkeys(errors):
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    if args.write_baseline:
        # AFTER the error gate, deliberately. The mode used to return the moment
        # it had a number, so a tree that failed some OTHER record check could
        # still bank its rot, and the banked figure would carry the authority of
        # a run that never passed. A baseline is a measurement of the record, so
        # it is only taken from a record that checks out.
        return write_record_anchor_baseline(anchors)

    counts = [
        "ENGINE="
        + str(
            sum(
                any(row.item_id.startswith(prefix + "-") for prefix in ENGINE_PREFIXES)
                for row in rows
                if row.path == ENGINE_MATRIX
            )
        )
    ]
    for prefix, (path, _) in MATRICES.items():
        counts.append(
            f"{prefix}="
            + str(sum(row.item_id.startswith(prefix + "-") for row in rows if row.path == path))
        )
    counts.append(f"ANCHOR-ROT={anchors.total}")
    print("agent record OK: " + " ".join(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
