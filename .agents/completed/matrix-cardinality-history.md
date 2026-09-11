# Retired matrix-cardinality history

Issue #3085 retired the checker constants that these comments accompanied.
The text below is preserved verbatim as historical evidence; it does not set
or validate a current matrix size. The SHA-256 covers every UTF-8 byte after the
begin marker through the newline immediately before the end marker. The marker
lines and digest line are excluded.

<!-- matrix-cardinality-history-sha256: 7c834576bc1119d52e3a88e8ded3c29bcce7b640ee4ef02bd7b9bdeef0b744b7 -->
<!-- matrix-cardinality-history:begin -->

## Model, quantization, kernel, and backend history

```python
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
    # 378 since 2026-08-26, and RE-DERIVED off the matrix rather than carried
    # forward: +1 for `MODEL-MM-qwen4-exp-qwen4-exp-for-conditional-generation`
    # (`Qwen4ExpForConditionalGeneration`, `Qwen/Qwen3.8-Flash-Next`), landing
    # `READY` with its spec committed (#1978). ONE row and not two: the MTP head
    # is a `mtp` block inside the same text config, not a separately registered
    # architecture, so this is not the IndexTTS-2.5 / dots3-note shape that moved
    # this pin by two. Beyond-pin in the strongest sense yet recorded here -- the
    # Muse Glimmer, Qwen3.5-text-only and dots3-note entries above are all
    # architectures vLLM registers on `main` AFTER `555967922`, whereas this one
    # vLLM does not implement at ANY revision: read live 2026-08-26 at
    # `origin/main` = `6a5e8f5979`, there is no `qwen4*` path, no `registry.py`
    # entry, and a repository-wide search for `qwen4` returns zero results. Its
    # Upstream cell therefore carries no pinned module/class target and its
    # algorithm source is transformers#48337, so the at-the-pin static invariants
    # (324/373/356/310/261) are UNCHANGED. Bumped because one row EXISTS, never to
    # make a transition pass.
    # 379 since 2026-08-26, and RE-DERIVED off the matrix rather than carried
    # forward: +1 for `MODEL-MM-glm5-next-glm5-next-for-conditional-generation`
    # (`Glm5NextForConditionalGeneration`, `zai-org/GLM-5.3-Flash`), landing
    # `READY` with its spec committed (#1998). ONE row and not three, which is
    # the arithmetic this comment exists to justify: the OPEN vllm#53906 would
    # register `Glm5NextForCausalLM`, `Glm5NextForConditionalGeneration` AND
    # `Glm5NextMTPModel`, so the IndexTTS-2.5 / dots3-note two-row shape is the
    # tempting read. It does not apply. None of the three names is registered at
    # ANY vLLM revision, and the only architecture a published artifact declares
    # is `Glm5NextForConditionalGeneration` -- the MTP head is `layers.45` inside
    # the same checkpoint, which the transformers reference discards outright at
    # `modular_glm5_next.py:1235`. Beyond-pin in the same strongest sense as the
    # qwen4-exp row above: read live 2026-08-26, `git grep "Glm5\|glm5_next"`
    # returns zero hits at `555967922` and at `origin/main` = `c71f6f8a81`. Its
    # Upstream cell therefore carries no pinned module/class target and the
    # at-the-pin static invariants (324/373/356/310/261) are UNCHANGED. Bumped
    # because one row EXISTS, never to make a transition pass.
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
    # 85 since 2026-08-28: +`QUANT-EXL3`, the exllamav3 trellis format (a QTIP
    # variant: MCG codebook, blockwise Hadamard-128 with sign vectors, and NO
    # scales). A genuinely new scheme rather than a state transition -- it is
    # expressible by no row in sections 1 or 2, which are keyed on GGUF encodings
    # and on vLLM-registered methods, and vLLM registers no EXL3 at the parity
    # pin, so its mirror source is the pinned secondary oracle `exllamav3`. The
    # kernels have existed since `MODEL-DSV4-EXL3` W2 and are device-proven, but
    # the ONLY consumer is the DeepSeek-V4 loader, so no other architecture can
    # reach the scheme -- which is what the row is for (#2181).
    # 86 since 2026-09-01: +`QUANT-EXL3-MUL1`, exllamav3's `mul1` codebook (cb 2)
    # and the 4- and 5-bit trellis widths. A separate row from `QUANT-EXL3` and
    # not a state transition on it, because cb 2 is a DIFFERENT DECODE rather
    # than a third multiplier: cb 0 and cb 1 mask, xor and sum the two fp16
    # halves of the product, while cb 2 sums the product's four bytes into an
    # fp16 bit pattern and maps it with a fused fp16 affine
    # (`codebook.cuh:82-89`). It has its own artifact
    # (`Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw`), its own widths, and its own owed
    # GEMV arm, none of which `QUANT-EXL3`'s cells can carry without saying two
    # things at once (#2495).
    # 87 since 2026-09-02: +`QUANT-EXL3-PERF`, the EXL3 `m<=8` GEMV arm set and
    # its selection envelope. A separate row from `QUANT-EXL3` and
    # `QUANT-EXL3-MUL1` because both of those are CORRECTNESS rows and say so --
    # they make a width RUN -- while this one owns what the format COSTS, which
    # is a different verdict on a different axis. #2570 named a verified
    # instantiation gap with no owner, and an unowned gap is one nobody reruns
    # (#2570).
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
    # 58 since 2026-08-25 (#1451): +`KERNEL-LTX2-VAE`, the ten stages of the
    # LTX-2.5 conv video VAE decode that sit BETWEEN its convolutions, as one
    # `vt::OpId::kLtx2Vae` provider payload. A separate family from
    # `KERNEL-CONV3D` rather than more entries on it, on the same axis that
    # separates the rows above: `KERNEL-CONV3D` is a dense CONTRACTION whose
    # difficulty is a reduction and whose accumulation order is its contract,
    # while these ten are elementwise affines, normalisations and pure GATHERS
    # whose contract is an index expression -- and the two are gated differently
    # in consequence, the convolution against an independent scalar reference and
    # these against the committed decode goldens they were transcribed from.
    # Bumped because the row EXISTS, never to make a state transition pass; it is
    # `ACTIVE` rather than `DONE` because its CUDA arm has never compiled
    # anywhere in this project's reach (spec `## Owed`, inheriting #1452) and
    # because `AttnBlock3d` remains the declared staged remainder.
    # 59 since 2026-09-05: +`KERNEL-QUANT-CIQ-GEMM-ROCM`, the RDNA4 rocWMMA
    # int8 tile arm of the ROCm keep-quant GEMM prefill path (issue #2109). A
    # separate row from `KERNEL-QUANT-CIQ-GEMM-CUDA` above rather than a
    # column on it: the two share a name pattern and a Q8_K-family activation
    # format, but serve different providers (`kROCM` vs `kCUDA`), gate against
    # different oracles (this row's own real-model `rocprofv3` trace on the
    # actual RX 9060 XT hardware vs the CUDA row's DGX GB10 run), and the CUDA
    # row's spec (`cuda-keepquant-gemm.md`) makes no claim about this row's
    # architecture-gated WMMA tile mechanism, which does not exist on CUDA.
    # Bumped because the row EXISTS, never to make a state transition pass.
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
    # 88 since 2026-09-01: +`BACKEND-GATE-ROCM-LLAMACPP` (#2497), the ROCm
    # GGUF k-quant floor. Every other backend already had its llama.cpp
    # gate row; ROCm had only the vLLM and SGLang rows, and neither has a
    # denominator on `gfx1151` because vLLM has no entry on that
    # architecture. The first measurement landed with nowhere to record
    # it. `INVENTORIED`, no owner, no spec of its own. Bumped for a real
    # new row, never to make a failing state transition pass.
    # 89 since 2026-09-05: +`BACKEND-TENSTORRENT-GDN-DEVICE-PURE`, the
    # device-resident GDN decode wave split out of the Qwen35 wiring row
    # (#2907, owed from #2812); spec committed first on the row branch.
    # 90 since 2026-09-05: +`BACKEND-TENSTORRENT-KEEPQUANT`, the dense
    # keep-quant dot row (#2959); spec committed first on the row branch.
```

## Engine history

```python
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
# 172 since 2026-08-25: +`KV-DSV4-MULTICACHE` (DeepSeek-V4's real KV topology,
# #1925). A genuinely-new row rather than a state move: no KV-* row and no
# DeepSeek-V4 row owned the cache topology, `MODEL-TEXT-deepseek-v4-...` promises
# "paged attention/KV" in its scope string while both its forwards discard
# `attn_kv`, and the work was tracked only as prose calling itself
# multi-Spark-blocked. `READY`, spec `specs/kv-dsv4-multicache.md`. Bumped for a
# real new row, never to make a failing state transition pass.
# 173 since 2026-08-26: +`ENG-POOL-BEST-FIT` (the shared device scratch pool
# could only hand a freed block back to a request in the block's OWN size class,
# so retention was a function of how many distinct shapes the traffic had shown
# rather than of how much one step concurrently needs, #1922). A genuinely-new
# row rather than a state move: `POOL-DEVICE-KEY` is `DONE` and owned the pool's
# DEVICE key, nothing owned its REUSE policy, and no engine row covered
# per-request memory growth at all. `ACTIVE`, spec
# `specs/pool-best-fit-retention.md`.
#
# 172 AND 173 ARE THE SAME NIGHT, and the pair is why this is 173 and not 172.
# #1925 and #1922 each added ONE engine row and each bumped this pin 171 -> 172
# on its own branch, so the two sides carry an IDENTICAL `ENGINE_ROWS = 172`
# line. A three-way merge sees no conflict on that line and keeps 172, which
# counts one of the two new rows and silently drops the other. The counter is
# the union, so it is 173. Bumped for real new rows, never to make a failing
# state transition pass.
# 174 since 2026-08-31: +`ENG-PREFLIGHT-COMPILES` (no gate compiles a translation
# unit before a push, and `main` was pushed twice on 2026-08-31 in a state that
# does not build with every record gate green -- `5263ac31f` and `08fa2f5aa`,
# #2401). A genuinely-new row rather than a state move: `GATE-PREPUSH-FAIL-LOUD`
# owns whether the hook can find the checkers it names, `ENG-CI-*` owns the CI
# lanes, and nothing owned whether anything compiles before the push at all.
# `ACTIVE`, spec `specs/preflight-compiles.md`. Bumped for a real new row, never
# to make a failing state transition pass.
# +4 on 2026-08-31: four dead-capability rows catalogued while gating the MoE
# placement install -- `ENG-WEIGHT-RESIDENCY`, `ENG-STRUCTURED-OUTPUT`,
# `ENG-ATTENTION-WINDOW` and `ENG-GATE-ENV-DOC`. Each is a symbol or knob with a
# user-facing promise and no production caller, verified by `git grep` restricted
# to `src include`. Real new rows, which is the only reason this number moves.

```

## Retired live-test cardinality prose

These test comments and docstrings were still live after the checker stopped
enforcing cross-file cardinalities. They are preserved verbatim here because
they explain the historical transitions, collisions, and mutation method. They
do not define current test behavior.

````text
--- tests/scripts/test_agent_record.py:245-245 ---
"""The #117 row and its ratchet bump are one semantic change."""
--- tests/scripts/test_agent_record.py:255-271 ---
"""The four dead-capability rows and the 173 -> 177 bump are one change.

        Each row records a symbol or knob with a user-facing promise and no
        production caller, found while gating the MoE placement install: the
        `VT_QWEN35_STAGE_MIN_FREE_FRAC` knob whose only hit in compiled code is a
        comment, jump-forward's `DrainForcedTokens`, the unreachable
        `ResolveAttentionWindow`, and the one-directional env-doc gate that let
        the first one outlive its reader.

        This is the ratchet's own evidence contract, and it is load-bearing in
        both directions. Against the BASE checker, pinned at 173 while the matrix
        carries 177, `check_matrices` reports an engine-row count error and this
        test FAILS. Against HEAD it passes. That is what separates a bump made
        for four real rows from a bump made to silence a failure -- the
        distinction `test_engine_row_ratchet_is_load_bearing` exists to protect
        and that this case supplies the instance of.
        """
--- tests/scripts/test_agent_record.py:295-303 ---
"""The #606 row and its 152 -> 153 ratchet bump are one semantic change.

        Same shape as the #117 assertion above, and it exists for the same
        reason: the bump and the row have to arrive together, or a number was
        moved to silence a failure. This one also pins WHICH matrix owns the
        row, because SERVE-* IDs are reachable from more than one, and a row
        that drifted into another matrix would leave the engine count short
        while the pin still read 153.
        """
--- tests/scripts/test_agent_record.py:313-323 ---
"""The #633 row and its 153 -> 154 ratchet bump are one semantic change.

        Same shape as the #117 and #606 assertions above, and it carries one
        extra hazard worth pinning. This bump COLLIDED: `main` took the constant
        152 -> 153 for `SERVE-RECIPE-ARGS` while the omni-pin branch took the
        same 152 -> 153 for its own row, so both sides read 153 and the merge
        looked clean. Resolving it by keeping either 153 would have dropped a
        real row while leaving the matrix internally consistent, which is
        exactly the state no other assertion here can see. Naming BOTH rows is
        what makes 154 checkable rather than plausible.
        """
--- tests/scripts/test_agent_record.py:436-448 ---
"""373 needs BOTH rows named, because the merge that produced it collided.

        The same hazard the omni-pin assertion above records, on the MODEL pin
        and on the same day. `main` took the constant 370 -> 372 for IndexTTS-2.5
        (two architectures) while the Music3 branch took 370 -> 371 for its own
        row. Neither side was wrong about its own change, and neither number was
        373 -- so whichever side an auto-merge kept, the tree would have been
        internally consistent while silently short a real architecture.

        A count assertion alone cannot see that: it only knows the pin matches
        the rows it can find. Naming the three rows is what makes 373 checkable
        rather than plausible.
        """
--- tests/scripts/test_agent_record.py:1131-1135 ---
"""The BACKEND ratchet bump is backed by a real row (#393).

    The count is re-pinned by hand, so a bump with no row behind it looks
    exactly like a bump for a new row. This ties this bump to this row.
    """
--- tests/scripts/test_agent_record.py:1147-1155 ---
"""The MODEL ratchet bump 373 -> 375 is backed by two real rows (#490).

    Same shape, and the same reason, as the BACKEND class above: the count is
    re-pinned by hand, so a bump with nothing behind it is indistinguishable
    from a bump for rows that really landed. `test_model_row_ratchet_is_
    load_bearing` proves the pin BINDS by moving it, which holds for any value
    of the pin; it cannot say whether THIS value is the right one. These two
    tests do, by tying the pin to the rows the matrix actually carries.
    """
--- tests/scripts/test_agent_record.py:1173-1180 ---
"""The BACKEND ratchet bump to 82 is backed by a real row (#670).

    Same shape as TenstorrentResidualGoldenRowIsCounted and for the same
    reason: the count is re-pinned by hand, so a bump with no row behind it is
    indistinguishable from a bump for a new row. `b55f6ec14` set the precedent
    that a ratchet bump lands with a case keyed to ITS OWN row; this is that
    case for BACKEND-TENSTORRENT-MISTRAL.
    """
--- tests/scripts/test_agent_record.py:1208-1208 ---
"""The BACKEND ratchet bump to 84 is backed by a real row (#1105)."""
--- tests/scripts/test_agent_record.py:1221-1221 ---
"""The BACKEND ratchet bump to 85 is backed by a real row (#1105)."""
--- tests/scripts/test_agent_record.py:1234-1247 ---
"""The BACKEND ratchet bump to 83 is backed by a real row (#979).

    Same shape and same reason as the two cases above: the count is re-pinned
    by hand, so a bump with no row behind it is indistinguishable from a bump
    for a new row. Inherits the removal mutation unchanged so only the row and
    its two required links differ.

    This row exists because the llama.cpp comparator on a CURRENT CUDA card had
    no owner at all. `BACKEND-GATE-CPU-LLAMACPP` is the CPU floor and
    `BACKEND-GATE-CUDA-LLAMACPP-LEGACY` is scoped to the pre-Ampere arches vLLM
    drops, so a GB10 GGUF comparison fell between them, which is how
    `bench-27b-five-way.md` came to list a llama.cpp CUDA arm with nothing
    tracking it.
    """
--- tests/scripts/test_agent_record.py:1283-1300 ---
"""The ENGINE ratchet bump 156 -> 157 is backed by a real row (#81).

    Same shape and the same reason as the BACKEND classes above, applied to the
    pin that actually moved in this change. `ENGINE_ROWS` is re-pinned by hand,
    so a bump with nothing behind it is indistinguishable from a bump for a row
    that landed. `test_engine_row_ratchet_is_load_bearing` proves the pin BINDS
    by moving it, which holds for any value of the pin and cannot say whether
    157 is the right value. This class says that, by tying the pin to the row
    the matrix carries.

    The ENGINE pin is not in `MATRICES`, it is the module constant
    `ENGINE_ROWS` counted over rows whose `path` equals `ENGINE_MATRIX`, so the
    removal mutation redirects `ENGINE_MATRIX` and `MATRIX_PATHS` TOGETHER.
    Rows are parsed from the list while the count is taken against the
    constant, so patching one alone counts zero engine rows for a reason that
    has nothing to do with the removal, and the case would go red for the wrong
    reason.
    """
--- tests/scripts/test_agent_record.py:1804-1812 ---
"""The ENGINE ratchet bump 164 -> 165 is backed by a real row (#1280).

    Same shape and the same reason as `MtpDepthRowIsCounted`, applied to the pin
    this change moves. `ENGINE_ROWS` is re-pinned by hand, so a bump with
    nothing behind it looks exactly like a bump for a row that landed.
    `test_engine_row_ratchet_is_load_bearing` proves the pin BINDS by moving it,
    which holds for any value and cannot say whether 165 is the right one. This
    class says that, by tying the pin to the row the matrix carries.
    """
--- tests/scripts/test_agent_record.py:1832-1850 ---
"""The ENGINE ratchet bump 167 -> 168 is backed by a real row (#1365).

    Same shape and the same reason as `HfModelDownloadRowIsCounted`, applied to
    the pin this change moves. `ENGINE_ROWS` is re-pinned by hand, so a bump
    with nothing behind it looks exactly like a bump for a row that landed.
    `test_engine_row_ratchet_is_load_bearing` proves the pin BINDS by moving it,
    which holds for any value and cannot say whether 168 is the right one. This
    class says that, by tying the pin to the row the matrix carries.

    This class asserts nothing about `.agents/issue-index.md`, where the sibling
    classes assert their issue number, and this change appends no row there.
    #1365's row already landed in `9e1a5e573` and a second row for one issue
    number is what `check-agent-record.py` reports as `issue #1365 listed
    twice`. The row's TEXT is stale, because #1365 was re-scoped in place from
    the symptom onto the cause after the row landed, so `assertIn("issues/1365)",
    index)` would pass here against a row describing the symptom and would
    measure nothing about this row's work. The staleness is recorded in the
    spec's `## Dependencies` instead, where prose can say it.
    """
--- tests/scripts/test_agent_record.py:1872-1885 ---
"""The ENGINE ratchet bump 169 -> 170 is backed by a real row (#1541).

    Same shape and the same reason as `BpeQuadraticMergeRowIsCounted`, applied
    to the pin this change moves. `ENGINE_ROWS` is re-pinned by hand, so a bump
    with nothing behind it looks exactly like a bump for a row that landed, and
    `scripts/check-pr-size.py`'s `governance_checker` contract refuses a checker
    constant whose only artifact is the constant.

    This class asserts nothing about `.agents/issue-index.md`, and that is not
    an omission. #1541's row already landed with the closing commit of
    `SPEC-BPE-QUADRATIC-MERGE`, the index is append-only, and a second row for
    one issue number is what `check-agent-record.py` reports as `issue #1541
    listed twice`.
    """
--- tests/scripts/test_agent_record.py:1907-1926 ---
"""The ENGINE ratchet bump 171 -> 172 is backed by a real row (#1925).

    Same shape and the same reason as `RequestLengthGuardRowIsCounted`, applied
    to the pin this change moves. `ENGINE_ROWS` is re-pinned by hand, so a bump
    with nothing behind it looks exactly like a bump for a row that landed, and
    `scripts/check-pr-size.py`'s `governance_checker` contract refuses a checker
    constant whose only artifact is the constant.

    `test_engine_row_ratchet_is_load_bearing` proves the pin BINDS by moving it,
    which holds for any value and cannot say whether 172 is the right one. This
    class says that, by tying the pin to the row the matrix now carries.

    One case beyond the precedent shape. The bump did not only move the total:
    it moved the `KV cache and memory` area from 25 rows to 26 and its `READY`
    column from 3 to 4, and those cells are a second hand-maintained record of
    the same landing. A row counted into the total but sitting under the wrong
    heading, or carrying a state the area column does not expect, keeps the pin
    at 172 and still misfiles the work, so the placement is asserted from the
    document's own headings rather than left to the total alone.
    """
--- tests/scripts/test_agent_record.py:2135-2149 ---
"""The KERNEL ratchet bump 57 -> 58 is backed by a real row (#1451).

    Same shape and the same reason as `MtpDepthRowIsCounted` above, applied to
    the pin that moved in this change. `MATRICES["KERNEL"]` is re-pinned BY
    HAND, so a bump with nothing behind it is indistinguishable from a bump for
    a row that landed -- and a recorded expected-count that can move without
    evidence stops catching the thing it exists to catch, the unrecorded row.
    That is the same shape as a floor set below the real count.

    `test_kernel_row_ratchet_is_load_bearing` elsewhere proves the pin BINDS by
    moving it, which holds for ANY value and cannot say whether 58 is the RIGHT
    one. These cases say that, two ways: by tying the pin to the row the matrix
    carries, and by deriving the expected value from the shipped file rather
    than restating the constant.
    """
--- tests/scripts/test_agent_record.py:2173-2182 ---
"""The BACKEND ratchet bump 86 -> 87 is backed by the Qwen3.5 GDN row
    (#1715).

    Same shape and reason as the BACKEND-TENSTORRENT-RESIDUAL-GOLDEN class
    above: the count is re-pinned by hand, so a bump with no row behind it is
    indistinguishable from a bump for a row that really landed. The first test
    ties THIS value of the pin to a real matrix line; the second proves the pin
    BINDS against the shipped matrix file through the checker's own entry point,
    which is what makes the pair semantic evidence rather than a restatement.
    """
--- tests/scripts/test_agent_record.py:2491-2502 ---
"""The BACKEND ratchet bump 88 -> 89 is backed by the GDN device-pure row
    (#2907, owed from #2812).

    Same shape as `Ltx2VaeKernelRowIsCounted`, applied to the pin that moved in
    this change: `MATRICES["BACKEND"]` is re-pinned BY HAND, so a bump with
    nothing behind it is indistinguishable from a bump for a row that landed.
    The cases tie THIS value of the pin to the row the matrix carries, derive
    the expected count from the shipped file rather than restating the
    constant, and prove the comparison real by moving the pin one in EACH
    direction -- an off-by-one that only reds downward would let the count
    grow silently, which is how an unrecorded row hides.
    """
--- tests/scripts/test_agent_record.py:2538-2549 ---
"""The BACKEND ratchet bump 89 -> 90 is backed by the keep-quant row
    (#2959).

    Same shape as `GdnDevicePureBackendRowBacksTheRatchet`, applied to the pin
    that moved in this change: `MATRICES["BACKEND"]` is re-pinned BY HAND, so
    a bump with nothing behind it is indistinguishable from a bump for a row
    that landed. The cases tie THIS value of the pin to the row the matrix
    carries, derive the expected count from the shipped file rather than
    restating the constant, and prove the comparison real by moving the pin
    one in EACH direction -- an off-by-one that only reds downward would let
    the count grow silently, which is how an unrecorded row hides.
    """
````
## Additional retired live-test rationale

````text
--- tests/scripts/test_agent_record.py:334-343 ---
"""The #632 row and its 154 -> 155 bump are one semantic change.

        Same shape as the #117, #606 and #633 assertions above. Worth naming
        here for one reason beyond the count: this row exists BECAUSE the
        `path:line` citations in these matrices were 83% unparsed by the very
        checker this test guards. That is no longer true, and the row is now
        the first thing its own ratchet polices -- `RecordAnchorRatchet` below
        counts the anchors in this row's `Our code` cell like any other. Pinning
        the row still says the thing a count cannot: that it exists.
        """
--- tests/scripts/test_agent_record.py:354-372 ---
"""The #1110 row and its 157 -> 158 bump are one semantic change.

        Same shape as the #117, #606, #633 and #632 assertions above, and owed for
        the same reason: the bump is the whole of what
        `scripts/check-agent-record.py` changed for this row, so without an
        assertion naming the row the constant is the only artifact and 158 is
        plausible rather than checkable. That is exactly the state the
        `governance_checker` evidence contract in `scripts/check-pr-size.py`
        refuses, and it refused this row's first commit by name.

        The pin is also the only mechanical statement available about WHERE this
        row belongs. It sits between two offload rows that could each plausibly
        have absorbed it -- `ENG-WEIGHT-OFFLOAD` owns the mirrored device-to-host
        tier and cannot grow a disk arm without breaking a 1:1 transcription of
        `vllm/config/offload.py`, and `ENG-EXPERT-STREAM` owns the streaming
        mechanism rather than its configuration -- so "a genuinely new row" is a
        claim, and naming it here is what makes the claim fail if the row is ever
        folded into a neighbour without the count following.
        """
--- tests/scripts/test_agent_record.py:383-398 ---
"""The #1922 row and its 171 -> 172 bump are one semantic change.

        Same shape as the #117, #606, #633, #632, #1110 and #1433 assertions
        around it, and owed for the same reason: the bump is the whole of what
        `scripts/check-agent-record.py` changed for this row, so without an
        assertion naming the row the constant is the only artifact and 172 is
        plausible rather than checkable.

        The hazard this row carries is not the count, it is the NEIGHBOUR.
        `POOL-DEVICE-KEY` is `DONE` and owns the same allocator's DEVICE key,
        and a reader who folds the two together leaves the matrix internally
        consistent while silently retiring the row that owns its REUSE policy --
        which is what #1922 is about and what `POOL-DEVICE-KEY` never covered.
        So the section is asserted too: this row lives in the engine matrix, and
        exactly once.
        """
--- tests/scripts/test_agent_record.py:409-423 ---
"""The #1433 row and its 170 -> 171 bump are one semantic change.

        Same shape as the #117, #606, #633, #632 and #1110 assertions above, and
        owed for the same reason: the bump is the whole of what
        `scripts/check-agent-record.py` changed for this row, so without an
        assertion naming the row the constant is the only artifact and 171 is
        plausible rather than checkable.

        This row carries one hazard the count cannot see. It is the SECOND
        upstream-pin row in the same section, beside `ENG-UPSTREAM-OMNI-PIN`,
        and the two are about different repositories -- `Lightricks/LTX-2` and
        `vllm-project/vllm-omni` -- that both answer for LTX-2.5. Folding either
        into the other leaves the matrix internally consistent and silently
        retires a pin, so naming BOTH is what makes 171 checkable.
        """
--- tests/scripts/test_agent_record.py:464-474 ---
"""The #634 rows and the 370 -> 372 bump are one semantic change.

        IndexTTS-2.5 is registered by vLLM-Omni as TWO architectures, a talker
        and an S2Mel decoder, so it moves the pin by two rather than one. That
        is the hazard worth pinning: a port described in prose as "a model" is
        the shape that lands one row and a bump of two, and the count alone
        cannot tell that from two rows landing. Both are named here, and both
        are asserted `INVENTORIED` rather than `SPIKE` — they are unclaimed and
        blocked on #633, and `SPIKE` would owe a `CLAIM-*` owner they do not
        have.
        """
--- tests/scripts/test_agent_record.py:491-514 ---
"""The #699 rows and the 373 -> 375 bump are one semantic change.

        dots3-note is the IndexTTS-2.5 shape again on a different lane: vLLM
        registers it as TWO architectures, `Dots3NoteForCausalLM` and its
        speculative head `Dots3NoteMTPModel`, so a port described in prose as
        "a model" moves the pin by two. Naming both is what makes 375 checkable
        rather than plausible.

        What this catches that nothing else does, measured: RENAMING the MTP row
        leaves the count at 375, touches no claim, and every other check stays
        green -- only this assertion goes red. That is the whole point of naming
        rows rather than counting them.

        The state assertions are deliberately weaker evidence, and the record
        says so rather than implying otherwise: mutating either row's lifecycle
        is already caught upstream of here by the claim-ownership and
        spec-structure rules (INVENTORIED -> SPIKE trips "SPIKE row has no
        CLAIM-* owner"; SPIKE -> ACTIVE trips the structured-spec requirement).
        They are pinned anyway because the asymmetry is intentional -- the
        target row is `SPIKE` with a committed spec and an owner, the MTP row is
        `INVENTORIED` because it is unclaimed and blocked behind the target's
        oracle and hardware gaps -- and a future refactor of those rules should
        not silently take the pin with it.
        """
--- tests/scripts/test_agent_record.py:529-570 ---
"""The #1978 row and the 377 -> 378 bump are one semantic change.

        Same contract as `test_dots3_rows_are_inside_the_model_ratchet` above,
        with the arithmetic going the other way. dots3-note and IndexTTS-2.5
        each moved this pin by TWO because vLLM registers two architectures for
        what prose calls one model. `Qwen4ExpForConditionalGeneration` moves it
        by ONE: its MTP head is an `mtp` block inside the same text config, not
        a separately registered architecture, so there is no
        `MODEL-SPEC-qwen4-exp-*` row and there must not be one. Naming the row
        is what makes 378 checkable rather than plausible.

        What this catches that nothing else does: renaming the row, or adding a
        second qwen4_exp row to "match" the two-row precedent, both leave the
        count reachable by a compensating edit elsewhere in the matrix while
        every other check stays green. Only an assertion that names the row
        goes red.

        The STATE pin is the weaker half of the evidence, stated rather than
        implied, and it has now been moved ONCE, deliberately and with an
        argument -- which is the movement it was written to make visible rather
        than to prevent. It was `READY` while the spec was committed and no
        product code had landed. W1 (#1981) landed the config surface: the
        architecture resolves, its config parses and validates, and the loader,
        forward and KV-cache spec refuse by name. That is a lifecycle change,
        and AGENTS.md Records requires the owning matrix row to move with it, so
        the row is `ACTIVE` and carries `CLAIM-MODEL-MM-QWEN4-EXP-W1`.

        The pin stays, at the new value, for the reason it was written: the
        structured-spec rules already catch `ACTIVE` without a spec and the
        claim-ownership rules already catch `ACTIVE` without a claim, but
        neither would notice a silent slide BACK to `READY` on a row that has
        shipped code, and neither names this row. Updating the value is not the
        same as removing the assertion -- everything below still names the row,
        still requires exactly one of it, and still requires it to live in
        `model-matrix.md`, which is what makes 378 checkable rather than
        plausible.

        The row is also beyond-pin in the strongest sense this file has carried:
        vLLM does not implement `qwen4_exp` at ANY revision, not merely after
        `555967922`. Its Upstream cell therefore names no pinned module or
        class, and the at-the-pin static invariants are untouched.
        """
--- tests/scripts/test_agent_record.py:594-631 ---
"""The #1998 row and the 378 -> 379 bump are one semantic change.

        Same contract as the qwen4-exp test above, and it guards the arithmetic
        against a *stronger* pull toward two-or-three. dots3-note and
        IndexTTS-2.5 each moved this pin by TWO because vLLM registers two
        architectures for what prose calls one model, and the OPEN vllm#53906
        would register THREE for GLM-5.3-Flash: `Glm5NextForCausalLM`,
        `Glm5NextForConditionalGeneration` and `Glm5NextMTPModel`. It still
        moves by ONE, because none of the three is registered at any vLLM
        revision and the only architecture a published artifact declares is
        `Glm5NextForConditionalGeneration`. The MTP head is `layers.45` inside
        the same checkpoint -- the transformers reference discards it at
        `modular_glm5_next.py:1235` -- not a separately registered architecture,
        so there is no `MODEL-SPEC-glm5-next-*` row and there must not be one
        until vLLM registers one.

        What this catches that nothing else does: renaming the row, or adding a
        second or third glm5_next row to "match" the upstream PR, both leave the
        count reachable by a compensating edit elsewhere in the matrix while
        every other check stays green. Only an assertion that names the row goes
        red.

        The state is pinned deliberately and is the weaker half of the evidence,
        stated rather than implied, for the same reason the qwen4-exp test gives:
        pinning it here means a future refactor of the structured-spec or
        claim-ownership rules cannot silently take this pin with it.

        It was `READY` when this test was written, on the stated premise that the
        spec was committed and no product code had landed. W7a (#2011) landed
        product code -- `scripts/convert-glm5-next-gguf.py`, the first thing on
        this row that is not a record -- so the premise expired and the pin moves
        with it to `ACTIVE`, in the same change that moves the matrix row. The
        assertion is NOT weakened: it still names one exact state, and a pin that
        followed the row automatically would assert nothing at all. What it stops
        catching is only the one transition it was updated for; it still goes red
        on a rename, on a second glm5_next row, and on any later state change
        made without touching this file.
        """
--- tests/scripts/test_agent_record.py:648-679 ---
"""The #2181 row and the QUANT 84 -> 85 bump are one semantic change.

        Same contract as the MODEL ratchet tests above, on the quantization
        matrix, and the arithmetic it holds is the one this scheme invites
        someone to get wrong.

        EXL3 ships in TWO on-disk layouts, and they are one scheme. The stock
        `turboderp/*-exl3` checkpoints store `{prefix}.{trellis,suh,svh}` with
        no rank segment, while the SparkInfer DeepSeek-V4 artifact stores
        `...{w1,w2,w3}.rank{r}.{trellis,suh,svh}` under its own declared
        `version: rank-sliced-deepseek-v4-v1`. Two readers, one format: the
        codeword window, the MCG codebook, the H128 sign vectors and the absence
        of scales are identical, and `vt::Exl3DequantLinear` decodes both. So the
        count moves by ONE. Splitting it into a native row and a rank-sliced row
        would be the dots3-note/IndexTTS-2.5 two-row shape applied where it does
        not belong, because there is one encoding here and not two.

        What this catches that nothing else does: renaming the row, or adding a
        second EXL3 row for the other layout, each leaves the count reachable by
        a compensating edit elsewhere in the matrix while every other check stays
        green. Only an assertion that names the row goes red.

        `ACTIVE` is pinned deliberately and is the weaker half of the evidence,
        stated rather than implied: the row is `ACTIVE` because W1a landed
        product code, even though that code is UNREACHED -- no production path
        constructs `Exl3LinearMethod` yet -- and pinning the state here means a
        later refactor of the claim-ownership or structured-spec rules cannot
        silently take this pin with it. The state was `SPIKE` in the first draft
        of this row while the spec's own `## Now` already said `ACTIVE`; a fresh
        review caught the divergence, and this assertion is what stops it
        recurring.
        """
--- tests/scripts/test_agent_record.py:733-740 ---
"""The #609/#610 rows and the 362 -> 369 bump are one semantic change.

        Mirrors `test_windows_release_row_is_inside_the_engine_ratchet`: name
        the rows the bump was taken FOR, so a count raised to silence a broken
        parse cannot look identical to a count raised because rows landed. Two
        of the seven are pinned, one per issue; seven near-identical assertions
        would add repetition, not force.
        """
````
<!-- matrix-cardinality-history:end -->
