# vllm.cpp status

**Roadmap-v1 reality audit (2026-07-31, `CLAIM-ROADMAP-V1-AUDIT`, `.agents/specs/roadmap-v1-audit-2026-07-31.md`):** an 8-lane code-grounded audit + adversarial DONE-verify (26 agents) vs the 18-row plan-of-record. Correctness + CPU-gated features are REAL; the dominant open gate across nearly all RI rows is **every-axis SPEED**; most GPU "DONE" proofs are DGX-recorded (not re-verifiable on the current box). **4 DONE claims refuted:** MM-video is near-tie not STRICT; Gemma-4 image is 16/18 near-tie not STRICT 18/18 (audio unbuilt); ROAD-V1-A (async code DONE but both SGLang-floor arms never ran, 27B 114/124 coin-flip not clean); C6 async-serving (SERVE-ASYNC-LLM still GATING). D5 LoRA is an unwired standalone CPU brick.

This is the per-capability status ledger. [README.md](../README.md) is the
user-facing landing page and carries only headline state; every capability's
current lifecycle stage, its active gaps, and its next gate live here.

Capabilities are labelled: *correctness-complete* (token-exact vs the vLLM
oracle), *speed-pending* (correct, throughput work in progress), *build-only*
(compiles for a target with no runtime proof here), and *hardware-blocked*
(cannot run on the hardware available).

Chronology and raw evidence live in the structured
[state manifest](../.agents/state.csv), immutable events,
[parity ledger](../.agents/parity-ledger.md), area matrices, and
[docs/BENCHMARKS.md](BENCHMARKS.md). This file keeps one binding current-state
line per capability, not a run log.

## Parity pin

The reference numbers here are measured against the pinned vLLM oracle on GB10,
greedy, same workload, same tokens. Where a claim cannot be measured on the
hardware here, it is stated as such rather than implied. **The parity pin
advanced 2026-07-26 to vLLM 0.26.0.dev0 (`55596792`) + transformers 5.14.1**
(from 0.25.0); correctness was re-validated bit-identical on the new oracle
(zero golden drift; see [.agents/specs/pin-advance.md](../.agents/specs/pin-advance.md)),
so the token-exact claims hold against the new pin. Every speed figure citing
"vLLM 0.25.0" was measured against the preserved rollback, and NOT only before
the advance: the benchmark harness hard-enforced 0.25.0 and *raised* on the pin
until 2026-08-12, so no run through it could have used the pin
([#520](https://github.com/mudler/vllm.cpp/issues/520)). **That re-benchmark
has now RUN.** The 2026-08-13 series selects the oracle by explicit path
(`0.23.1rc1.dev1511+g555967922`, FlashInfer `0.6.15.post1`), asserts that
identity per leg, runs it graphed and passes `--language-model-only`, which the
canonical driver never did
([#414](https://github.com/mudler/vllm.cpp/issues/414)). Every figure citing
"vLLM 0.25.0" is therefore SUPERSEDED and OPTIMISTIC on two counts that both
flattered us, not merely stale: nothing is withdrawn, and the values keep their
attribution.

**Clock attribution (`BENCH-ASSERT-CLOCK-STATE`, ACTIVE, 2026-08-12,
[#543](https://github.com/mudler/vllm.cpp/issues/543)):** the SM clock on
`dgx.casa` differs BETWEEN BOOTS without throttling — a 12.79% delta repriced a
byte-identical `marlin::Marlin` by +9.65%, larger than either deficit that
comparison ranked. The harness now records the clock window, `clocks.max.sm`,
the applications clock, the active throttle reasons, persistence mode and the
**boot id** per leg, refuses a cross-boot pair, and voids a run whose window was
over-spread or barely observed. The cross-boot override waives the boot id and
nothing else ([spec](../.agents/specs/bench-assert-clock-state.md)); the sample
floors and the comparison it keeps are derived there. Every figure recorded
before this date carries no clock attribution: not withdrawn, not restated. The
first attributable grid has now landed (2026-08-13): clocks pinned with
`sudo -n nvidia-smi -lgc 2190` while holding `$HOME/gpu.lock`, under an
always-fires reset trap, delivering a flat 2184 MHz across every timed window on
one boot id (a leg reads n=861, min 2158, med 2184). GB10 enumerates no
`SUPPORTED_CLOCKS`, so 2190 is the frequency the worst observed boot sustained
flat.

## Capability status

Cold start: `MEASURED`. Load (#150): 27B bf16 loads **1.54x warm / 1.61x cold**, moving
**100.2 -> 81.3 GiB** ([detail](../.agents/specs/load-direct-upload.md)). Startup (provisional)
**6.07x** ([detail](../.agents/specs/startup-latency-axis.md)).

Releases: v0.0.2 published eight primary server archives plus checksums,
provenance, and two generated indexes (26 assets) from
`7020de93652ca920424a10ac5255b34810dd2f24` in run `31466516224`. Native Windows
CPU/Vulkan W14-W16 are implemented for v0.0.3-pre.1, but hosted MSVC evidence,
the merged-SHA ten-tuple dry run, prerelease publication, and its 32-asset audit
remain pending; no Windows ZIP is published.
<!-- ENG-RELEASE-WINDOWS: state=ACTIVE publication=pending artifact=unpublished -->

Container images (#170): the cpu (amd64) and cuda (arm64) lanes pass their gates. One SBSA
arm64 image serves BOTH families: GB10 `sm_121a`, and Jetson AGX Orin `sm_87`
where Qwen3-0.6B generates with the GPU at 95-97% (Tegra needs
`--runtime nvidia --gpus all`). Thor is unprobed; nothing is published.
See [RELEASES](RELEASES.md).

Protocol (2026-08-09): `776c56f1` has 157 imports = 3,231,342 exact bytes;
append preserved prior 156 wrappers/rows. Archive/new raw-row mutation guards bind.
Tests: 95: validator/core 44 (checker 20 + core 24), NOW 18, migration 22,
cutover 11.

Protocol repair (2026-08-08): release AST pins pass 30 tests on Python
3.12/3.13; Gemma-4 MoE is known drift pending the shared merged-GeGLU fold;
embeddings #137 is landed/partial, not an active claim. No runtime change.

Protocol live position (#374): `DONE` at implementation merge `dbd0d51c`.
Lifecycle changes still update STATUS and BENCHMARKS, while the moved row spec's
`## Now` replaces the per-row write to `.agents/NOW.md`. Runtime and performance
are `VOID`; no product behavior changed.

Supported-model registry guard (2026-08-06): the public per-architecture list in
[FEATURES](FEATURES.md) is CI-bound to the C++ registry by
`scripts/check-supported-models.py` (+ mutation test), so the 37
`REGISTER_VLLM_MODEL` architectures and the FEATURES rows cannot drift.

GCC 12 production-library maintenance (2026-07-31): the two known `-Werror`
blockers in GGUF prefaulting and KV-offload temporary-file naming are fixed
without behavior or lifecycle changes. The production library and focused
tests build clean; the all-target build remains PARTIAL on two unrelated
test-only `-Wrestrict` diagnostics, recorded in [BENCHMARKS](BENCHMARKS.md).

cuBLAS invocation-parity CI guard (2026-08-04): new
`scripts/check-gemv-invocation-consistency.py` (+ mutation test, wired into CI)
locks the dtype-faithful cuBLASLt op-contract the Laguna `gemvx<bf16,FLOAT>`-vs-
`gemvx<bf16,bf16>` template bug exposed — the C/D layout stays `out_type` (never a
hardcoded `CUDA_R_32F` literal) and requestedAlgoCount is the named
`kGemvHeuristicAlgos`. The byte-exact `kGemvHeuristicAlgos` refactor of
`src/vt/cuda/cuda_matmul.cu` is type-safe and zero-behavior; CUDA build-verified on dgx
(clean -Werror; re-verified @`812de8ca`).

vllm.cpp implements an intentionally focused subset of vLLM, held to
token-for-token correctness against the pinned oracle.

| Capability | State | Notes |
|---|---|---|
| Qwen3.6-27B (NVFP4) text generation | Correctness-complete; speed is CHECKPOINT-dependent | Token-exact GB10 on both. `unsloth` @`890bdef7` beats vLLM every c (1.007-1.045x), 115/124; `nvidia` @`0893e160` **flat 0.937-0.956 c1-c32** (#349; 0.838 void) |
| Qwen3.8-2.4T-A95B (`UD-Q1_0`, 370 GiB) | **Loads and generates on ONE 119 GiB GB10**; speed is the gap | Resident 62 GiB; 66.7 s/tok, streaming OFF. Streaming lands but its decode figure is VOID: the step clock had no caller, so the cache died in token 3 ([#912](https://github.com/mudler/vllm.cpp/issues/912)) |
| Qwen3.6-35B-A3B (NVFP4, GDN MoE) | Correctness-complete; **canonical 0.918-0.972x c1-c32** @`348c265d` (first c16/c32), STALE as of 2026-08-12 (+136 src commits then, incl. a +2.05% c8 lever); regrid owed | Token-exact SYNC+ASYNC; `VT_ASYNC_DEVICE_MIRROR` ON fixes async batch-1 token-0 degeneration. Router warp kernel (#378) device-gated 315/315 both arms, kernel 1.363x, step-level not separable |
| Qwen3.6-35B-A3B (published BF16, GDN MoE) — TEXT | Correctness-gated 2026-08-15 (#740, #864, both `DONE`); **no throughput, latency or memory number exists for this checkpoint, and none is claimed** | Greedy vs the pinned oracle @`995ad96e`: **6/7 prompts STRICT 16/16**; the 7th is one exact logit tie (`0.0 mnats`) our argmax breaks toward the higher id (#910). SACRED 3/3, goldens byte-identical |
| Qwen3.6-35B-A3B (BF16) — IMAGE / VIDEO | Implemented, **NOT gated** (#891); row stays `PARTIAL` | The 333 `model.visual.*` tensors load and the tower computes (`[1,28,28]`→`[196,2048]`, finite, absmax 2.08) on sm_110 FALLBACK attention. The token-exact mm gate vs the oracle is OWED |
| Qwen3 / Qwen2 dense (BF16) | Correctness-complete, speed-pending. Async-serving P0 FIXED (#323: the decode graph replayed stale HOST ids, now declines while the mirror is live; async 7/7 incl. Llama/Mistral/InternLM2) | Near-tie-robust token-exact vs vLLM (Qwen3-0.6B, Qwen3-4B); c1 effective parity, c8 decode residual. **Async device-mirror (`ROW-SERVE-ASYNC-DENSE-MIRROR`, `f9c969ae`): the #31 fix ported to the classic dense family, dgx-VERIFIED.** The shared dense `EmbedInto` (qwen3.cpp) raced the async combine's device input-ids write against a stale host upload → token-0 degeneration on the depth-2 AsyncLLM serving path (quant-independent). `EmbedInto` now consumes the device override published by `ForwardQwen3ForCausalLM`'s `DeviceTokenIdsScope` (27B-dense template); gate `test_qwen3_dense_async_serving` RED on `VT_ASYNC_DEVICE_MIRROR=0`, GREEN default, byte-identical mirror-off. dgx GB10: async gate RED→GREEN 0.6B+4B, SACRED 0.6B+4B 184/184 unchanged (byte-neutral sync path), memcheck 0 errors; Yi30/Qwen3-8B-MXFP4 default-config e2e coherent + 3/4 token-exact (p2 = oracle-ratified near-tie, gap 0.0000), closing the QUANT-CT-MXFP4 async-default residual. RESIDUAL: sibling InternLM2/Mistral/Llama scope one-liner; W4 bench RAN; FA2 GQA-swap default-ON, c2-c8 <1.0x. `FLASH-PTXAS` #82: codegen at PARITY (no ptxas lever); gap=engine context. **D1 (2026-07-31, `CLAIM-D1-BF16-MERGED-QKV`): the bf16 merged-QKV path (`Qwen3QkvMergeEnabled`/`VT_QWEN3_QKV_MERGE`) is now default-ON** — one `vt::MatmulBT` over the merged `[qdim+2kdim,H]` owner + a contiguous `vt::QkvSplit` (OLMo-2 exemplar), replacing three per-shard GEMMs. Bit-exact GEMM math (A/B unit `test_ops_qkv_merge` byte-identical, RED-first); the wider-N cuBLASLt K-reduction flips the 0.6B genuine bf16 near-tie so the SACRED 0.6B golden was regenerated (all tokens within the near-tie band, max 0.125 nats), while Qwen3-4B is byte-neutral (0 diffs, stays STRICT). Re-gated 0.6B 16/16 + 4B 16/16; consistency/launch-count fold (measured NEUTRAL on 4B decode), no new throughput owed |
| Qwen3.5-4B BF16 direct-load on discrete CUDA | Correct; throughput/host PSS ahead, acceptance `PENDING`; latency/VRAM open; sampled-token lever CLOSED, next is the TTFT intake/prefill split ([#527](https://github.com/mudler/vllm.cpp/issues/527)) | Atomic pretoken exact. Ratios: tput 1.0283x; TTFT/TPOT/E2E 1.0853/1.0165/1.0288x slower; VRAM +118.7 MiB. GDN local stack retained ([data](bench-evidence/qwen35-4b-sm120-main-20260807.md)) |
| Qwen3-Coder-30B-A3B MoE (BF16) | Correctness-complete, speed-pending | Near-tie-robust token-exact 6/6; 11 of 16 binding grid cells at or above vLLM. **D1 (2026-07-31): inherits the default-ON bf16 merged-QKV via the shared dense `AttnBlock` — byte-neutral (0 token diffs, golden UNCHANGED); re-gated 6/6** |
| Llama-3.x dense (BF16) | Correctness-complete, speed-pending | Near-tie-robust token-exact 16/16 (Llama-3.2-1B); llama3 RoPE scaling |
| Mistral dense (BF16) | Correctness-complete, speed-pending | Paged-engine token-exact 16/16 (Mistral-7B-v0.3) |
| OPT (learned pos-emb, cross-family) | Correctness-complete, speed-pending | Strict token-exact 6/6 (OPT-125m); additivity canary across model families |
| DeepSeek-V2 MLA | Correctness-complete, speed-pending | Token-exact 8/8 (DeepSeek-V2-Lite); 0.86-0.95x output rate, TTFT faster at c4/c8. A2+A5 MLA norm-rope fold default-ON (`VT_MLA_FUSED_NORM_ROPE`, bit-exact rollback, SACRED 8/8 unchanged; forensics in benchmark-record) — kimi_k3/kimi-linear inherit it |
| GLM-4 dense (sandwich norms, partial rope) | Correctness-complete, speed-pending | Token-exact 16/16 (GLM-4-9B-0414); first GLM-family model; partial interleaved RoPE + Gemma2 sandwich norms + biased qkv |
| GLM-4.7-Flash (MLA + GLM MoE) | Correctness-complete, speed-pending | Token-exact 8/8 (GLM-4.7-Flash, 31.2B); reuses the DeepSeek-V2 MLA stack; first e2e coverage of the q_lora query branch + noaux_tc sigmoid router with routed-scaling |
| Kimi-Linear-48B-A3B (KDA + NoPE-MLA + MoE hybrid) | **RUNNER FOLD LANDS (ROW 7 §21, #122): engine==CLI 128/128 byte-identical; golden 122/128 (near-tie profile); FA2 MLA default-ON; `vllm_complete_tokens` (ABI v13).** Grouped-router top-k block-parallel (byte-identical); no binding speed number: ckpt is tiktoken-only, so no warm-server harness. STRICT stays CLOSED. Server 19.0 tok/s wall (~0.90× vLLM floor) = speed open | paged suite 8/8·206; SACRED post-fold 35B 315/315 + 27B 235/235; thin ABI client (ratchet 8) |
| Nemotron-3.5-Lightning-30B-A3B (Mamba2 + GQA + relu2 MoE) | **Host gate PASSES 96/96 `STRICT PASS`; GB10 fixed ([#1157](https://github.com/mudler/vllm.cpp/issues/1157)), sm_121a re-run pending** | ABI-only driver; G-SAFE `num_reqs <= 1`. The 23 FP8 Mamba2 blocks reach the device (A2-Q1, [#810](https://github.com/mudler/vllm.cpp/issues/810)); `lm_head` is the last host arm |
| Gemma-3 dense (GeGLU, dual rope, sandwich norms) | Correctness-complete, speed-pending | STRICT token-exact 48/48 greedy (gemma-3-1b-it); first Gemma-family model; GeGLU (gelu_pytorch_tanh) + dual per-layer RoPE theta + Gemma-RMSNorm sandwich norms + sqrt(hidden) embed-scale + query_pre_attn_scalar scaling |
| Gemma-2 dense (attn + final logit soft-cap) | Correctness-complete, speed-pending | Near-tie-band 48/48 (gemma-2-2b-it): 44/48 strict on vLLM's greedy + 4/48 at 0.0-nat ties in vLLM's own logits; proves the attention + final logit soft-cap primitives (attn_logit_softcapping 50 + final 30); the inverse of Gemma-3 (both soft-caps, no QK-norm) |
| Gemma-1 dense (the original Gemma) | Correctness-complete, speed-pending | STRICT token-exact 48/48 greedy (gemma-2b); two fused norms/layer, head_dim scale, GeGLU + sqrt(hidden) embed-scale, tied lm_head; no soft-cap/QK-norm/sliding. **D1 (2026-07-31): the whole Gemma family (1/2/3/4) folded to the default-ON bf16 merged-QKV descriptor (`MergedQkvEnabled`); re-gated Gemma-2 SACRED 48/48 (global+sliding) + Gemma-4 STRICT 32/32 — its existing gate held** |
| OLMo-2 dense (pure post-norm, full-width QK-norm) | Correctness-complete, speed-pending | Token-exact 16/16 (OLMo-2-0425-1B); first OLMo-family model; ZERO new compute kernel (pure post-norm `norm_after` + full-width QK-norm reuse existing ops); real ByteLevel-tokenizer gate (no BOS). **D3 (2026-07-31, `CLAIM-D3-OLMO2-FULLWIDTH-QKNORM`): the full-width q/k RMSNorm + RoPE preamble folds onto the SHARED FusedChain catalog** — new sibling recipe `kAttnQkNormRopeFullWidth` (the `norm_full_width` shape-param variant of Qwen3-dense's `kAttnQkNormRope`; `FStep.norm_full_width` additive, default false ⇒ every existing recipe byte-identical). Composite-only realization (`fast_op=kNoFastOp`) is byte-exact to the pre-fold standalone `RmsNorm(qdim)+RmsNorm(kdim)+RopeFromCache` sequence — CPU RED-first `test_ops_fused_chain` 10/10·379 GREEN (byte-exact, full-width proven distinct from per-head). Detail in `.agents/benchmark-record.md`.
| Phi-3 / Phi-4 dense (Llama subclass, LongRoPE) | Correctness-complete, speed-pending | Token-exact (near-tie-robust) 16/16 (Phi-4-mini-instruct: 7 strict + 9 near-tie, 0 divergent) by the ratified root-divergence gate; bigger-dense STRICT anchor phi-4 (14B) 14/16 fully token-exact + 2 exact-tie (0.0 nats); pre-fused qkv/gate_up loader + LongRoPE cache bit-identical to vLLM's |
| Granite-3 dense (Llama + 4 scalar multipliers) | Correctness-complete, speed-pending | Token-exact 16/16 (granite-3.3-2b-instruct); first IBM Granite model; ZERO new compute kernel; embedding/residual/attention/logits multipliers (attention scale 1/64, not 1/sqrt(head_dim)) threaded over the shared dense path |
| StableLM dense (LayerNorm, partial rope, qkv bias) | Correctness-complete, speed-pending | Token-exact 16/16 (stablelm-2-1_6b): 14/16 strict on vLLM's greedy + 2/16 at bf16 near-ties (max gap 0.438 nats), 0 forward-divergent; first Stability AI model; ZERO new compute kernel (nn.LayerNorm with weight+bias + partial NeoX rope 16/64 + merged qkv bias, all reuse existing ops) |
| OLMo-3 dense (dual rope, interleaved sliding window) | Implemented, oracle-blocked | Loads + runs in our engine (dual rope: plain sliding + YaRN full-attn, per-layer sliding window); no SACRED gate: vLLM 0.25.0 oracle cannot run OLMo-3-1025-7B (`KeyError: 'rope_theta'`; transformers 5.13.1 nests `rope_parameters` per layer-type, no flat `rope_theta`; run-verified W0 2026-07-26) |
| Laguna-S-2.1 MoE (`LagunaForCausalLM`, 118B/8B) | **BINDING 2026-08-04: was 87% of vLLM (37.55 vs 43.10, same-tool nsys)**; root cause was bf16 projections on UNIFIED/ATS host memory, and device-resident staging (byte-exact) gives 44.6, parity+ vs 43.1, default-ON | 48 layers (12 global + 36 SWA-512), 256 routed top-10 + 1 shared expert, per-head softplus attn out-gate, sigmoid `noaux_tc` router, dual per-layer RoPE, GQA 8 KV / 128 head-dim, 1M ctx. History: benchmark-record |
| InternLM2 dense (fused-`wqkv` interleaved split) | Correctness-complete, speed-pending | Token-exact 16/16 (internlm2-chat-1_8b): 12/16 strict + 4/16 bf16 near-tie (max gap 0.0 nats), 0 divergent; first InternLM model; ZERO new compute kernel (reuses the Llama dense forward; the only delta is a loader-side de-interleave of the fused `wqkv`, which packs q/k/v interleaved by KV-group) |
| MiniMax-H3 (`MiniMaxH3DiTModel`, video+audio DIFFUSION) | **ABI v12 ONE SURFACE; device selector uses generic `DeviceType`; DSR 32.** t2va+fl2va COHERENT; bf16 shards STREAM | ref2va ckpt fidelity §8.12; encoder A/B §8.15; GB10 re-verify residual; CPU fold 6/137 (one queue + device provenance mutation-gated) |
| LTX-2.5 (`LTX2VideoTransformer3DModel`, video+audio DIFFUSION) | **L1-L9c landed (#435).** 21.00B / 48 blocks. `VideoEngine` seam + ABI **v18**, DiT forward (CPU f32 parity, bf16 device-resident), Gemma-4 TE, both VAEs, connector, pipeline, NVFP4/FP8, keyframe bias (#658) | BOTH shipped DiTs load inside the contract; one runs device-resident on GB10. Caption projection on `vt::MatmulBT` (#1208); IC-LoRA fusion on `vt::Matmul` (#1202), 143x, residual now the add-back (#1254). Render OWED |
| MiniMax-Music3 (`MiniMaxMusic3ForConditionalGeneration`, text-to-MUSIC) | **`ACTIVE`: W0-W7 landed; every stage including the 8.6B LM forward is implemented and gated (#672).** Oracle is the OPEN diffusers PR #14456 `c6da9936` | GGUF arms for 4 components owed. LM gated in a control; HTTP OBSERVED (#852). PARTIAL device arm, Thor sm_110 (#672): 8.6B LM + 2.4B fp32 DiT (§14). Depth 4.45x, wall 2.74x, WAV byte-identical (§16). No reference number |
| Command-R / Cohere dense (`CohereForCausalLM`) | Implemented, gate-blocked | ZERO-new-kernel port grounded in vLLM `commandr.py`: weight-only Cohere LayerNorm + GPT-J full-width RoPE + PARALLEL residual + `logit_scale` + tied embeddings, all reuse; compiles, links, self-registers. No SACRED gate yet (real checkpoints HF-gated, ungated ones tiny-random, GPU box disk-full); oracle run-verified at W0. See docs/BENCHMARKS.md |
| Phi-1 / Phi-2 dense (`PhiForCausalLM`, parallel residual) | Correctness-complete, speed-pending | Token-exact 16/16 (microsoft/phi-2): 9/16 strict + 7/16 bf16 near-ties (max gap 0.25 nats), 0 forward-divergent; the OLDER Microsoft Phi arch, DISTINCT from Phi-3/Phi-4; ZERO new compute kernel (GPT-J parallel residual, LayerNorm-with-bias, biased qkv/dense, partial NeoX rope 32/80, non-gated NewGELU MLP reusing `vt::GeluTanh`, untied biased lm_head); F16 dtype-aware loader |
| MiniCPM dense (`MiniCPMForCausalLM`, three scalars) | Correctness-complete, speed-pending | Token-exact 16/16 (openbmb/MiniCPM-2B-sft-bf16): 10/16 strict + 6/16 bf16 near-ties (max gap 0.0 nats), 0 forward-divergent; first OpenBMB MiniCPM model; ZERO new compute kernel (the Llama/Granite dense forward plus three scalars: scale_emb, scale_depth/sqrt(layers) residual, dim_model_base logit scaling), tied lm_head; `.bin`-only weights converted to safetensors via trusted torch |
| MiniCPM3 (`MiniCPM3ForCausalLM`, MLA + three scalars) | Correctness-complete, speed-pending | Token-exact 16/16 (openbmb/MiniCPM3-4B): 13/16 strict + 3/16 bf16 near-ties (max gap 0.0 nats), 0 divergent; the first MLA-attention MiniCPM; ZERO new compute kernel (the MiniCPM three scalars with attention swapped GQA to DeepSeek-style MLA, reusing the landed DeepSeek-V2 MLA block; DeepSeek-V2 re-gated 8/8 unchanged); tied embeddings, `.bin`-only weights converted via trusted torch |
| Yi (Llama architecture) | Correctness-complete, speed-pending | Token-exact 16/16 (01-ai/Yi-Coder-1.5B-Chat): 13/16 strict + 3/16 bf16 near-ties (max gap 0.125 nats), 0 divergent; modern Yi adopted the Llama architecture (`architectures: ["LlamaForCausalLM"]`), so it runs on the existing Llama path with zero code changes; confirms a distinct non-Llama-branded checkpoint (64000 vocab, RoPE theta 1e7) loads and decodes correctly |
| InternLM3 (`InternLM3ForCausalLM`, Llama alias) | Correctness-complete, speed-pending | Token-exact 16/16 (internlm3-8b-instruct): 14/16 strict + 2/16 bf16 near-ties (max gap 0.0 nats), 0 divergent; a plain Llama architecture in vLLM 0.25.0 (registered as a one-line Llama alias; dynamic-NTK RoPE factor 6.0, GQA kv=2, untied lm_head), not InternLM2 + sliding window; zero forward or loader delta |
| Long-context RoPE + sliding-window attention | Correctness feature-positive on GB10, speed-pending | Shared scaled-RoPE (YaRN, Llama-3, Phi-3/4 LongRoPE, dynamic-NTK) + sliding-window attention, GPU-gated vs the vLLM 0.26 oracle: LongRoPE (Phi-4-mini) + llama3 (Llama-3.2-1B) + dynamic-NTK (InternLM2) 16/16, sliding-window (Gemma-2/Gemma-3) 48/48, operator local-mask kernel positive. YaRN and chunked-local model e2e are reachable-blocked (no cached vehicle); speed pending |
| LoRA / multi-LoRA adapters | In progress (W1+W2 CPU runtime), not yet usable end-to-end | Highest-demand missing feature. W1+W2 landed the punica CPU brick (shrink/expand, `-1`-slot skip) plus packed adapters and the wrapped layer family (merged qkv/gate_up slices, TP slicing, embedding and logits LoRA). CPU-gated vs double references, RED-first: `test_punica_cpu` 8/8 + `test_lora_layers` 16/16 (4,647 assertions); the fully-sharded (S-LoRA) apply REFUSES at tp_size>1 instead of returning a partial delta. W3-W7 (mapping, adapter load, manager, endpoints, GPU kernels + model gate) are in `.agents/specs/lora-adapter.md`. No model can be served with an adapter yet |
| Safetensors loading | Supported | Both gate models plus every registered dense/MoE family |
| GGUF loading (F32/F16/BF16/Q4_0/Q8_0/Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/IQ2_XXS/IQ3_XXS/IQ2_S/MXFP4/NVFP4) | Supported; compute-in-quant (keep-quant) on CPU AND now CUDA for the six K-block encodings PLUS Q2_K/IQ2_XXS/IQ3_XXS (DeepSeek-V4 W8, 2026-07-29 - the FIRST CUDA keep-quant GGUF k-quant GEMM `KERNEL-QUANT-CIQ-GEMM-CUDA`, MMVQ-style dequant-in-kernel, GB10-gated 92401/92401 vs the CPU oracle, so a CUDA runner keeps blocks compressed and dots them on the GPU instead of the ARM cores); **NVFP4 now COMPUTES IN FP4 on CUDA for the dense-MLP and full-attention projections (2026-07-29, `CLAIM-GGUF-NVFP4-COMPUTE`), no longer materialize-only** | Weights in six block encodings stay compressed from file to matmul on CPU (no BF16 expansion). NVFP4 (ggml type 40) DEQUANTIZES, including the per-tensor (per-expert) `<stem>.scale` sidecar the container keeps outside the blocks; gated BIT-EXACT against the compressed-tensors NVFP4 path on real Qwen3.6-27B bytes from both containers. **It no longer expands to bf16 on CUDA:** an NVFP4 matmul/expert weight is REPACKED at load into the same (`weight_packed [N,K/2]`, `weight_scale [N,K/16]`) operand pair the compressed-tensors path produces - a pure byte permutation, gated BYTE-IDENTICAL against that container - and the existing `vt::MatmulNvfp4*` kernels run on it, so no new kernel exists and no numerics are re-derived. Covers the dense MLP + full-attention q/k/v/o and the MoE shared/routed experts; the GDN `in_proj_*` family and `ssm_out` still expand (the V-head reorder rewrites their layout) and a CPU build still expands everything - the documented `part` subset. **MEASURED GB10 (2026-07-29), same-binary A/B, 2 reps/arm:** peak RSS **50.8 -> 25.7 GiB**, load-and-generate **1:58 -> 0:41**; the 256 moved projections cost 35 840 MiB expanded vs 10 080 fp4-resident (3.56x). **The safetensors-sibling divergence CLOSES:** the fp4 arm is token-IDENTICAL over 24 greedy tokens where the same binary's bf16 arm diverges at index 4. REPORTED, not gated: the containers are not the same model - the GGUF NVFP4-quantizes 192 GDN `in_proj` tensors the safetensors keeps BF16 (mean rel. weight error ~0.18) and their activation global scales differ, so identity is not guaranteed and a cross-container throughput arm is invalid. SACRED gates unmoved: `test_qwen27_paged_engine` 235/235, `test_qwen36_paged_engine` 315/315. **The MoE (35B) stacked-expert arm is HARDWARE-GATED too (2026-07-29)**: the real 35B A3B NVFP4 GGUF loads and generates through the fp4 path, its 120 routed-expert stacks x 256 experts repack to the modelopt safetensors' operands with ZERO differing bytes over 840 sampled (tensor, expert) slabs, and all 840 per-expert `<stem>.scale[e]` are bit-identical to that expert's `weight_scale_2` - the scale INDEXING, mutation-proved against a `scales[0]`-for-all and an expert-0-slab-for-all mutant. Same-binary A/B: peak RSS 68.5 -> 22.7 GiB (3.01x), load-and-generate 1:51.9 -> 0:28.8, tokens IDENTICAL (the 35B routed experts run the W4A16 grouped GEMM in both arms). Recorded OPEN: this case's 24-token greedy stream is NOT run-to-run stable (1 of 3 `use_a16` and 1 of 4 reference runs differed), so the binding results are the weight-level byte identity and the residency audit, not token-exactness; `test_qwen36_paged_engine` is token-exact at ITS engine params, so the instability belongs to this case's configuration and attributing it is owed work. It also FIXED a latent defect the MoE arm made reachable: the two fp4 fused MoE blocks issued the router GEMM assuming the safetensors `[K,N]` gate layout and threw `matmul: inner dims mismatch` on the GGUF's `[N,K]`; `MoeRouterLogits` now branches on `nk` (inert for safetensors, SACRED unmoved). **Q2_K (id 10) + IQ2_XXS (id 16) DEQUANTIZE (2026-07-29, `CLAIM-DSV4-GGUF-LOADER`):** the ~2-bit types the single-Spark `DeepSeek-V4-Flash-GGUF UD-IQ2_XXS`/`UD-Q2_K_XL` vehicles use, ported 1:1 from llama.cpp `ggml-quants.c` (`iq2xxs_grid` codebook + signs; Q2_K nibble sub-scale/min), unit-gated on hand-derived bytes (`test_gguf_dequant` 15/15). Dequant-only (no vec_dot -> expand-bf16). A V4-GGUF model cannot RUN yet: the name map (tensor-manifest-blocked) + the V4 forward (W3-W8) remain. **Multi-shard split GGUF READING (2026-08-03, `CLAIM-GGUF-SPLIT-SHARDS`):** `GgufFile::Open` now transparently stitches llama.cpp `gguf-split` shards (`...-00001-of-00003.gguf`) — every shard mmap'd, tensor tables merged, KV metadata from shard `00001`, sibling mappings kept alive by the primary so keep-quant mmap-borrows stay valid across shards (`OwnsSpan` is shard-aware); `VT_GGUF_NO_SPLIT=1` opts out; unit-gated (`test_gguf` split-merge / no-split / count-mismatch cases, 33/33 local). This unblocks the real 3-shard `unsloth/DeepSeek-V4-Flash-0731 UD-IQ2_M` (~91 GiB), whose layout is the NATIVE `deepseek4` arch — per-block `ffn_gate_tid2eid` hash tables (hash layers 0/1/2) + `hc_*` MHC + DSA compressor/indexer are all PRESENT (name-map 1328/1328), `vocab_size` derives from `token_embd` — NOT a standard llama.cpp conversion, so no loader-layout change is owed. It now loads THROUGH 1324/1328 tensors; the sole remaining gap is 4 routed-expert slabs quantized with IQ2_S (id 22, ×2) + MXFP4 (id 39, ×2) — encodings we have GGUF block traits for but no keep-quant vec_dot, so they hit the expand→dequant path which lacks them. Expanding those 4 expert tensors to bf16 would add ~17 GiB (~106 GiB total → OOM-reboot risk), so the memory-safe fix is an IQ2_S+MXFP4 keep-quant kernel (CPU dequant dispatch + `iq2s_grid` + CUDA `DotSuperblock<kIQ2_S/kMXFP4>`), spec'd as the next brick **IQ2_S (id 22) + MXFP4 (id 39) DEQUANTIZE + KEEP-QUANT on CPU (2026-08-03, `CLAIM-DSV4-UDIQ2M-QUANT`, off-GPU):** the extra per-tensor "dynamic" encodings the `unsloth/DeepSeek-V4-Flash-GGUF UD-IQ2_M` checkpoint mixes into its last routed-expert slabs (IQ2_S `ffn_gate/up` dotting Q8_K, MXFP4 `ffn_down` dotting Q8_0) — ported 1:1 from llama.cpp `ggml-quants.c` @ 237ad9b96 (`iq2s_grid` 1024-entry codebook + DIRECT sign bytes; MXFP4 `kvalues_mxfp4` + `e8m0_to_fp32_half` micro-scaling, distinct from the compressed-tensors `E8M0ToF32` NVFP4 path). CPU dequant + keep-quant `vec_dot`, unit-gated on hand-derived golden bytes (`test_gguf_dequant` 17/17), an INDEPENDENT f64 dequant-then-dot + GEMM NMSE (`test_ops_quant_dot` 19/19), and keep-quant routing (`test_gguf_keep_quant` 37/37) — all CPU-green, so UD-IQ2_M's four previously-`unsupported ggml type 22/39` slabs now load COMPRESSED (no ~17 GiB bf16 expansion that OOM-reboots the box). CUDA: the IQ2_S device `DotSuperblock<WType::kIQ2_S>` is wired into the Q8_K grouped-MoE GEMM and now **CUDA-BUILT + LINKED on GB10 (sm_121a, CUDA 13.0, `-Werror`, 2026-08-03 integration)** — it compiles clean and the merged binary links; MXFP4's device dot (`DotMXFP4`) is written but NOT wired (Q8_0-activation needs a separate 32-block GEMM) so it is marked `[[maybe_unused]]` to keep the ready math without tripping nvcc #177-D, and on GPU MXFP4 CPU-fallbacks like Q4_0/Q8_0. The V4-GGUF forward + a real UD-IQ2_M GPU load/coherence run are owed |
| AWQ / GPTQ quantization | W0 spike + W1 CPU INT4 dequant primitive; not yet loadable end to end | INT4 unpack+dequant-to-bf16 for BOTH community formats, mirroring vLLM 1:1 (AWQ reverse-order `awq_triton.py`; GPTQ `qdq_4.cuh` with zero_offset v1/v2 + act-order g_idx). Unit-gated RED-first (hand-computed known bytes + double-precision roundtrip). NOT wired to a loader, no GPU Marlin compute, no model run yet: config recognizer (W2), Marlin GPU GEMM riding the vendored NVFP4 Marlin (W4), CPU e2e (W3), GPTQ 8/2/3-bit (W5) and MoE (W6) are named next bricks. See [.agents/specs/awq-gptq-quant.md](../.agents/specs/awq-gptq-quant.md) |
| MXFP4 (compressed-tensors `mxfp4-pack-quantized`) | Compute PROVEN (#38); GQA-swap ON (#49); decode-graph+gate_up FUSION default-ON. `VT_MARLIN_DENSE` DEFAULT-ON (`KERNEL-MARLIN-DENSE-EXEC`): dense marlin 48-CTA byte-faithful (32B 0.000, 263/263), binding beats #51 every axis (c1 1.020, c8 0.969, mem 2.63x). **`QUANT-CT-MXFP4-FINAL-STACK` TERMINAL — both last levers exhausted: num_splits cap `VT_FA2_NSPLITS_CAP` gated-OFF (c1-only, self-corrects@c8; 32B strict char-identical); glue folds via `vt::FusedChain`; `FLASH-AUDIT` #68: c8 flash gap +12.5us/call is occupancy/L2-bound; `-use_fast_math` TRIED, REGRESSES flash (168.8→189.8), rejected. c1 1.020x PASS, c2-c8 0.962-0.969.** state.md | Shared with DeepSeek-V4-Flash + Kimi-K3 MXFP4 paths. CPU E8M0 dequant 5/5·1142. GPU W4A4 + MoE-expert e2e later |
| CPU backend vs llama.cpp | 20-core Arm at floor, **denominator SUPERSEDED** (fork `237ad9b96`, owed a re-take vs stock `b10451`, #1003); RPi5/A76 below floor `GATING`; **x86_64 open on every axis** | Pi: AAPCS64 beats SDOT 3.66-5.08%; llama.cpp 2.17x pf / 1.53x dec faster (0.461x/0.653x); RSS -24.2% vs stock `b9892`; 64-tok byte-exact. BF16 GEMM open. x86_64 first measured 2026-08-11 (#433): peak RSS 1.0022x = hairline OPEN GAP (6.33 MB against us), throughput pending a quiet host, `G5` load-discipline gate failing, quant path portable-tier only (CIQ `G5`) |
| Paged KV cache + prefix caching | Supported | Block-paged full attention, hybrid full-attention + GDN state groups, automatic prefix caching (APC) on by default for dense models (cache-ON gated end to end: token-identical output, cache hits, faster TTFT) |
| fp8 KV cache (`cache_dtype=fp8`) | In progress (W1 CPU brick), not yet usable end-to-end | HIGH-priority memory/throughput lever (halves the KV footprint). W0 spike + W1 CPU brick landed (`KV-FP8` ACTIVE): fp8-e4m3 K/V STORE (`Quantize(hp/scale)`) + the paged-attention READ dequant (`Dequant(fp8)*scale`) + the `cache_dtype` config parse, all CPU-gated RED-first (`test_ops_fp8_kv_cache` 8/8·511; a wrong store direction fails 3/480). Storage is 1-byte fp8 (`DType::kI8`) + a `Fp8KVCacheDataType` interpretation enum, per-tensor k/v scales (mirroring vLLM `BaseKVCacheMethod`). The CUDA store + fp8 paged-attention read (the GPU memory-halving path, DGX-blocked), the runner/spec integration (half-sized KV blocks + checkpoint-scale threading + `--kv-cache-dtype`/`--calculate-kv-scales`), fp8_e5m2 and per-head scales are named W2-W5 in [.agents/specs/fp8-kv-cache.md](../.agents/specs/fp8-kv-cache.md). No model can run with an fp8 KV cache yet |
| Prefix-cache matching unit (`--prefix-match-unit`) | Partial (resolver landed, config/scheduler wiring pending) | 0.26-new knob setting the finest token boundary a prefix-cache hit can land on (the `hash_block_size`). W1: `resolve_kv_cache_block_sizes` ported 1:1 (hybrid `hash_block_size = prefix_match_unit if set else gcd(group block sizes)`; single-group inert; back-off on no-consumer / mamba-non-align; throws on non-divisible), CPU unit-gated RED-first (default gcd vs `=16`). Pending: the config/CLI/ABI field (W2), scheduler threading of a finer-than-block matching unit (W3, needs the KV-block-pool align path), and the benchmark (W4). Default path byte-identical (dense single-group models ignore it). |
| CLI `chat` / `complete` (`SERVE-CLI-CHAT`) | `SPIKE` (implementation pending) | The accepted CPU-only spike corrects a stale inventory claim: pinned vLLM 0.26 ships both commands as clients of a running OpenAI-compatible server. The selected design mirrors their URL/auth/model-discovery, quick/interactive, SSE, history, and TTFT/TPS contracts while preserving the existing in-process `--model --prompt` invocation as a compatibility alias. No remote command code or support claim has landed; next gate is W1 parse/dispatch plus W2 fake-server transcript tests. See [the spike](../.agents/specs/cli-chat-complete.md). |
| KV offload to CPU / disk | Built, opt-in, off by default; the disk connector is engine-refused | CPU and disk tiers with identity-checked blocks, selected by `--kv-transfer-config` (or programmatically) over one abstract KVConnector ABI. Worker-side KV store/load is implemented for the LMCache connector only; the CPU/disk connector is scheduler-side only, so the engine now REFUSES it at construction (a loud error, not silently wrong output). Guide: [docs/KV-OFFLOAD.md](KV-OFFLOAD.md) |
| LMCache client (`lm://` remote KV) | Built, opt-in, off by default; a working, verified external KV cache | Pure-C++ `lm://` client wired as an `LMCacheConnector`, no `lmcache` in-process; keys agree byte-for-byte with a real vLLM+LMCache peer, mismatched blocks refused. Proven in a real OPT-125m loop vs a live `lmcache.v1.server`: connector-ON tokens are BIT-IDENTICAL to the connector-OFF cold run (both after an in-process restart and from a cold second process). See docs/BENCHMARKS.md |
| KV-cache events (for external routers) | Built, off by default; generation and payload gated, live ZMQ transport deferred | Block store/remove/clear events (`BlockStored`/`BlockRemoved`/`AllBlocksCleared`) emitted at the prefix-cache sites when enabled, with a `msgpack` payload byte-identical to vLLM's `msgspec` encoding. Behind a publisher seam faithful to `--kv-events-config`; the live ZMQ transport is not wired yet. Off by default, so the prefix-cache path is byte-identical. |
| Sampling | Supported | Greedy, temperature, top-k/p, min-p, presence/frequency/repetition penalties, seed, stop/stop_token_ids, min_tokens, logit_bias, allowed_token_ids, bad_words, in vLLM's exact order. Custom logits processors are supported through a per-request C-ABI callback (`vllm_logits_processor`, ABI v8), absent by default (byte-identical). Sample logprobs are emitted end-to-end for `/v1/completions` and `/v1/chat/completions` (`logprobs`/`top_logprobs`). A request may instead score an EXPLICIT set of vocab ids (`logprob_token_ids`, generative scoring): exactly those ids plus the sampled token, whose rank is still over the full vocab, and no full-vocab sort — library surface only, the OpenAI request field is not wired yet. Parallel sampling (`n>1`) is supported, returned as n indexed `choices` (`n==1` byte-identical). Beam search is supported through the `BeamSearch` driver — an outer engine loop that scores beams by cumulative logprob with a length penalty and returns the top `beam_width` sequences (deterministic, token-exact vs vLLM's algorithm); it is wired on the OpenAI `use_beam_search` request field for `/v1/completions` and `/v1/chat/completions` over BOTH the synchronous engine AND the production AsyncLLM HTTP server (an async `BeamSearchAsync` driver, token-identical); the C-ABI beam params and streaming beam are not exposed yet; per-beam concurrent stepping is a named residual — beams are stepped sequentially). `best_of` is supported on both endpoints (`best_of==n` is the default no-op). `prompt_logprobs` is computed and returned on `RequestOutput`; the OpenAI `echo` serialization is not wired yet. All four `logprobs_mode` values work (`processed_*` shows the top-k mask); in-library only. |
| Structured output | Supported (subset), engine-enforced; xgrammar backend W1 (CPU, not yet production-wired) | JSON schema, JSON object, regex, choice, GBNF grammar. Constrained decoding runs in the production engine (native grammar backend, per-step logits bitmask) and is reachable from OpenAI `response_format` and the C ABI (ABI v2 `structured_*` fields). A second, xgrammar-faithful backend (`XgrammarStructuredOutputBackend`, vLLM's default `auto`) is built behind the same seam: it reuses the native pushdown-FSM/trie matcher (xgrammar's own algorithm) and adds the xgrammar JSON-schema→EBNF converter that preserves property declaration order + `any_whitespace` + the `basic_*` grammar, closing the key-order/whitespace/exotic-schema parity gap. CPU-gated (`test_backend_xgrammar` 6/6, RED-first); production wiring + the `auto` fallback + GPU oracle parity are the named residuals (see `.agents/specs/xgrammar-backend.md`) |
| Tool-call parsing (`TOOLS-PARSER-BREADTH`, PARTIAL) | 38 parser families / 42 accepted names, streaming | 40 of upstream's 44 registered names at the pin, plus 2 that upstream's registry does not carry there (`qwen3`, our alias for the Hermes-JSON Qwen dialect, and `muse_glimmer`) = 42 accepted names; pure-text parsers ported 1:1, the engine-backed families reimplemented from their wire formats. `inkling` landed 2026-08-13 (W1) as a `ParserEngineToolAdapter` over the already-ported Inkling ParserEngine — the only one of the five upstream-only names that was a port from vLLM source at all. FOUR remain NOT ported, and none of them has its GRAMMAR in vLLM source: `minimax_m3` is backed by upstream's Rust crate; `openai` (`GptOssToolParser`) is a stub that raises on both methods and delegates to `vllm/parser/harmony.py`, which IS vLLM source (and has its own test) but itself wraps the out-of-tree `openai_harmony` package; `cohere_command3` and `cohere_command4` are shims over the out-of-tree `cohere_melody` package — tracked as W1-remaining/W2 in `.agents/specs/tool-parser-breadth.md`, which records the decision each one owes. Upstream's shared `ToolParserTestConfig` harness is also unported, so the per-parser test floor is set file by file rather than enforced (W3). Selection via `--tool-call-parser` (server), `tool_parser` (C ABI), or template auto-detection over a 27-row ordered marker table (`inkling` is EXPLICIT-ONLY: it has no jinja chat template upstream, so there is nothing to sniff); native-syntax forced tool_choice where expressible. Tables: docs/BENCHMARKS.md |
| Reasoning parsing (`SAMPLE-REASONING`, ACTIVE, partial coverage) | 12 names, streaming | think_auto (auto-detect default: content unless markers appear), deepseek_r1, deepseek_v3 (passthrough) / holo2 (thinking→R1), mistral ([THINK]), minimax_m2 (+append_think), step3, olmo3, muse_glimmer, and qwen3 / mimo - reasoning split engine-side BEFORE tool parsing, streamed as `reasoning` deltas in the chat chunks. qwen3+mimo are the first ENGINE-BACKED adapter (one reasoning face over the shared `src/vllm/parser/engine/` parser, so `<tool_call>` ends reasoning with no `</think>`); the rest are text parsers. Coverage: 12 of upstream's ~28 registered names (remaining engine-backed adapters + text families tracked as W3/W2 in specs/reasoning-parsers.md); each ported parser doctest-gated vs its tests/reasoning case |
| Unified streaming parser engine | Core, assembly, serving-SSE dispatch landed, gated; all 10 engine-backed families ported (family parity closed); JSON-schema tool-arg type coercion landed | The vLLM 0.26 declarative `parser/engine/` (shared state machine plus all 10 configs: qwen3, seed_oss, kimi_k2, minimax_m2, glm47_moe, deepseek_v4/v32, nemotron_v3, gemma4, inkling) and assembly layer, gated field-for-field vs vLLM 0.26. An engine-backed `--tool-call-parser` name drives the live chat SSE chunks, off by default. When a request's tools declare typed parameters, the assembled tool-call arguments are coerced to the declared JSON types (int/number/bool/string/array/null) 1:1 with vLLM `_fix_arg_types`, in both streaming and one-shot; no schema means the arguments pass through as strings unchanged. Details: .agents/specs/parser-assembly-c8.md |
| OpenAI server | Subset; v0.0.2 publishes eight server bundles; Windows v0.0.3-pre.1 pending | Completion/chat (SSE), models, health/version/ping, metrics, tokenize/detokenize, tokenizer/server info, prefix-cache reset, abort, and Sora-shaped video creation/content. Tokenizer info and abort are flag-gated; cache reset lacks live async backing. Details: docs/USAGE.md |
| Pooling task class (embeddings / classify / score / rerank) | **EMBEDDINGS LIVE ON THE ONE SURFACE (ROW 6)**: `LlamaModel` registered, `PoolingRunner` in the engine step, `vllm_embed` (ABI v15) + live `/v1/embeddings`; classify/score/rerank engine-side only | The non-generative task class. W0 spike over the whole vLLM pooling surface (`.agents/specs/pooling-task-class.md`, `CLAIM-POOLING`). W1 landed the pooler OP (CLS/LAST/MEAN + Identity/Normalize/MultiLabelClassify/Classify activations, double-precision-gated). **W2 landed the pooler HEADS composite** (`EmbeddingPoolerHead`, `ClassifierPoolerHead`, `SequencePooler` + factories, `DispatchPooler` routing, `PoolerConfig`/`PoolingParams`; `test_pooler_heads` 27/27-240, RED-first). **W3 landed the pooling RUNNER path** (`PoolingRunner`: pooled embeddings instead of sampled tokens, structural cosine gate vs an f64 LAST+normalize reference, `test_pooling_runner` 5/5-14, RED-first). **ROW 6 (2026-08-08): embeddings LIVE** — fold gate `test_llama_embedding_fold` 4/4-231 (engine path == direct registry path, f64 LAST+normalize ref, chunked is_valid arm); residuals: REAL checkpoint + `LLM(task="embed")` oracle cosine (no number fabricated), score/rerank/classify endpoints, matryoshka/base64/token-array inputs, tokwise (W5). Detail: `.agents/specs/embeddings-one-surface.md` |
| Plugin system (out-of-core registration) | Spiked; first CPU brick landed, not yet wired into any production path | The extensibility-first discovery layer. W0 spike over vLLM's plugin surface (general / platform / io_processor / endpoint groups, the `register_model` an out-of-tree plugin calls, the invocation seams) is committed (`.agents/specs/plugin-system.md`, `ENG-PLUGIN-SYSTEM` ACTIVE, `CLAIM-PLUGIN-SYSTEM`). W1 landed `vllm::plugins::LoadGeneralPlugins()` + the out-of-core general-plugin registration seam (`RegisterGeneralPlugin` / `REGISTER_VLLM_GENERAL_PLUGIN`) over the existing `REGISTER_VLLM_MODEL`-style registries (the in-tree factory `MODEL-FACTORY-registry` is record-repaired `DONE` 2026-08-05: 28 self-registering TUs, dgx debt paid by the 2026-07-23 seven-gate run): a 1:1 mirror of `load_general_plugins` (load-once idempotence, the `VLLM_PLUGINS` allowlist, per-plugin failure isolation). Proven by an out-of-core toy-model plugin that registers a toy architecture through the public `RegisterModel` seam — unit-gated RED-first (`test_plugin_system` 1 case / 29 assertions: the toy arch resolves ONLY after LoadGeneralPlugins runs it, and not under `VLLM_PLUGINS=""`). Python entry points have no C++20 analogue, so discovery is the project's static-init/`dlopen` registration idiom (recorded porting-inventory §9). NOT yet wired: real shared-object `dlopen` + the C-ABI `vllm_plugin_register` entry (W2), the engine/CLI `--load-plugins` wiring that calls LoadGeneralPlugins from the construction paths (W3), the platform/quant plugin kinds (W4), and the io_processor/stat_logger/endpoint groups (W5) are named residuals. See docs/BENCHMARKS.md |
| Offline Batch API (JSONL file runner) | Spiked; first CPU brick landed, not yet exposed as a CLI | The offline OpenAI Batch API: read a JSONL of OpenAI-format requests, run each through the engine, write a JSONL of responses. W0 spike over vLLM's `run_batch.py` (schema, endpoint dispatch, run loop, file I/O) is committed (`.agents/specs/batch-api.md`, `SERVE-BATCH-API` ACTIVE, `CLAIM-BATCH-API`). W1 landed `RunBatch` (`RunLine`/`RunLines`/`Run`) + `RunBatchFile` — a pure orchestrator over the existing `OpenAIServingChat::create_chat_completion` (NO reimplemented generation), 1:1 with vLLM's endpoint_registry url→handler map: `/v1/chat/completions` dispatch, the `BatchResponseData`/`BatchRequestOutput` schema (`vllm-<uuid>` ids, custom_id echoed), the `run_request` AllResponse/ErrorResponse/stream branches, and the unsupported-endpoint/url error rows. Unit-gated RED-first (`test_openai_run_batch` 7 cases / 80 assertions over the synthetic serving engine: ordered rows + custom_id echo + per-line BatchRequestOutput schema round-trip, a malformed line isolated into an error row so the batch continues, dispatch + 404 error rows; dropping the custom_id echo fails 9 assertions). Recorded deviation: a malformed line is isolated (batch continues) where upstream aborts the job. NOT yet exposed: the `vllm run-batch` CLI + `BatchFrontendArgs` (W2), embeddings/score/rerank dispatch (W3, rides pooling endpoints), audio transcription/translation + media fetch (W4), and http(s)/data-URL file I/O + metrics + overlapped `AsyncLLM` submission (W5) are named residuals. See docs/BENCHMARKS.md |
| Tokenizers | Supported | Byte-level BPE (Qwen/Llama-3/OPT/GPT-2/DeepSeek/OLMo-2) and SentencePiece BPE (Mistral/Gemma), plus GGUF vocab; added-token `lstrip`/`rstrip` whitespace semantics (e.g. Phi-4-mini's special tokens); byte-exact vs the vLLM oracle |
| Multimodal: image to text | Correctness-complete; vision-forward speed beats vLLM; OpenAI-server content-part parse + processor routing + engine mm-request plumbing landed (CPU); end-to-end serving pending | Strict token-exact 32/32 vs vLLM 0.25.0 on Qwen3-VL-4B and Qwen3.6-27B (`Qwen3_5ForConditionalGeneration`); C++ image processor + vision tower + MRoPE/DeepStack backbone with on-GPU greedy sampling; the 27B decode step is now graph-capturable (routes through the production captured decode, token-exact held). The vision tower now defaults to the flash-tiled non-causal attention (`vt::AttentionDenseFlash`, byte-identical to the previous warp kernel — token-identical, goldens unchanged): the per-image tower forward is ~142 ms vs vLLM 0.25.0's ~250 ms eager encode = 0.57x (faster). Attribution-first profiling found the tower attention is serial-latency-bound (not K/V-bandwidth-bound), so flash tiling is only a 1.04x same-binary A/B over the warp kernel here — the tower already beats vLLM; batched serving speed still pending. OpenAI-server multimodal wiring — CPU bricks 1+2 landed (`CLAIM-MM-SERVING-W1`/`W2`, .agents/specs/mm-serving.md): the chat request parses the OpenAI multimodal content-part array (`image_url`/`input_audio`/`audio_url`), decodes the base64/`data:` payloads, routes them through the existing single-sequence processors to a placeholder-expanded prompt + mm-feature handles, AND the engine now carries them — additive `LLMEngine`/`AsyncLLM` `add_request(MultiModalInputs)`/`generate` overloads (via `InputProcessor::process_inputs_mm`, mm_features onto `EngineCoreRequest`/`Request`), chat-template placeholder-string helpers, and a serving_chat `MultiModalChatFn` seam (default unset ⇒ text path byte-identical). The W3 `MultiModalChatFn` seam BODY now lands (`CLAIM-MM-SERVING-E2E`): `MakeQwen3VLImageChatFn` (chat_mm.{h,cpp}) turns an image chat request into the placeholder-EXPANDED engine input — marker-inject at the mm part position, render the chat template, tokenize (the single `<|image_pad|>` maps to one image_token_id via `EncodeWithSpecialTokens`), then `RouteImageRgb` expands to 196 image tokens + mm_features — wired into `src/vllm/entrypoints/openai/server_main.cpp` (guarded on `preprocessor_config.json`; text-only models leave it unset ⇒ byte-identical). Unit-gated `test_chat_mm` 8/8 (the W3 seam test drives the real tokenizer + chat template → 196 image tokens; RED line = the text-only path renders 0) + `test_input_processor` 10/10 + `test_openai_serving` (the production seam is invoked on an image request + routed to the engine mm generate overload; text-only never touches it; streaming+mm rejected), CPU-only, no model weights; bare-string chat requests byte-identical (proven). **ENGINE MM-FORWARD LANDED (`CLAIM-ENGINE-MM-FORWARD`, 2026-07-28): multimodal now runs through the engine's REGISTERED forward.** The architectural block is resolved: `ModelForwardInput` gains an additive default-nullopt `mm` field (merged inputs_embeds + 3-D MRoPE positions + DeepStack; nullopt for text ⇒ the shared runner path is byte-identical by construction), `Qwen3VLForConditionalGeneration` is now `REGISTER_VLLM_MODEL`-registered, and the registered forward folds the M2c decode into `ModelRegistry::Forward` via the shared `Qwen3VLForwardStepLastLogits` (`Qwen3VLGenerateGreedyViaRegistry` drives every step through the registry). GPU token-exact gate `test_qwen3vl_registry_e2e` on dgx.casa GB10: an image→text run THROUGH `ModelRegistry::Forward` emits the M2c golden tokens 32/32 STRICT (RED-first — unregistered ⇒ resolve throws); the M2c standalone cross-check holds 32/32 (the refactor is byte-neutral). Text inertness (the shared-path RED line): `test_runner` 16/16 + `test_scheduler` 36/36 + `test_model_registry` 24/24 + `test_chat_mm` 8/8 + `test_openai_serving` 41/41 all green. **Residual:** the FULL in-runner scheduler-fed tower run (the batched engine loop building the mm field from staged encoder outputs) + the real server `/v1/chat/completions` GPU end-to-end + video/multi-image/audio through the registered path — recipe in `.agents/specs/mm-serving.md`. **GEMMA-4 IMAGE through the registered forward LANDED (`CLAIM-GEMMA4-MM-E2E`, 2026-07-29):** `Gemma4ForConditionalGeneration` is now `supports_multimodal=true` with an mm branch routing `ModelForwardInput.mm` → `Gemma4Model::ForwardMm` (the SigLIP2 projector output masked-scattered into the `<image>` rows + PLE image-rows→0; 1-D positions, no MRoPE/DeepStack); the driver `Gemma4GenerateGreedyViaRegistry` runs image→text through `ModelRegistry::Forward`. dgx sm_121a gate `test_gemma4_registry_e2e`: **16/18 content tokens BIT-EXACT** vs the STRICT `gemma4_e4b_image` golden (full sentence), the single divergence a terminal-punctuation bf16 near-tie (top1-top2 margin ~0.10-0.12 logit, invariant to vision-input precision ⇒ backbone bf16-accumulation, not the fold); text SACRED 32/32 UNCHANGED (inertness). Residuals: STRICT-18/18 (bit-match vLLM prefill bf16), Gemma-4 AUDIO→text e2e (mel A1 + merge; the G3 audio tower is per-stage proven), speed. **C2 (2026-07-31, `CLAIM-C2-VISION-QKV-BIAS`): both vision towers' attention QKV now fold to ONE `vt::MatmulBT` + a fused merged-`[3H]` BIAS epilogue + a contiguous `vt::QkvSplit` via the shared `models::FusedMergedQkvBiasSplit` helper** — qwen3_vl_vision (weight already resident-fused `[3H,H]`+`[3H]` bias) and gemma4_vision (loader-concatenated `qkv_proj [3H,H]`, no bias, per-slice clamp epilogue kept). Bit-exact (A/B unit `test_ops_qkv_merged_bias` byte-identical, RED-first); qwen3_vl gated e2e by `test_qwen3vl_e2e` image→text token-exact, gemma4_vision CPU-A/B + CUDA build-verified; existing tower gates held. See docs/BENCHMARKS.md |
| Multimodal: video to text | Correctness-complete, speed-pending; OpenAI-server content-part parse + engine mm-request plumbing landed (CPU), end-to-end serving pending | End-to-end on Qwen3-VL-4B (near-tie-robust) and Qwen3.6-27B (strict 32/32); reuses the image tower and temporal MRoPE plus video preprocessing (frame sampling, temporal grid) |
| Multimodal: audio to text | Decode beats vLLM; encoder TTFT measured and improved but not yet at parity; OpenAI-server content-part parse + audio processor routing + engine mm-request plumbing landed (CPU), end-to-end serving pending | End-to-end on Voxtral-Mini-3B (near-tie-robust, decoder token-exact 48/48) plus a Whisper-class encoder tower. Decode is graph-captured and runs vLLM's FA2 varlen split-KV kernel (the KV block-size is rounded up to a multiple of 16 to meet its precondition): audio TPOT 39.5 ms/token BEATS vLLM 0.25.0's 40.8 ms (0.97x, non-overlapping bands). The Whisper encoder attention now uses a flash-tiled non-causal head-dim-64 kernel (`vt::AttentionDenseFlash`): a block of query-warps shares each streamed K/V tile out of shared memory (FlashAttention K/V tiling), with the per-warp online-softmax math copied verbatim from the warp kernel so the output is bit-identical (token-identical). Same-binary A/B: the encoder self-attention drops from 35.11 to 19.29 ms/layer (1.82x, non-overlapping) and the encoder forward from ~1.83 s to ~1.37 s (1.33x), token-identical output (16/16, STRICT golden unchanged, sanitizer 0). The encoder weights are now device-resident: each of the 487 encoder weight tensors is converted to bf16 and uploaded to the GPU once (mirroring the decoder's residency), reused across forwards instead of being re-marshalled every call; same-binary A/B (`VT_WHISPER_ENC_REMARSHAL`) drops the encoder forward a further ~1.37 s to ~0.73 s (1.89x, non-overlapping), removing ~648 ms of per-call host weight marshalling (nsys: 974 fewer Host-to-Device copies, ~2.5 GB less traffic), byte-identical (16/16, STRICT golden unchanged, sanitizer 0). But encoder TTFT (~0.73 s) is still far above vLLM's 43 ms (~17x, was ~32x): the residual is now GPU-compute-bound (the scalar warp-per-query attention plus the conv GEMMs), so closing it needs a tensor-core MMA head-dim-64 non-causal flash attention. Correctness held under the ratified distributional near-tie gate (teacher-force PASS, 0 divergent, strict prefix exact vs vLLM; STRICT golden unchanged; 16/16). Remaining: a tensor-core MMA encoder-attention kernel and dropping the conv host round-trip (a device im2col kernel), and there is no batched c2+ or `audio_url` serving ingestion. See docs/BENCHMARKS.md Encoder attention now has an opt-in FA-2 tensor-core path (`VT_WHISPER_ENC_FA2=1`) that is 5.50x faster and takes the ENCODER FORWARD from 15.90x to 2.89x of vLLM's whole TTFT (pinned oracle `555967922`, TTFT median 46.02 ms, production/graphed). Those are encoder-forward-vs-TTFT ratios and NOT TTFT ratios: our projector, merge and prefill are unmeasured and so are absent from the numerator. Held OPT-IN because it costs 3 near-tie divergences against the default's 0; adoption is a developer decision. Which of the five FA-2-vs-scalar differences flips those tokens is NOT established: the originally stated cause (bf16 P before the PV MMA) was refuted by forcing exactly that rounding into the scalar kernel, which left the tokens unchanged (multimodal-speed.md S17.5, #432). |

| SGLang parity (competitor floor + oracle) | Oracle STOOD UP + first floor MEASURED (cache-neutral, 27B, c8/c16): throughput/TTFT WIN, TPOT/ITL open GAP | SGLang v0.5.15 (`f63458b`) whole runtime surface inventoried (44 rows: 23 FUSED into our vLLM-derived engine, 8 SGLANG-DISTINCT opt-ins, 5 inventoried, 8 out-of-scope). The `v0.5.15-cu130` arm64 image (`@sha256:d0a667e`) PULLED and RAN the 27B-NVFP4 gate model on GB10 sm_121a with no from-source build. First reproduced SGLang-vs-ours cache-neutral comparison (27B, 3 reps, idle box, one flock, engines sequential): **ours beats the SGLang floor on total/output throughput + req/s (2.21×@c16, 1.44×@c8) and TTFT (6–12× lower), but SGLang wins per-token latency (TPOT/ITL 1.18–1.49× below ours) — a reproduced OPEN GAP.** SGLang is a competitor, not the mirror source (vLLM stays behavior truth). Residuals: 35B, c1/c2/c4 low-conc sweep, shared-prefix cache-ON arm, token-exact cross-check. Numbers + repro: `docs/BENCHMARKS.md`, `.agents/sglang-matrix.md`, `.agents/specs/sglang-parity-oracle.md`. **UPDATE 2026-07-28 (`CLAIM-DECODE-LATENCY-EXPLORE`, measurement only, no source changed): the TPOT/ITL gap is CONFIRMED batch-composition, NOT a decode-kernel deficiency** — on our engine ITL(decode-batch=1)=101.75 ms is already ≤ SGLang's op-point 104–105 ms and rises monotonically with batch (→158.5 ms @ B16); nsys shows every hot decode kernel sub-linear in batch (per-token cost ↓~10×); SGLang's effective decode concurrency is ~4 (not 16) due to its 33 s admission queue, so its low ITL is simply the ITL of a small batch. Our throughput win IS the ITL cost — same lever; knob `max_num_seqs`/`max_num_batched_tokens` already exists, latency-oriented point `max_num_seqs≈8` = ITL −21% at 1.38× SGLang throughput; default stays throughput-oriented. Full data: `.agents/specs/decode-latency-lever.md` **UPDATE 2026-08-16 ([#979](https://github.com/mudler/vllm.cpp/issues/979), records only, nothing measured): two entries were stale and are corrected.** `BACKEND-GATE-CUDA-SGLANG` moves `BLOCKED` to `PARTIAL`: its recorded blocker `SERVE-ASYNC-LLM` is discharged, so the reason "HTTP TTFT/ITL cannot be measured honestly yet" is retracted. The production server streams incrementally over `AsyncLLM` (`serving_completion.h:9`, `api_server.cpp:971-981`), the benchmark harness ENFORCES it rather than assuming it (`tools/bench/run_serve_low.py:296-310` refuses `first_chunk_s >= total_s`), and the run above demonstrates it: c16 mean TTFT 2980 ms against mean ITL 154.4 ms over 128 tokens means first byte preceded completion by about twenty seconds, which a buffered server cannot do. That run is NOT voided by [#931](https://github.com/mudler/vllm.cpp/issues/931), because the keepalive frame needs 15 s of silence and the worst observed p99 TTFT was 7220 ms. The SGLang oracle also moves to `gateable = yes`: `.agents/oracles/sglang.md` still said "no SGLang run has been recorded on this project's hardware" two and a half weeks after this measurement, and `docs/BENCHMARKS.md` still said "Never ran". This discharges the SGLang third of [#647](https://github.com/mudler/vllm.cpp/issues/647). **REVERSED 2026-08-18 ([#1265](https://github.com/mudler/vllm.cpp/issues/1265), records only, nothing measured and nothing retracted): the oracle moves back to `gateable = no`.** The 2026-07-28 numbers stand, but the only method that produced them — `ssh` to the host plus `docker run` under `flock` — is the path AGENTS.md §"Work on a GPU happens inside a lease" now forbids for a fleet device, and no substitute is demonstrated: `rc run` has no `--image`, the leased worker carries no container runtime, and `scripts/dgx-sglang-low-concurrency.sh` is image-based and unrunnable as written. That returns the SGLang third of #647 to open, on the new ground that the reach is gone rather than the pin. A new `BACKEND-GATE-CUDA-LLAMACPP` row is filed `INVENTORIED` for the llama.cpp-on-current-CUDA arm, which had no owner. Campaign spec: `.agents/specs/bench-qwen38-27b-four-way.md` |

**CUDA graph capture takes BREAK POINTS** as of `ENG-CUDAGRAPH-BREAK` W1
(`ACTIVE`, 2026-08-18, [spec](../.agents/specs/eng-cudagraph-break.md), #1192,
parent #1163). A forward can be captured as a SEQUENCE of segments split at
break points, so a forward containing a host-dependent operation is still
graphed except at that operation instead of falling out to eager for the whole
step.

`vt::BreakableGraph`, `vt::GraphCaptureScope` and `vt::GraphBreak`
(`include/vt/breakable_graph.h`) replace the nine hand-rolled per-model drivers
as those migrate. **Coverage and correctness, not speed**: no throughput gate is
declared, because our prefill has no launch bubbles to collapse (3.8% host idle,
GPU-busy above 96%) and decode already banked its launch-overhead win.

The boundary is vLLM's — its v1 default splits at `splitting_ops`, defaulted to
the attention family — and only the registration form is SGLang's, because vLLM
gets its split from Dynamo and FX and we have no compiler. W1 registers ONE
break point, at the dense attention entry of `Qwen3ForCausalLM`.

W2 (2026-08-18, #1261) migrates the first driver: the shared dense decode graph
now captures and replays through the seam instead of its own `BeginCapture` pair
and raw handle. It captures in FULL mode, mirroring vLLM's decode arm, so a
decode step keeps the single graph it already had; the PIECEWISE arm has no
production driver yet, because the break closure W1 registered reads a returned
frame on replay. Bit-exactness against a REPLAYED capture on a real GPU is owed,
not met.

A capture that FAILS is now distinguishable from a scope that was INERT. The
capture scope has to swallow a throwing `EndCaptureGraph` — a destructor that
propagates terminates — and both states left the container reporting
"not captured", so a driver reading only that bit could return uncomputed device
memory as a decode step's logits. The container records the failure and the
exception, and the driver propagates it.

The stage's exit criterion was measured rather than assumed: ending a capture
and BEGINNING A NEW ONE on the same stream mid-forward, with eager work between,
is legal under the thread-local capture mode our CUDA backend uses (`orin:gpu0`
via an `rc` lease, driver 12060, three replays with fresh inputs, zero
mismatches).

Gates: `tests/vt/test_breakable_graph.cpp` ports SGLang's unit suite case for
case, with its arithmetic chains and its post-replay assertions intact (24
cases, 163 assertions); `tests/vllm/models/test_qwen3_break_point.cpp` drives
the production forward with a scope open, counts one segment per layer plus one,
and compares the logits BIT FOR BIT against the unscoped forward (500 values, 0
differing).

The seam ENFORCES what it used to document. A destination-carrying break point
takes a `vt::BreakSlot` whose storage the seam owns, because the following
segment bakes that address and a caller's local dies first; a destination with no
`CopyOutput` overload is a compile error naming the type rather than a silent
drop into the no-writeback path; a second capture scope on one thread and a
re-entered container are refused; and a forward that throws OUT OF the scope
mid-capture leaves no partial graph reporting itself as captured. Interleaved
replay, the `VLLM_CPP_CUDAGRAPH=0` kill switch and each refusal above are gated,
every one proven by a mutation that reds them.

Two break points in one capture writing through ONE destination are refused as
well, at registration. `BreakSlot` closed the lifetime half of that rule and
left the aliasing half writable: both replay closures bound to the same address,
so the earlier writeback was overwritten. One shape still escapes, and the spec
names it and assigns it to W2 — an exception CAUGHT INSIDE the scope leaves the
rest of the forward uncaptured while `captured()` stays true, because nothing is
unwinding at scope exit for the drain to see.

**Not yet entered from a production step.** No driver opens a capture scope
until W2 migrates `Qwen3DenseDecodeGraph`; the break point itself runs on every
forward and takes the pass-through arm. The spec's `## Owed` names that with its
owner, along with the auxiliary-stream auto-join, the ROCm and Tenstorrent arms,
and GPU bit-exactness over more than one replay.

## Speculative decoding

**MTP draft DEPTH is configurable** as of `SPEC-MTP-K-GT-1` (`ACTIVE`,
2026-08-16, [spec](../.agents/specs/mtp-k-gt-1.md), #81):
`num_speculative_tokens` loops the single MTP head autoregressively, CPU-gated
at k=1..4. A token-identity gate cannot see a clamped drafter, so each arm
carries TWO witnesses: the `k-1` draft decode forwards per propose call catch a
propose that never ran the loop, and a varied-draft counter over the DELIVERED
rows catches one that ran it and then padded. Neither proves that column j came
from forward j, and no draft is ACCEPTED at any depth in the CPU gate, so
provenance and the accept path both await the owed DGX gate. The DEFAULT is
unchanged at k=1 and **no speed number is claimed above it**.

On real 27B NVFP4 weights the depth arms DO accept: re-measured 2026-08-17 with
all seven arms in one uncontended window, k=2 gives depth-0 0.878, depth-1 0.731.
**The token gate is still NOT claimed**: our spec-ON is not token-identical to
our spec-OFF on 3 of 4 prompts, at the same positions for every k, and the vLLM
leg that would attribute the split has never run here. Three passes failed to run
it: the reimaged box has no C compiler, so the oracle's Triton JIT dies after the
weights load, and once that is fixed the oracle consumes the whole 119 GiB host
in the step after `torch.compile`. `gpu_memory_utilization` does NOT control that
second one: an A/B at 0.30 collapsed as 0.75 did, and rebooted the box.

Speculative decoding is available on the Qwen3.5/3.6 checkpoints via
`--speculative-config`. **MTP (k=1)** is end-to-end token-exact vs vLLM on
both gate models (the 27B GDN hybrid `Qwen3_5MTP` and the 35B MoE `Qwen3_5MoeMTP`):
three-way identical at concurrency 1 (our spec-ON == our spec-OFF == vLLM
`--speculative-config mtp` greedy) and faster than vLLM there, on par or above
at higher concurrency (mixed-batch), with the draft head alive and acceptance
matched to vLLM.

**Block-diffusion DFlash** (the z-lab Qwen3.6-27B draft over the 27B NVFP4
target) is correctness-complete and at or above vLLM throughput on GB10 (final
concurrency-1 A/B our-on 29.32 tok/s vs vLLM-on 29.24, non-overlapping bands,
1.003x); it is recorded DONE across the engine, model and kernel matrices on the
vLLM 0.26.0.dev0 stack (which resolves vllm#40898), and it remains gated behind
a spike while its user-facing serving surface is finalized.

**Method surface (enumerated from vLLM source 2026-08-06, `.agents/specs/spec-decode-inventory.md`).** Of the 13 vLLM `SpeculativeMethod` strings we ship MTP (any k), DFlash, DSpark and n-gram; draft_model is a CPU brick and Medusa a spike; EAGLE1/EAGLE3, ngram-gpu, suffix, custom_class, extract_hidden_states, dynamic-k and the synthetic/block acceptance variants are INVENTORIED; mlp_speculator is upstream-deprecated (no V1 proposer). Draft DEPTH: MTP k>1 is BUILT and CPU-gated (`SPEC-MTP-K-GT-1`, no speed number yet). Dynamic (batch-size-keyed) and adaptive (acceptance-driven) depth stay unbuilt (`ROAD-V1-D3-SPEC-K`, #81).

**DSpark block floor** (`SPEC-DSPARK-BLOCK-SIZE-GUARD`, ACTIVE, [#1225](https://github.com/mudler/vllm.cpp/issues/1225)). A speculative length below the draft's block was accepted silently: both `ResolveDspark` call sites passed `std::nullopt`, so the `k >= block` floor reached no user path, and our draft block is sized by `k` alone. It is refused now, with `block_size` supplying the floor when upstream's `dspark_block_size` is absent — one recorded divergence, because neither published Qwen3 draft sets that key. The GPU run gate that exhibits the garbling is owed.

**DSpark draft routing** (`SPEC-DSPARK-QWEN3-ROUTING`, ACTIVE) makes the loader classify a DSpark draft from the draft's own `config.json` before it resolves anything else. `Qwen3DSparkModel`, `Gemma4DSparkModel` and — ahead of the pin, mirroring vllm#52197 — `DSparkDraftModel` with `model_type` `qwen3` take the landed Qwen3 lane; a draft that resolves to the DeepSeek-V4 DSpark lane is refused BY NAME instead of being rewritten into a stub. CPU-gated only: the token-exact run gate against the pinned oracle waits on a draft download and GPU time that are not authorized, so it stays owed (#1193).

**DeepSeek-V4 native MTP** (`DeepSeekV4MTPModel`, ACTIVE — W1 self-spec wiring,
2026-07-30) has its nextn draft head wired to the same lossless spec-decode path.
Unlike V3's fused `eh_proj`, the V4 nextn layer keeps separate `e_proj`/`h_proj`,
`enorm`/`hnorm`, an MHC-aware `mtp_block` decoder layer, and its own `hc_head`
vocab collapse — ported 1:1 from `nvidia/mtp.py` as a tiny-config host draft
forward (`DeepseekV4MtpDraftLogitsHost`) reusing the DS4 attention/MoE/MHC
composition, verified by the SHARED greedy `RejectionSampler` so MTP-on greedy is
token-IDENTICAL to MTP-off (`test_deepseek_v4_mtp` 5/5, RED-first). The
real-model MTP-on==MTP-off + acceptance/speedup gate is WEIGHT-BLOCKED: both
shipped DeepSeek-V4-Flash GGUFs advertise `nextn_predict_layers=1` but the
llama.cpp converter dropped every nextn tensor (`DeepseekV4GgufHasMtp` returns
false → clean fall-back to MTP-off). The DS4-native propose/verify decode loop +
engine spec-config registration are named residuals. **BEAT-ds4 sweep (2026-07-30,
`CLAIM-DSV4-BEAT-SWEEP`, analysis-only):** our raw autoregressive decode is at its
GB10 ceiling (~13.1-13.2 tok/s, Q8_0 front CLOSED after Bricks 3-14), and ds4's
16.5-17.2 is itself RAW decode (its MTP/DSpark spec is opt-in, NOT in that
number). Its "only path PAST 16.5 is MTP self-spec" verdict is SUPERSEDED by the
1.14x BINDING below, which passes 16.5 with no spec at all; the self-spec math
(break-even p≈0.31, R1/R2/R4 residuals) and the refuted or tied levers (fp16
dequant cache, MHC) are in the spec. See
`.agents/specs/deepseek-beat-ds4-sweep-2026-07-30.md`.

**DeepSeek-V4-Flash decode levers (2026-08-03, byte-exact, default-ON; SUPERSEDED by the 1.14x BINDING below).** Four glue/geometry folds climbed decode ~13.5 -> 14.96 tok/s; per-lever forensics in `.agents/benchmark-record.md`.

**BINDING 2026-08-05: `VT_V4_RESIDENT_W` (default-ON) BEATS ds4 — 18.69 vs 16.33 tok/s (1.144x), byte-exact:** the dense Q8_0 MLA/shared/lm_head proj tower was read from GGUF-mmap over ATS; staging it `cudaMalloc`-device once (Q8_0 per-launch ~20% each, ids-IDENTICAL) lifts decode 16.23→18.69 (median-of-3, drop_caches, PEAK flat 86.68 GiB, net move). Mirrors Laguna `VT_LAGUNA_RESIDENT_BF16W`; our GEMV was ATS-bound, not at ds4 parity. **Phase-2 routed-expert residency (`VT_V4_RESIDENT_EXPERTS`) MEASURED NEGATIVE, HELD default-OFF (2026-08-05):** the ~70 GiB IQ2/Q2_K expert slabs staged device-resident (madvise move-semantics) are byte-exact but **−3.4% steady** (18.76 vs 19.43 tok/s, same-binary median-of-3) + a one-time capture cost — the grouped-MoE kernels are dequant/latency-bound (~19-24% of DRAM peak), so residency (a bandwidth lever) cannot help, and device pinning cuts pool headroom (103→30 GiB avail). Prior (superseded, see record): PARITY 16.28 vs 16.33 (0.997x); `VT_V4_MHC_SINK4` +4.6% byte-exact; HC-expand + f16-DSA held default-OFF.

**Integration re-verify (2026-08-03, GB10 sm_121a, clean CUDA+Triton Release build off `6576814b`).** Both decode glue-fold levers (`CLAIM-DSV4-ROPE-FLOAT` + `CLAIM-DSV4-MHC-LEAN`) cherry-picked clean and re-gated on a fresh GB10 build: **SACRED `test_qwen36_paged_engine` 315/315, 0 failed (UNMOVED)**; `test_cuda_deepseek_v4` **20/20·67073**; **byte-exact re-confirmed on the resident-decode path** — `deepseek-v4-gen --gpu --kv-cache` on ds4flash IQ2XXS gives decode ids token-IDENTICAL for `VT_V4_ROPE_FLOAT=1`/`=0` AND `VT_V4_MHC_LEAN=1024`/`=0` (the `--kv-cache` T==1 steps run `ForwardResidentDecodeGguf`, so the gated device kernels are genuinely exercised). The 27B paged-engine gate SKIPs (checkpoint absent on box).

**Cross-reference lever scan (2026-07-31, `CLAIM-XREF-LEVER-SCAN`, `.agents/specs/cross-ref-lever-scan-2026-07-31.md`).** A 5-source parallel scan (vLLM/SGLang/llama.cpp/ds4/ours) for batch-1 raw-decode levers, no-spec, adversarially verified against our code. **Corrects the record:** ds4/DwarfStar has NO register-prefetch (grep-verified) — the Brick-14 theory below was a misattribution; ds4's 90% comes from full-warp-per-row + lane-strided 34-B Q8_0 blocks (1088-B coalesced burst). Our production Q8_0 ships SUB-warp (8/16-lane/row) for medium-K (`LaunchQ8_0Subwarp`), so the one non-refuted Path-B lever left is **B1: port ds4's exact 32-lane/8-row `warp8` for medium-K + measure** (DGX A/B, not guaranteed). Verified cross-cutting gap: **PDL** (Programmatic Dependent Launch) — we have zero, both llama.cpp+SGLang use it on Blackwell. Path A (Laguna beat vLLM): the NVFP4 arm (#230) + the note that vLLM's 18.8 is MARLIN W4A16 (a lower bound; real default FLASHINFER_CUTLASS W4A4 is faster). Full ranked plan + citations in the spec.

**DeepSeek-V4-Flash framework-conformance — device-resident logits on the registry
forward** (2026-08-02, `CLAIM-DSV4-RUNNER-ROUTE`, branch `deepseek-v4-runner-route`,
CPU-compile-verified, DGX runtime gate OWED). First slice of moving DS4 off its
off-framework decode toward the shared runner. `DeepseekV4Model::ForwardDevice` (the
function the registry hook `ForwardDeepseekV4ForCausalLM` runs on the runner's
`ModelRegistry::Forward` gather-logits path) now returns **device-resident** logits on
a CUDA queue (`ForwardLogits::on_device()`) via a new `WrapV4DeviceLogits` helper that
mirrors `qwen3_moe.cpp WrapDeviceLogits` — dropping the `.host` logit download so the
runner's on-device sampler consumes it like every framework-conforming model. DS4's
keep-quant forward is GB10-unified by construction (every GEMM reads/writes coherent
unified-memory views in place; `ForwardComposeImpl` already drains the lm_head GEMM),
so the composed buffer is device-addressable + complete and ownership moves into the
`shared_ptr` the runner holds across execute_model→sample. CPU queue keeps the
byte-identical `.host` path (the portable host oracle + every CPU gate). CPU build
`-Wall -Wextra -Werror` clean; the tiny-synthetic CUDA composition gate
(`test_cuda_deepseek_v4.cpp`) updated to consume the device-resident return (backend
`Copy` → host for the near-tie compare, mirroring the runner's `VT_GPU_SAMPLE=0` path).
**OWED:** the CUDA unit gate + a real-model coherence/tok-s re-gate on GB10 (deferred —
box at 99% disk, DS4 loads ~86 GiB, no-regression re-gate of the shipped resident/graph
default was not run). **Scoped follow-up (not done):** DS4's private resident/graph
DECODE (`ForwardResidentDecodeGguf`/`V4Graph`, CLI-driven) + its own `DeepseekV4KvCache`
off the shared `AttnBlock`/`PagedKvCache` remain off-runner — the larger KV/attention
port is the next brick. See `.agents/specs/deepseek-v4-forward-device.md`.

**DeepSeek-V4-Flash decode ds4-gap — Brick 14 (Q8_0 intra-row register-prefetch —
the ds4 raw mechanism)** (2026-07-31, `CLAIM-DSV4-DECODE-BRICK14`, GB10 sm_121a,
branch `ds4-q8-register-prefetch`, NOT pushed). Built `QuantDotGemmQ8_0PrefetchKernel
<OutT,PF>` (`src/vt/cuda/cuda_quant_dot.cu`, env `VT_V4_Q8_PREFETCH` default-OFF `=1`,
opt-in `=2`/`=4`, wired in `MatmulQ8_0Cuda` so both eager forward and captured
`V4Graph::Step` inherit it): same one-row-per-warp map, per-lane block loop
unroll-and-JAMmed by PF so PF weight+act block loads are hoisted into registers
before the dependent `__dp4a` chains (the ds4 register-resident-MLP mechanism from
`ds4-q8-raw-mechanism-2026-07-30`). BIT-IDENTICAL to the plain kernel (same block/dp4a
order + f16-scale fold; `test_cuda_quant_dot` Brick-14 A/B `1/1·1968`; real-model
token-identity OFF==PF2==PF4 16/16). **sudo-ncu (`--replay-mode application` +
memory watchdog) grid-256 (n=2048): registers 39→64→96, long-scoreboard 56.1→30.7→28.4
(toward ds4's 17.2), occupancy 72→55→30% (toward ds4's 60%) — the mechanism REPRODUCES —
but per-kernel GPU-active FLAT (46.06→46.27→46.19 µs) and clean decode tok/s FLAT
(13.145→13.140→13.133).** MEASURED-NEGATIVE: lowering long-scoreboard ~2× at ds4's
register/occupancy operating point yields ZERO throughput ⇒ long-scoreboard is NOT the
causal gate on the 67%→90% BW gap (the raw-mechanism premise is refuted; the GEMV is
bound by DRAM byte-delivery, not latency-exposure). CORRECTS Brick 13's "intra-row is
the untried right axis" — that axis hits ds4's counter target and still yields nothing.
Kept DEFAULT-OFF like Bricks 4/8/11/12/13; ~13 tok/s stays the honest Q8_0-kernel ceiling
and the raw beat of ds4's 16.5 is not reachable via any Q8_0-GEMV-kernel-structure lever
(the residual is the MoE + glue front). See `.agents/specs/deepseek-v4-last-mile.md`
Brick 14 + `docs/BENCHMARKS.md`.

**DeepSeek-V4-Flash decode ds4-gap — Step 0 + Lever 1** (2026-07-30,
`CLAIM-DSV4-DECODE-LEVER1`, GB10 sm_121a). **Step 0** (`kWarpsPerBlock` 4→8 on the
dense Q8_0 GEMV, ds4's 8-rows/block): bit-identical, MEASURED NEUTRAL
(`QuantDotGemmQ8_0Kernel` 63.2 µs/launch unchanged; the 41 ms GEMV is
memory/latency-bound). **Lever 1** (`QuantizeQ8_0PreqKernel`, `VT_V4_Q8_PREQ_QUANT`
default-ON): ports ds4's warp-per-32-block quant grid, dropping the dense Q8_0
activation-quant 4.51→0.99 ms/step (6.98→1.53 µs/launch, ds4-parity), in the
captured decode graph body. BIT-IDENTICAL (`test_cuda_quant_dot` 4/4·106081,
preq==legacy byte-for-byte), token-exact. **Decode 11.44→11.92 tok/s (+4.3%,
non-overlapping); SHIPS default-on.** Go/no-go for the 11.41→16.5 campaign =
**PARTIAL GO**: Gate A(ii) 9.1→<1 ms is NOT met — the `QuantizeQ8K` half (4.60 ms,
grouped MoE) is Brick-8-refuted for fusion and Brick-2-deduped, so the plan
mis-scoped it into Lever 1; realized +0.49 tok/s vs projected +2.3, so the ladder
to 16.5 must be re-based. See docs/BENCHMARKS.md.

**DeepSeek-V4-Flash decode ds4-gap — Lever 3 (route → warp-topk)** (2026-07-30,
`CLAIM-DSV4-DECODE-LEVER3`, GB10 sm_121a, base `baff8a28`). `RouteWarpKernel`
(`VT_V4_ROUTE_WARP_TOPK` default-ON) structure-ports ds4's `router_select_warp_topk`
(`ds4_cuda.cu:10113`): the E=256/topk=6 expert top-k, previously ONE thread per token
at T=1, now runs one WARP per token (8 experts/lane, per-lane argmax → `__shfl_xor`
reduce → mask/repeat). **BIT-IDENTICAL** — argmax under a strict total order
(value, tie-break lower index) is reduction-order-invariant, and the unbiased weight +
our exact renorm run the same float ops in j-order; kept OUR arithmetic, not ds4's.
Wired via a shared `RouteDispatch` into BOTH the eager forward AND the captured
`V4Graph::Step` (nsys shows `RouteWarpKernel` in the graph body, 43 launches/step).
**GATE B (bit-exact, RED-first):** `test_cuda_deepseek_v4` 20/20·67072 (new A/B case:
warp == single-thread BYTE-IDENTICAL ids AND weights; deliberate perturbation goes RED),
`test_cuda_quant_dot` 4/4·106081, `test_deepseek_v4_gguf_load` 15/15·931; real model
warp-ON == legacy-OFF golden `11111 16 455 6102 294 8760 344 …` (TOKEN-EXACT).
**Re-profile (nsys 30-step kwin):** route `RouteKernel` 5.9723 → `RouteWarpKernel`
0.3779 ms/step (138.89 → 8.79 µs/launch, **−5.59 ms, 15.8×**, 4.7% → 0.5% GPU-active).
**Honest realized-vs-projected:** the study's route magnitude (6.5 ms → <0.5 ms) was
ACCURATE this time; −5.59 ms of ~81.6 ms/step = −6.9% GPU-active → GPU-active-bound
~+0.7–0.8 tok/s (11.92 → ~12.7). Wall-clock tok/s NOT cleanly resolvable this session
(no `sudo drop_caches`; churned-box page-cache variance swamps the ~0.7 tok/s delta).
SHIPS default-on (bit-exact, real GPU-work win). See docs/BENCHMARKS.md.

**DeepSeek-V4-Flash decode ds4-gap — Lever 2 / Brick 11 (Q8_0 sub-warp GEMV tiling) —
MEASURED NEGATIVE, recorded** (2026-07-30, `CLAIM-DSV4-DECODE-LEVER2`, GB10 sm_121a,
base `d1ff7414`). The plan's decisive speculative bet: the `QuantDotGemmQ8_0Kernel`
(41.1 ms/step = 52.7% GPU-active, the whole remaining ds4 gap) is 1-warp-per-output; at
short K a 32-lane warp was theorized to waste ~50% of lanes and starve the device, so a
templated sub-warp kernel (LANES∈{32,16,8}, `nb`-dispatch, ground: ds4
`quarter/half_warp_sum_f32` `ds4_cuda.cu:16609`, moe sub-warp `:17073`) should raise
lane-utilization/occupancy. **MEASURE-FIRST REFUTED THE PREMISE:** the DS4-Flash decode
Q8_0 projections have **nb∈{32,64,128,256}** (K∈{1024,2048,4096,8192}) — the smallest,
nb=32, **fills all 32 lanes (1 block/lane, ZERO idle lanes)**; there is no nb∈{16,48}
lane-waste regime, and that lone nb=32 projection (n=32768) issues 4096 blocks — NOT
device-starved. **nsys 30-step kwin, VT_V4_Q8_SUBWARP A/B, worker down, flock:** total
Q8_0 **41.11 → 41.06 ms/step (flat, −0.1% = noise)**; the ONLY sub-warp-eligible
projection (nb=32, LANES=16, grid 4096→2048) **9.13 → 9.19 ms (marginally WORSE)** —
well under the study's ~2 ms "worth it" bar. Interleaved warm wall-clock ~12.7 tok/s
both arms (off {12.67,12.80,12.77} / on {12.82,12.71,12.70}, overlapping); token-exact
golden `11111 16 455 6102 294 8760 344 …` byte-identical off==on (the near-tie flips zero
tokens). **HARD STOP-CONDITION MET → cut losses, no more variants.** Gate: A/B unit case
`test_cuda_quant_dot` 5/5·106483 (sub-warp==plain, LANES=32 byte-identical big-K + NMSE≤5e-4
short-K, RED-first), `test_cuda_deepseek_v4` 20/20·67072, `test_deepseek_v4_gguf_load`
15/15·931. Merged **default-OFF** as a guarded recorded-negative (`VT_V4_Q8_SUBWARP=1`,
`VT_V4_Q8_PROBE` for the nb-attribution). **Honest reachable ceiling:** L1 (+0.49) +
L3 (route −5.59 ms GPU) bank **~12.7 tok/s**; the Q8_0 GEMV is the **irreducible
DRAM-bandwidth residual** for THIS model's nb≥32 dims (occupancy/tiling has nothing to
fix), so **~13 tok/s is the honest ceiling and 16.5 is NOT reachable via this lever** —
a valuable measured-negative like Bricks 4/8. See docs/BENCHMARKS.md.

**DeepSeek-V4-Flash decode ds4-gap — ds4's Q8_0 kernel DIRECTLY PROFILED → the
"irreducible roofline" is REFUTED, lever RE-OPENED** (2026-07-30,
`CLAIM-DSV4-Q8-KERNEL-PROFILE`, GB10 sm_121a, measurement-only). Nobody had ever
profiled ds4's own Q8_0 GEMV; done now (nsys `cuda_gpu_kern_sum` + GGUF byte-accounting
— `ncu`/`nsys --gpu-metrics` both `ERR_NVGPUCTRPERM`, no admin, so L2-hit is unmeasurable
and achieved-BW is the counter-immune proxy). Both engines read the **identical 6.15 GiB
Q8_0 tower once/step** (GGUF `--inspect`: 345 q8_0 tensors; ds4's fp16-dequant cache was
budget-exhausted this run → it reads the same raw 34-B blocks). **Diff (pure decode/step,
`(-n60 − -n4)/56`): ours `QuantDotGemmQ8_0Kernel` = 40.93 ms over 646 launches = 161 GB/s
= 67% of the 240 roofline; ds4 = 30.38 ms over 259 launches = 217 GB/s = 90% of roofline.**
Same bytes (reading >6.6 GB in 40.9 ms would exceed 240 GB/s — impossible — so our 646 are
row-splits summing to the same tower, not re-reads). **Root cause = launch consolidation,
NOT the inner dot / bytes / alignment / lane-occupancy (Bricks 3/4/8/11 all correctly flat):
ds4 packs the tower into fewer, bigger launches** — `matmul_q8_0_pair_preq_warp8` (2 weights
+ 1 shared activation/launch, 38.8 vs our 63.4 µs/matmul = **1.63×**), `hc_expand_preq`
(MHC-expand fused into the matmul epilogue), `grouped_q8_0_a` — keeping DRAM saturated,
where our 646 tiny warp-per-row launches under-subscribe it. Brick 11 built sub-warp
*lane-splitting* (within-launch) and correctly found it flat; it never changed the launch
COUNT/SIZE — the one axis that moves 67%→90%. **VERDICT (A) PORTABLE LEVER: build Q8_0
projection-pairing (mirror `matmul_q8_0_pair_preq_warp8_kernel`, bit-exact) + fold the
MHC/residual epilogue in (near-tie); target 40.93→~30.4 ms (−10.5) → ~13.9 t/s (from 12.1
GPU-active), does not reach 16.5 alone (MoE+glue residual ~15 ms is a separate front) but
re-opens the ladder.** Premise correction: the direct Q8_0 gap is **10.5 ms/step**, not the
older whole-step-derived "19.8 ms". No code changed. See
`.agents/specs/ds4-q8-kernel-profile-2026-07-30.md` + docs/BENCHMARKS.md.

**DeepSeek-V4-Flash decode ds4-gap — Brick 12 (IMPL): Q8_0 LAUNCH CONSOLIDATION built —
hits ds4's EXACT 259-launch structure BIT-EXACTLY, but the projected bandwidth win does
NOT materialize; the "launch count/size → ds4's 90%" hypothesis is REFUTED** (2026-07-30,
branch `ds4-q8-launch-consolidation` off `1ddcc42c`, GB10 sm_121a, `VT_V4_Q8_PAIR` default-ON,
rollback `=0`, NOT pushed). Three bit-exact consolidations wired into BOTH the resident-decode
`Step` and the captured `V4Graph` RunChain: **projection-pairing** (`QuantDotGemmQ8_0PairKernel`,
port of ds4 `matmul_q8_0_pair_preq_warp8_kernel` `:4485`) fuses the two A-projections sharing the
layer hidden — MLA `wq_a`+`wkv` and shared-expert `gate`+`up` — into one launch each (activation
quantized once); **block-diagonal o-LoRA** (`QuantDotGemmQ8_0GroupDiagKernel`, ds4
`grouped_q8_0_a_preq_warp8_kernel` `:5509`) collapses the ng=8 per-group `wo_a` GEMVs (344/step,
53% of the 646) into ONE launch; the MHC-expand epilogue fold was DEFERRED. **MEASURED (nsys
`cuda_gpu_kern_sum`, diff `(-n60 − -n4)/56`): launches/step 646 → 259 (EXACTLY ds4's 259: 130
plain + 86 pair + 43 group-diag); the pair kernel is 37.3 µs/matmul-equiv, at/under ds4's 38.8.
Q8_0 GPU-active/step 40.96 → 39.86 ms (−1.10, −2.7%); achieved BW 161 → 166 GB/s (67.1% → 69.0%
of 240 — NOT the projected ~90%/30.4 ms); wall decode 12.24 → 12.27 tok/s (flat, within
page-noise).** HONEST PARTIAL — the launch-count target is fully + bit-exactly met yet the
DRAM-saturation time win is absent: under `VT_V4_DECODE_GRAPH=1` the 646 launches were ALREADY one
`cudaGraphLaunch`, so collapsing graph nodes 646→259 removes host overhead the graph had eliminated
and does not raise on-device BW. **ds4's 67%→90% advantage is therefore NOT launch consolidation**
(this build reproduces ds4's launch structure and stays at 69%) — it must lie in ds4's fp16-dequant
weight cache or L2 residency (counter-blocked). Token stream **byte-identical ON==OFF** (golden
`11111 16 455 6102 294 8760 344 …`, 16/16). GATES (RED-first): new `test_cuda_quant_dot` A/B cases
(pair==two-separate + group-diag==per-group-loop byte-identical) **7/7·106496**,
`test_cuda_deepseek_v4` **20/20·67072**, `test_deepseek_v4_gguf_load` **15/15·931**. Bit-exact,
ds4-faithful, DEFAULT-ON (marginal-positive, not a regression). NOT pushed. See
`.agents/specs/deepseek-v4-last-mile.md` (Brick 12 IMPL) + docs/BENCHMARKS.md.

**DeepSeek-V4-Flash decode ds4-gap — Brick 13 (IMPL): Q8_0 ILP lever (N output-rows per
warp) — MEASURED NEGATIVE via sudo-ncu; ~13 tok/s is the honest Q8_0 ceiling, Q8_0 front
CLOSED** (2026-07-31, `CLAIM-DSV4-DECODE-BRICK13`, branch `brick13-q8-ilp` off `137c739f`,
GB10 sm_121a, `VT_V4_Q8_ILP` default-OFF, opt-in `=2`/`=4`, NOT pushed). The one untried,
measurement-pointed lever from `ds4-q8-ncu-2026-07-30` (our `QuantDotGemmQ8_0Kernel` is
memory-LATENCY-bound: long-scoreboard 54.4 @ 71.9% occupancy, L1-hit 96.7%): raise
memory-level parallelism per thread with N INDEPENDENT weight-load streams. Built
`QuantDotGemmQ8_0MultiRowKernel<OutT,NROWS>` — each warp computes NROWS consecutive output
columns of the same activation row (activation read once, `__dp4a`'d against NROWS independent
weight rows), **BYTE-IDENTICAL** to NROWS plain outputs — wired in `MatmulQ8_0Cuda` (eager +
captured `V4Graph::Step`). **CAUSAL METRIC (sudo `ncu`, regex `QuantDotGemmQ8_0`, decode
steady, n-keyed):** the target n=2048 kernel long-scoreboard **57.8 → 87.0 (ILP2) → 51.7
(ILP4)** (did NOT drop materially), achieved occupancy **71.4% → 42.5% → 21.3% (COLLAPSED)**,
per-launch GPU-active **46.7 → 44.9 → 51.0 µs (flat-to-WORSE)**; n=4096 LS **56.4 → 122.2 →
91.3**. **Clean wall-clock (sudo `drop_caches`, 3 reps, `--max-tokens 64`):** OFF **13.19**
median, ILP2 **13.12 (−0.5%)**, ILP4 **13.05 (−1.0%)**. ROOT (measured): the load latency was
already hidden by INTER-warp parallelism (71% occupancy), not starved for intra-thread ILP;
folding N rows/warp divides the warp count by N and adds registers ⇒ occupancy collapses and
removes more latency-hiding than the extra streams add. Token stream **byte-identical
OFF==ILP2==ILP4** (golden `11111 16 455 6102 294 8760 344 …`, 16/16). GATE (RED-first,
byte-identical): `test_cuda_quant_dot` **8/8·108464** (new Brick-13 A/B `1/1·1968`, ILP2==ILP4==plain
over nb∈{16,48,224}, n∈{1,7,16,17} incl. tail n∤N). **HARD STOP-CONDITION met on BOTH N-values →
RECORDED-NEGATIVE, kept DEFAULT-OFF** (bit-exact-safe, like Bricks 4/8/11/12). This was the LAST
measurement-pointed Q8_0-GEMV lever: **~13 tok/s is the honest Q8_0-kernel ceiling on GB10 and
the campaign's Q8_0 front is CLOSED**; ds4's 67%→90% edge stays attributable only to the
unobservable fp16-dequant weight cache / L2 residency (`ERR_NVGPUCTRPERM`-blocked). NOT pushed.
See `.agents/specs/deepseek-v4-last-mile.md` (Brick 13) + docs/BENCHMARKS.md.

**ngram** (method `ngram`, draft-FREE) proposes the next tokens by matching the
sequence's own suffix n-gram, so it needs no draft model and works on any model;
on the 27B it is token-exact vs vLLM's own `--speculative-config ngram` on
repetitive workloads (5/5 prompts, every draft accepted).

**Generic separate draft-model** (method `draft_model`, `SPEC-DRAFT-MODEL`
ACTIVE) is spiked with its first CPU brick landed, not yet wired into any
production path. The classic model-agnostic path runs a full smaller standalone
draft LM K autoregressive greedy steps to propose K tokens, which the target
verifies in one forward (longest-accepted-prefix), reusing the landed rejection
sampler unchanged (only the proposer is net-new). W1 landed the greedy propose
(`DraftModelProposeGreedy` over a next-token-logits oracle) + the `draft_model`
`--speculative-config` accept, unit-gated RED-first (`test_draft_model_proposer`
6 cases / 41 assertions: the accepted tokens equal the target's own greedy run
for every draft/target agreement pattern, a target-matching draft is fully
accepted, and full acceptance depends on the autoregressive feed-back). NOT yet
wired: the real GPU draft-model forward behind the oracle (paged KV, CUDA-graph)
+ the e2e greedy our-ON==vLLM-ON token-exact gate + the throughput speed gate are
DGX-offline residuals (W3). **Medusa** (method `medusa`, `SPEC-MEDUSA` SPIKE) is
spiked only; its N-head single-pass proposer is deferred to W2 (needs the
target's Medusa heads). See docs/BENCHMARKS.md.

The correctness form and the full D0-D14 measured chronology live in
[docs/BENCHMARKS.md](BENCHMARKS.md),
[docs/SPECULATIVE-DECODING.md](SPECULATIVE-DECODING.md) and
[.agents/specs/dflash-spec-decode.md](../.agents/specs/dflash-spec-decode.md).

CPU elementwise GEMM, wide x86 tiers (2026-08-07): AVX2 (16 outputs as two groups of 8, mr 4) and
AVX-512 (16 outputs in one ZMM, mr 6) elementwise tiers replace SSE2 when the runtime probe finds
them, each in its own ISA-flagged TU. Byte-identical to the portable tier on every tier and dtype:
widening adds OUTPUT lanes, each output's K reduction stays sequential, and products are
mul-then-add, never FMA. Speed is indicative only, the x86 box is VOID for timing.

CPU elementwise GEMM, transpose-free `[K,N]` path (2026-08-07): M-blocked `[K,N]` micro-kernels on every
tier, the `mr` dispatch degated from `[N,K]` only (it forced mr=1 on `[K,N]`, so a 131-row activation
re-read the whole weight 131 times), and an opt-in load-time `[N,K]` to `[K,N]` repack
(`VT_CPU_ELEM_KN_REPACK`, CPU-only, default off) so `vt::MatmulBT` can reach that family at all.
Byte-identical: both orientations accumulate each output over K in strict increasing order, so the
repack is a layout choice and never a numerical one.

Parakeet ASR (2026-08-07): *CPU-correct, ON THE ONE SURFACE (ROW 1)*. Ids exact vs HF; registered transcription-only; `vllm_transcribe` (ABI v11) + `/v1/audio/transcriptions`; thin `vllm.h` example, golden-exact; ratchet 12 -> 11. No CUDA, no speed number.

## Not supported yet

LoRA (W1 CPU runtime brick landed; not yet usable end-to-end), multi-GPU,
hybrid CPU+GPU device placement (`ENG-HYBRID-PLACEMENT` READY, spec only:
routed-MoE expert COMPUTE on the CPU backend with the rest on GPU, the inverse
of weight offload; #149's CPU-MoE half. Its speed floor vs llama.cpp `-ncmoe`
needs a discrete-GPU rig, which unified-memory GB10 cannot provide, so that
gate is blocked rather than pending
[spec](../.agents/specs/hybrid-placement.md)),
Vulkan (opt-125m exact; 25 native +8 GDN, both
recurrences + fused attn preamble; 27B prefill 21.5x, decode
4.36/4.35 MET; 27B load 100.8 -> 53.4 GiB; #125
[campaign](../.agents/specs/vulkan-full-support.md)), ROCm (W0 community-green
on 5 gfx archs; the APU unified-memory fix remains unverified; gfx1200 runs
Gemma-3 and Qwen3 all-native, with Gemma-3 strict 48/48 against two vLLM-ROCm
oracles and Qwen3 in a measured near-tie regime; on gfx1100 the M4 gate now
runs against the **pinned vLLM-ROCm oracle built on the same box**
(`555967922` / `0.23.1rc1.dev1511+g555967922`): Qwen3-0.6B **16/16 PASS**
(11/16 strict token-exact, 5/16 near-tie band, max teacher-forced gap 0.125
nats, 0 forward-divergent; oracle K=10 deterministic in every cell) with the
ROCm device-golden lane in `test_qwen3_paged_engine`; Qwen3.5-0.8B GDN runs
all-native and is gated **16/16 against the same-box pinned oracle** (15/16
strict token-exact, max gap 0.125 nats) since the `AttnQkNormRopeGate`
output-dtype dispatch fix; gfx1201 Gemma-4 FP8 MoE is contributor-measured on
2x R9700 and CPU-link-verified our side; a `head_dim=128` decode arm lands
opt-in behind `VT_ATTN_DECODE_D128`, default OFF, which moves gfx1200 per-token
decode from 6.35x to 1.75x slower than the pinned vLLM oracle on one shape, a
directional figure that leaves the ROCm throughput axis PENDING; #1047 item 3 on gfx1201 Gemma-4 FP8 prefill-peer is attribution only (T=2029: deleting two Finish success-path barriers is +2.55% / 46.05 ms/req vs the retained RetirePinThenUnpin wait, not a license to land the deletion and not a ROCm throughput pin; [BENCHMARKS](BENCHMARKS.md));
[guide](ROCM.md)), inference-time CPU weight offload (`ENG-WEIGHT-OFFLOAD`
ACTIVE; the config surface landed W0a (the backend enum, both sub-configs, the
validator's two errors and three warnings, and the dot-anchored segment match),
all UNREACHABLE for now because nothing constructs an `OffloadConfig` yet, so
no engine behaviour changes. Still owed: vLLM's `cpu_offload_gb` UVA arm with
dotted-segment `cpu_offload_params` targeting, plus the layer-group `PrefetchOffloader` — a
pure mirror floor, and #149's dense half. Its memory and speed gates need a
discrete-GPU rig, because on unified-memory GB10 offloading to "CPU" frees
nothing, so those gates are blocked rather than pending
[spec](../.agents/specs/weight-offload-uva.md)), the disk-residency tier as a
config surface (`ENG-RESIDENCY-CONFIG` ACTIVE, #1110: mmap residency, prefault
and the expert-stream lane are reachable from `--offload-config`'s `vllm_cpp`
key, not only from `VT_*`, which keep working as overrides; env var beats config
beats default; the 370 GiB reproduction through the JSON form is owed and needs
GB10 [spec](../.agents/specs/weight-residency-config.md)), and the full
tool-calling template surface. **Muse Glimmer's
GGUF arm generates coherently** (#347, #359), is NOT token-exact, and has
only a llama.cpp bar (#333). Its CPU decode was **synchronisation-bound, not
kernel-bound**: the threadpool's never-yielding spin-wait cost a full scheduler
timeslice per dispatch whenever the pool was wider than the cores available to
it, which the default `hardware_concurrency()` width reaches on its own. A
bounded spin then `sched_yield` (#391 Lever 1, 2026-08-11) takes in512 decode
**3.41x** and in128 prefill to **1.023x of llama.cpp**. That denominator is stock
`7044859`, SUPERSEDED and owed a re-take against the `b10451` pin (#1003), and
the 2.3% margin sits inside our own arm's 4.5% leg spread. The fix also collapses the
run-to-run decode spread from 73.3% to 15.2%. Prefill at 512 input tokens is
**unmoved** and remains the open half of the gap (#391 Lever 2). **Scale-out / distributed execution is scoped, with two legs landed
CPU-gated** (2026-07-28): one `vt::` collective / process-group abstraction
with backend transports (NCCL / RDMA / MLX-ring) mirrors vLLM's
`device_communicators` across multi-GPU TP+PP, 2×DGX-Spark over ConnectX-7
RoCE (the DeepSeek-V4-Flash fp8 ~167 GiB across 2×119 GiB path), and MLX
multi-node; `world_size==1` stays byte-identical. Landed: `vt::Communicator`
(in-process multi-rank CPU gate) and the same-host TP wiring
(multi-device registry, `OpId` routing, the NCCL transport behind
`-DVLLM_CPP_NCCL=ON`, `TensorParallel` in the Qwen3-dense forward equal to
tp=1). The **real TP-2 GPU run + NCCL build-verify remain HW-blocked** (no
≥2-GPU box), as do PP, multi-Spark and MLX
([scale-out spec](../.agents/specs/scale-out-distributed.md); every
parallelism mode is enumerated and ranked in
[parallelism-modes](../.agents/specs/parallelism-modes.md), noting vLLM's
"sequence parallel" is a TP compilation pass, not an axis).
**Shared layer names no device**: async readback is a `Backend` capability.

**Tensor parallelism is scoped end-to-end at the pin** (#287,
[tensor-parallelism-spike](../.agents/specs/tensor-parallelism-spike.md)):
~40% of the TP surface landed/reusable; **TP-W1 LANDED** (group table); W2..W4+W7
(the engine-level TP2-on-CPU token-exact gate) are CPU-completable
NOW; only NCCL + gate-model perf wait on hardware. The DSpark speculator
(DFlash-derived block drafter for our Qwen3 + Gemma4 families) is
**IMPLEMENTED and MEASURED cross-engine**
([spec](../.agents/specs/dspark-spec-decode.md) §6h, §6j): against the pinned,
graphed oracle the 35B-A3B MoE lane runs at 0.92-0.98x with acceptance matching
upstream (20.8% vs 20.4%). On matched content our proposals are near
token-identical to upstream's and acceptance is at parity (12.1% vs 11.1%); the
dense lane's 0.35x cell is void, because upstream's arm there degenerated into a
repetition loop (8 distinct token ids across 128 tokens) that is trivially
draftable. The open lever was per-step cost: 28% of the draft step was
host-side sampling, now moved on device (#436, byte-identical output, sampling
-11%/-15%, ~3% e2e on the 35B lane); what remains of that phase is a memory
bandwidth bound on the per-step Markov GEMV that upstream pays identically.
W8 (#442) then CAPTURED the T=1+k verify, mirroring vLLM's uniform-decode graph
predicate (`uniform_decode_query_len = 1 + num_speculative_tokens`) instead of
our `query_len == 1` gate: the MoE lane measures **0.996x-1.012x** of the
pinned graphed oracle ON MATCHED WORK, byte-identical, with the same-binary
capture A/B worth +12.2%/+3.5%. Upstream's speculative decode is not
run-to-run deterministic (its fast draw on one prompt is ONE extra accepted
token, 17 steps instead of 18), and our acceptance equals its modal value
(48.6%, 4.94 tokens/step). Per-step the engines are aligned (30.4 vs ~30.1 ms,
34.7 vs ~34.5). Under a LOW-NOISE harness (clocks pinned at 1800
MHz, 30 reps, drift bracketed at -0.088%, the oracle's non-modal draws excluded)
that session's code cell measured **0.975x with NON-OVERLAPPING
distributions** — a real gap, not noise, and the earlier "within resolution"
reading was too generous. It is one of four within-session ratios, listed
below. Ours slowed
more than the oracle when the clock was pinned, so the residual is
SM-clock-sensitive work. PAIRED profiling appeared to localise the residual to
`marlin_moe_wna16::Marlin` (ours 249.22 ms vs upstream 230.39 ms over the same
1520 launches), and every source-level explanation was eliminated -- kernel
source, template instantiation, grid, block size, shared memory, flags, scale
layout, alignment, residency, toolkit, arch -- with ncu and cuobjdump showing
the compiled kernels EQUIVALENT (94 registers, 3664 SASS instructions on
both). THAT LOCALISATION IS REFUTED. `scripts/marlin-moe-standalone.py` and
`benchmarks/marlin_moe_standalone.cpp` drive each engine's own kernel outside
its engine -- which is also what finally lets ncu attach to upstream,
previously recorded as impossible in both replay modes -- and at matched work
the two are indistinguishable: over 12 interleaved paired points ours averages
5.3187 us/block against upstream's 5.3330, ratio 0.9973, sign flipping between
runs, inside one standard deviation. The in-situ 8.2% therefore describes the
RUNS, not the kernel, and so do the 12.8%-per-unit-work and 186.6-vs-210.7
GB/s figures derived from it -- the latter also divided GRAPHED times by
EAGER-mode block counts, so numerator and denominator came from different
execution modes. END-TO-END, valid within-session paired ratios are 0.9757,
0.9646, 0.9569 and 0.9889 (a fifth run was REJECTED on a -2.13% drift gate): a
spread of 0.957-0.989 ACROSS BOOTS, with no single value being the ratio and
the gap not resolved better than 1-4% on this hardware. An earlier claim here
that the gap was 1.1% rather than 3.4% is WITHDRAWN: decomposed, our arm moved
+1.10% between those sessions while the ORACLE denominator moved -2.17%, so
most of it was the oracle's boot state, and differencing ratios across boots
is what this page forbids elsewhere. The warm-up arm does demonstrably remove
a 6.6% WITHIN-RUN drift, which makes a run internally valid without moving the
ratio. Standing traps: dram__bytes.sum reads n/a on GB10, so ncu's Memory
Throughput % excludes DRAM traffic and is not a bandwidth utilisation; the GPU
lock is $HOME/gpu.lock, and absolute timings from runs that took /tmp/gpu.lock
ran unserialised and are LOWER bounds on achievable bandwidth, so a clean re-
take can only raise the plateau; any MoE comparison that lets routing vary
between arms measures the draw, not the change; and both blocks AND distinct
experts must be controlled, since cost per distinct expert spans 4.47-7.50 us
and is flat only above ~40 experts. NOT parity, and the row stays open.
The environment was REBUILT after the reimage (engine, both checkpoints at the
pinned revision, and the TRUE pinned oracle 555967922 + torch 2.13.0 +
flashinfer 0.6.15.post1, built from source because vllm==0.26.0 hard-pins
torch==2.11.0). On it, with generation length MATCHED at 89 tokens and the
container compile cache PERSISTED, the oracle is stable at 171.4 tok/s and our
engine at 143.2, a paired ratio of **0.835** -- far below the 0.957-0.989 this
row recorded. Two causes are possible and this data cannot separate them:
every earlier paired run invoked the oracle ONCE, so a cold-JIT denominator
would have been handicapped ~17%; and the BOX IS NOT THE SAME MACHINE, since
dgx.casa now resolves to kairos-17dd while the recorded ratios were taken on
promaxgb10-4ad8, which no longer exists. Our arm reads ~142 on both, which
argues against a pure hardware explanation without excluding one. Safe to say:
on this box, matched and warm, we are at 0.835 of the pinned oracle, NOT
parity. Not safe to say: that the older numbers were wrong, since cross-
machine ratios cannot be differenced any more than cross-boot absolutes can.
Owed: re-run the pre-reimage single-cold-invocation protocol HERE -- ~0.97
would convict cold JIT, ~0.83 would mean the machine changed.



Multimodal
(image/video/audio) is correctness-complete and its OpenAI-server wiring has
landed all three CPU bricks (content-part parse + processor routing, the
default-inert engine mm-request plumbing, and the `MultiModalChatFn` seam body
rendering an image chat request into the placeholder-EXPANDED engine input).
It is not yet servable end-to-end: the closing GPU gate `MM-SERVE-E2E` is
blocked on the engine model runner having no multimodal forward
(the vision tower + merge lives in the standalone M2c driver, not in
`ModelRegistry::Forward`) — folding that forward into the registered engine path is
the named residual (`.agents/specs/mm-serving.md`).

**Memory budgeting is manual and behind vLLM** (`ROAD-V1-MEM`,
[#83](https://github.com/mudler/vllm.cpp/issues/83), scoped 2026-08-06): there is
no device memory profiling yet, so the DEFAULT KV pool is still a raw block
count (`--num-blocks`, default 256). Target shape, rowed in
[.agents/roadmap_v1.md](../.agents/roadmap_v1.md): auto-size to the declared
workload by default, cap the TOTAL engine footprint when asked, refuse before
allocating with a computed remedy breakdown rather than an OOM. **M1+M2 LANDED** (`specs/kv-sizing.md`): `--kv-cache-memory` sizes the pool via a group-aware divisor (ABI v16); M3 profile run dgx-gated.

**Open, not root-caused (observed 2026-07-28):** the C-ABI custom logits
processor case (`tests/capi/test_capi.cpp:410`, ABI v8) SIGSEGVs in a CUDA build
on the GB10 box, while the same suite is 232/232 green on a CPU build. Confirmed
present on pristine `main` `ee3d5960` with no local changes, so it is not a
regression from any in-flight work, but it does mean the "supported" claim for
custom logits processors in the capability table above is verified on CPU only.
One build directory, not yet bisected.

## Model family notes

### Gemma

**Gemma 3** (`Gemma3ForCausalLM`, `google/gemma-3-1b-it`) is the **first landed
member** (correctness-complete, speed-pending): STRICT token-exact 48/48 greedy
vs the vLLM 0.25.0 oracle. It reuses infrastructure already in the tree
(gemma-RMSNorm `(1+w)`, the GLM sandwich norms, the shared dense-attention path,
sliding-window attention, tied embeddings) plus one genuinely-new compute
kernel, GeGLU (`gelu_pytorch_tanh`), a bf16 embedding-scale multiply, dual
per-layer RoPE theta, and query_pre_attn_scalar scaling. **Gemma 2**
(`Gemma2ForCausalLM`, `gemma-2-2b-it`) and **Gemma 1** (`GemmaForCausalLM`,
`gemma-2b`) have since landed too, both correctness-complete and speed-pending
at 48/48 greedy vs the same oracle
([spike](../.agents/specs/sweep-gemma.md)): Gemma 2 proves the attention + final
logit soft-cap primitives, Gemma 1 the original two-fused-norms block.

**Gemma 4**, the newest registered variant, now has its **text path
correctness-complete and gated** (G1b, 2026-07-28): the
`Gemma4ForConditionalGeneration` language_model stack of `unsloth/gemma-4-E4B-it`
loads through our engine and greedily emits the **exact 32 golden token ids —
STRICT 32/32 token-exact vs the vLLM 0.25.0 golden** (gate
`tests/parity/test_gemma4_paged_engine.cpp`, dgx CUDA). It brings up the large new
primitive stack (per-layer embeddings, YOCO KV-sharing, plain RMSNorm, proportional
partial-RoPE, heterogeneous 256/512 head dims, GeGLU, a per-layer scalar; the
Gemma-4 MoE / k_eq_v / double-wide MLP are off for the E4B checkpoint and stay the
larger-variant follow-on). The G1-named blocker is resolved: **the runner now
allocates a per-layer KV head dimension** (`KVCacheConfig::per_layer_attn_specs`),
so Gemma-4's per-layer 256/512 heads each get a correctly-strided cache — and it is
byte-neutral for every uniform-KV model (the field is empty ⇒ the previous single
uniform allocation; the full CPU runner/KV suite plus the OLMo-2 SACRED GPU gate
16/16 are unchanged). Three additive loader gaps were fixed en route to the
first-ever Gemma-4 forward: nested per-layer `rope_parameters` (config loader), the
Gemma `Replace(" "->"U+2581")` metaspace normalizer (tokenizer), and reading
Gemma-4's per-arch scalars from `raw["text_config"]` rather than the full config
(the real 256/512 head-dim source). Speed vs vLLM is pending. Its
multimodal path (image + video + **audio**, the only
audio-capable model in the pin) has been assessed
([spec](../.agents/specs/gemma4-multimodal.md)) and is now **oracle-gateable —
run-verified** (W0, 2026-07-28): the pinned 0.25.0 oracle (transformers 5.13.1)
loads, runs, and greedily generates the ungated `unsloth/gemma-4-E4B-it`
(`Gemma4ForConditionalGeneration`, 15.99 GB) on GB10, K=5 deterministic, and the
32-token greedy golden is captured
(`tests/parity/goldens/gemma4_e4b_text/`). The earlier "oracle-blocked at gate
time" concern (transformers lacking `gemma4`) is refuted — the module is present
and the model runs. **The IMAGE modality oracle is now captured too** (G2,
2026-07-28): the same E4B model on the pinned 0.25.0 oracle greedily describes a
fixed image (K=5 deterministic ⇒ STRICT gate form, 18 tokens → a coherent
gradient-image caption, 256 soft tokens), committed with four staged vision-tower
reference tensors for per-stage unit-gating
(`tests/parity/goldens/gemma4_e4b_image/`). Grounding the tower corrected an
earlier assumption: the Gemma-4 vision tower is a **custom NaFlex SigLIP2 with
multidimensional vision-RoPE, q/k/v RMSNorm, Gemma-2 sandwich norms, a learned 2-D
position embedding, and a √hidden average-pool-by-position pooler** — it does *not*
drop-in reuse the Qwen3-VL ViT (the block GEMMs/attention are reusable; the
patch-embed, RoPE, norms, pooler and the Gemma-4 NaFlex image processor are new).
The C++ NaFlex SigLIP2 vision tower is now IMPLEMENTED (additive
`gemma4_vision.{h,cpp}`) and PASSES its four per-stage gates vs the
transformers-eager references on the dgx CUDA build (patch-embed rel-L2 2.15e-3,
encoder 3.14e-2, pooled 1.36e-2, projected 1.85e-2; 220/220; compute-sanitizer 0) —
the tower-in-isolation milestone (mirrors Qwen3-VL M2a before its M2c e2e); grounding
it also revealed the vision `Gemma4ClippableLinear` layers carry FINITE trained QAT
activation clamps (not the no-ops the port-map assumed), now implemented. So Gemma-4
multimodal image is blocked only on the remaining engine wiring: the C++ Gemma-4
NaFlex image processor + the mm merge-plumbing (register SupportsMultiModal, the
masked-scatter merge of the 256 projected soft tokens at the image placeholder rows,
the tower→merge→decode fork) for the image→text e2e. **The USM-Conformer AUDIO
tower is now IMPLEMENTED too** (G3, 2026-07-28): the additive
`gemma4_audio.{h,cpp}` forward (`Gemma4AudioModel` + the audio projector) PASSES all
seven per-stage gates f32-exact vs the transformers-eager reference (host f32,
1256/1256: subsample rel-L2 5.4e-7, position-embeddings 8.9e-8, block0 4.2e-7,
block_mid 3.8e-7, block_last 4.4e-6, output_proj 5.9e-6, projected 6.3e-6), ported
1:1 from `modeling_gemma4.py` — a 2×Conv2d subsample, a relative positional
encoding, and 12 Conformer layers (half-step feed-forwards, a chunked-local
attention with Transformer-XL relative-position bias + a tanh soft-cap +
per-dim-scale softplus, and a GLU + depthwise-causal-conv light-conv module), then
`output_proj` and the audio embedder; the FINITE QAT clamps and the exact sliding
window (`dist ∈ [0,12)` — a RED-first-caught off-by-one) are grounded in the source.
So Gemma-4 now has all three modalities tower-proven (text STRICT 32/32, vision and
audio per-stage), and the remaining audio work is the same engine wiring as image —
the Gemma-4 audio feature extractor (mel frontend) + the mm merge-plumbing — plus
a device-resident bf16 forward for speed. The Gemma-4 MoE / k_eq_v / double-MLP
backbone stays the larger-variant follow-on. Audio as a standalone modality is also
proven end-to-end on the smallest oracle-runnable audio models (Whisper encoder,
then Voxtral-Mini-3B on the already-landed Mistral backbone).

### OLMo

The **OLMo-2 family** (`Olmo2ForCausalLM` / `Olmo3ForCausalLM`) is the **first
landed OLMo member** (correctness-complete, speed-pending): token-exact 16/16
greedy vs the vLLM 0.25.0 oracle on `allenai/OLMo-2-0425-1B` (STRICT 13/16 +
near-tie-band 3/16, max gap 0.094 nats, 0 forward-divergent). It is the cleanest
dense bring-up yet, needing no new compute kernel: its two distinctive traits
both reuse existing infrastructure. The reordered post-norm placement
(`norm_after`) is a subset of the GLM/Gemma sandwich norms (the same standalone
output-norm plus a plain residual add, without the pre-norms), and its QK-norm
is a full-width RMSNorm reusing the existing norm op at a new shape. Its GPT-NeoX
ByteLevel tokenizer (which prepends no BOS) makes it a real tokenizer-inclusive
gate. `Olmo3ForCausalLM` rides the same class (the 0.25.0 oracle constructs it);
the Olmo-3 interleaved sliding-window path has since landed and runs, but is
oracle-blocked for a gate (see the capability table above).

### Frontier and hardware-blocked families

**Qwen3.5 text-only arms (`Qwen3_5ForCausalLM`, `Qwen3_5MoeForCausalLM`) —
REGISTERED 2026-08-12, RUN GATE OWED (#490).** An ahead-of-pin forward port of
upstream PR vllm#50210 (`ad5d29db7`), which registers both arms against the same
`qwen3_5` module our gated `ForConditionalGeneration` wrappers already use. Two
additive `REGISTER_VLLM_MODEL` lines against the EXISTING dense and MoE
factories: no forward, no KV-cache spec, no loader fork.

The other half is ONE backbone weight-namespace decision per checkpoint:
`model.` for a text-only arm, `model.language_model.` for the wrappers, and a
MIXED index refused rather than half-bound. `Qwen/Qwen3.8-2.4T-A95B` declares
`Qwen3_5MoeForCausalLM` / `qwen3_5_moe_text` and is the token-exact
Qwen3.6-35B-A3B GDN-hybrid MoE backbone at larger scale, every knob
config-driven, with the BACKBONE weight names identical modulo that prefix.

**CORRECTED 2026-08-12: the prefix is NOT the only thing between this code and
`Qwen/Qwen3.8-2.4T-A95B`, and the first two commits of this row said it was.**
The MoE loader then read only PER-EXPERT NVFP4 routed experts, with no stacked
and no bf16 branch, while both published indices carry 3-D STACKED experts and
**zero** `weight_scale` / `input_scale` tensors — so a published MoE checkpoint
was REFUSED by a message naming the gap. The DENSE arm was never affected:
`LoadQwen3_5Dense` already routed BF16 / FP8 / NVFP4 per projection by presence.

**That loader gap is CLOSED, and closed with a token (2026-08-15,
[#740](https://github.com/mudler/vllm.cpp/issues/740) +
[#864](https://github.com/mudler/vllm.cpp/issues/864), both now `DONE`).**
`LoadQwen3_5Moe` reads 3-D stacked bf16 routed experts and resolves the GDN
tower, attention tower, shared expert and `lm_head` by tensor presence, so a
published bf16 Qwen MoE repo loads end to end and both published indices satisfy
the load plan completely.

**Binding gate — `Qwen/Qwen3.6-35B-A3B` bf16 @`995ad96e` (26 shards,
71,903,645,408 bytes, sha256 recomputed), greedy 7 prompts x 3 repeats x 16
tokens vs the pinned oracle `0.23.1rc1.dev1511+g555967922` on GB10, oracle
deterministic 7/7 across repeats: 6/7 prompts STRICT 16/16.** The seventh
diverges once on a bit-identical logprob (`top2_gap_mnats = 0.0`), where
`torch.argmax` takes the lower id and our on-device argmax the higher — an exact
tie, PASS under the ratified near-tie doctrine, filed as
[#910](https://github.com/mudler/vllm.cpp/issues/910).

Only the FIRST divergence in a prompt is validly adjudicable — past it the arms
carry different prefixes — so the raw 108/112 position count is not a quality
score and is not recorded as one. SACRED inertness 3/3 with real counts and
`GOLDENS_BYTE_IDENTICAL=1`: 27B **235/235** @`890bdef7`, 35B **315/315**, Coder
**138/138** @`b2cff646`, 688 assertions. **No throughput, latency or memory
number exists for this checkpoint, and none is claimed.**

**Both rows nevertheless stay `PARTIAL`, and the reason has changed: what was
owed was the loader, what is owed now is hardware.** That gate ran through
`Qwen3_5MoeForConditionalGeneration` on the 35B, not through
`Qwen3_5MoeForCausalLM`. The DENSE run gate closes when a `Qwen3_5ForCausalLM`
checkpoint that fits GB10 appears; the MoE one when a fitting
`Qwen3_5MoeForCausalLM` checkpoint does.

`Qwen/Qwen3.8-2.4T-A95B` is ~4.8 TB bf16 (~2.4 TB FP8) against 128 GB of unified
memory, so it remains unrunnable here. Its load plan resolves completely against
the published index — name, shape and dtype resolution, **not a token**. Still
owed: the MTP and GGUF arms for 3.8. This does not advance the parity pin.

**A smaller Qwen3.8 sibling DOES exist, and this page said otherwise until
2026-08-15.** `Qwen/Qwen3.8-27B` @`1d4bf0f2` (55.6 GB bf16, 18 shards) fits GB10
and declares `Qwen3_5ForConditionalGeneration` — the already-gated Qwen3.6-27B
shape retrained, `config.json` differing in exactly one key. It closes neither
text-only run gate above, because it is not a `Qwen3_5[Moe]ForCausalLM`.

**It is now token-gated ([#915](https://github.com/mudler/vllm.cpp/issues/915),
[spec](../.agents/specs/qwen38-27b-bf16-gate.md)).** Greedy, 7 prompts x 16
tokens vs the pinned oracle on GB10: **4/7 prompts STRICT 16/16**, and all three
first-divergence positions are **EXACT fp32 ties** — top-2 gap and
oracle-minus-ours both **0.000 mnats**, our token at rank 3 / 2 / 2 in the
oracle top-20, so `ALL_TIES_OR_IN_BAND` against `kNearTieMnats = 500`. All three
are the [#910](https://github.com/mudler/vllm.cpp/issues/910) tie-break and
nothing else: vLLM takes the lower token id, we take the higher.

Only the first divergence per prompt is adjudicable, so that is three numbers,
and a raw position count over the grid is not a quality score. The tie verdict
rests on the oracle's **fp32** logprobs, read twice — a greedy re-decode and a
teacher-forced probe that asserts the prefix it conditions on. A `transformers`
bf16 CPU probe agreed, but is recorded as secondary only: every runner-up gap it
printed was a multiple of 0.125, one bf16 ULP, so it could not have resolved a
real gap below that and could not have reported anything but a tie.

**Speed: one of three concurrency cells is established, and the reason the other
two are not is a defect of ours.** Against vLLM's production graphed config on
GB10, clocks pinned at 2184 MHz, c4 is the only cell where both arms completed
every request: **0.963x** output throughput and **1.008x** median ITL. At c1 and
c8 our server failed 1 of 6 in all three reps and 12/11/12 of 48 where vLLM failed none, so
those throughput cells are **withheld, not quoted**
([#931](https://github.com/mudler/vllm.cpp/issues/931)) — the metric divides
tokens by a duration that still contains the dead request, which is why c1 reads
0.677x while median TPOT in the same file reads 1.014x in our favour.

**That cause has landed, and the two cells are now waiting on a re-run rather
than on a diagnosis.** The dropped requests were our own SSE keepalive comment
frame, and `VT_SERVER_SSE_PING_S` now defaults to `0`. Both cells stay withheld
until [#915](https://github.com/mudler/vllm.cpp/issues/915) re-runs them paired.

Resource axes on the same series: cold start to first `/health` **53 s vs
780 s = 14.7x**, and host memory after warmup **42.5 vs 110.1 GiB = 2.59x** —
the latter with the caveat that vLLM's figure is set by
`--gpu-memory-utilization 0.85` pre-reserving KV on a unified-memory box.

**Its quantized arms are not gated.** Three rows enter `READY` on
[#821](https://github.com/mudler/vllm.cpp/issues/821)
([spec](../.agents/specs/qwen38-27b-quant-arms.md)): `LOAD-GGUF-MMPROJ`,
`QUANT-QWEN38-27B-GGUF-ARM` and `QUANT-QWEN38-27B-NVFP4-ARM`. `AGENTS.md` makes
the quantized arms a standing requirement, and these are the arms a user can run:
17.1 GB of Q4_K_M against 53.8 GB of bf16 GGUF.

Nothing is implemented. No production path accepts a second `clip` GGUF projector
beside the language file. The artifact published as `unsloth/Qwen3.8-27B-NVFP4`
is a compressed-tensors `mixed-precision` checkpoint with **zero `*.input_scale`
tensors**, which is why the reported load dies. And the Q4_K_M file ships the MTP
drafter as block 64, so a loader reading `block_count` as decoder depth builds a
65-layer model out of a 64-layer checkpoint.

**Both token gates are `PENDING` on named external authorities, and this page
claims no number for either.** The Q4_K_M arm's only comparator is llama.cpp,
whose pin is recorded `gateable = no` with
[#857](https://github.com/mudler/vllm.cpp/issues/857) owing the measurement. The
NVFP4 arm's oracle is the pinned vLLM, which builds and imports inside an `rc`
lease but has never been shown to RUN a model there
([#1185](https://github.com/mudler/vllm.cpp/issues/1185)).

Neither blocker is these rows' to clear, and neither stops the CPU-side work: ten
of the sixteen declared tests need no GPU and no lease.

Larger DeepSeek / GLM / MiniMax / Gemma-4 variants are recorded as
**hardware-blocked** (they do not fit 119 GiB of unified memory on this box) or
**spiked-only**, per the [model matrix](../.agents/model-matrix.md).
**DeepSeek-V4-Flash** is scoped with W1/W2 CPU scaffolding landed (not yet
runnable) in [deepseek-v4-flash.md](../.agents/specs/deepseek-v4-flash.md): a
~167B/256-expert NEW architecture (DeepSeek Sparse Attention MLA + Manifold
Hyper-Connections + NVFP4/MegaMoE + sqrtsoftplus/hash MoE). As of 2026-07-28 the
additive registry stub + config parse + checkpoint loader name-map are landed and
VERIFIED against the real `nvidia/DeepSeek-V4-Flash-NVFP4` safetensors header
(HTTP-range, no download). The forward host-composition (W7) + the W7-device CUDA
kernels are now landed: the four NEW V4 op families (MHC / DSA indexer+seams /
compressor+fp8_ds_mla / sqrtsoftplus-hash MoE) are ported to CUDA, registered
through the OpProvider seam, dispatched by a real `DeepseekV4Model::ForwardDevice`,
and RUNTIME-VERIFIED on the DGX GB10 at small shape vs the host-ref oracle
(`test_cuda_deepseek_v4` 11/11·153, compute-sanitizer memcheck 0 errors, RED-first
proven; 2026-07-29, `CLAIM-DEEPSEEK-V4-W7-DEVICE`). The 512-wide MLA attn + expert
grouped-GEMM REUSE the existing NVFP4/FP8 kernels. The **W2b GGUF keep-quant tower
materialization is now landed** (2026-07-29, `CLAIM-DEEPSEEK-V4-W2B`):
`LoadDeepseekV4FromGguf` (+ `DeepseekV4ParamsFromGguf`) wires the landed `blk.N.*`
name-map + keep-quant blocks into the `DeepseekV4` weight towers (the MLA linears,
router gate, 256 routed + shared experts, and lm_head KEEP their ~2-3-bit blocks
COMPRESSED via `OwnGgufQuantBlocks`, the ~91 GiB-vs-~316 GiB OOM enabler; norms /
MHC / DSA / embed / `tid2eid` dequant), accounts for every tensor, lifts the registry
GGUF reject, and dequants the tiny CPU composition tower so a loaded model forwards
(`test_deepseek_v4_gguf_load` 5/5·149 — accounting 126/126, keep-quant residency,
load→forward, RED-first) — DERIVED + BUILD-VERIFIED at tiny synthetic shape. The
**keep-quant expert/MLA GEMMs now run ON THE GPU** (2026-07-29,
`CLAIM-CUDA-KEEPQUANT-GEMM`): the FIRST CUDA keep-quant GGUF k-quant GEMM
`KERNEL-QUANT-CIQ-GEMM-CUDA` (`cuda_quant_dot.cu`, MMVQ-style dequant-in-kernel over
the compressed Q8_K-family blocks) registers a kCUDA `kMatmulBTQuant` provider,
which flips `GgufQuantComputeAvailable` TRUE on `kCUDA` — so on a CUDA runner
`vt::MatmulBT` dispatches these GEMMs to the GPU instead of the 20 ARM cores
(the biggest DeepSeek-V4 speed lever), weights staying COMPRESSED in the unified
pool. GB10-gated `test_cuda_quant_dot` 2/2·92401 vs the CPU oracle + f64 dequant,
memcheck 0, RED-first proven. The experts-on-GPU tok/s is the follow-on benchmark.
Still NOT runnable end-to-end on the REAL checkpoint: the 91 GB `UD-IQ2_XXS` single-Spark
run (keep-quant load + `ForwardDevice` greedy gen + self-consistency/coherence gate +
benchmark) is the W8-final operational residual (download + DGX), and the NVFP4/fp8
paged-engine strict gate (W8) is multi-Spark-blocked (156.7 GiB does not fit ONE GB10).
**HW-fit correction:** that NVFP4 checkpoint is **156.7 GiB**, NOT the ~83 GiB the
scoping spike estimated (only the 256 routed experts are W4; the MLA and shared
linears are FP8 plus NVFP4 double-scale overhead), so it does **not** fit ONE
GB10's 119 GiB unified pool. **Single-Spark IS viable via a ~2-bit GGUF (user
correction 2026-07-28):** `unsloth/DeepSeek-V4-Flash-GGUF` `UD-IQ2_XXS` = 90.9 GB
(3 shards, ungated) FITS one GB10 with ~28 GiB headroom (also UD-IQ1_S/IQ1_M/IQ2_M/
Q2_K_XL). So two vehicles: single-Spark ~2-bit GGUF, or 2x-Spark NVFP4/fp8 over the
interconnect. The GGUF vehicle's correctness reference is llama.cpp-on-box (the
pinned vLLM cannot load V4-from-GGUF) and needs a V4-GGUF loader + IQ i-quant dequant
(IQ1_S/IQ2_XXS — not in our C4 K-quant set). The NVFP4/fp8 vehicle stays multi-node.
A source-level spike (2026-07-28) confirmed a **same-quant IQ2_XXS GGUF benchmark of
our engine vs vLLM is NOT viable today**, blocked on both sides: vLLM 0.26 moved GGUF
out-of-tree to the uninstalled `vllm-gguf-plugin` (which *does* dequant IQ2_XXS) but
`DeepseekV4ForCausalLM` has no `packed_modules_mapping`/GGUF wiring; and our engine
hard-rejects GGUF for DeepSeek-V4/V2 and lacks IQ2_XXS/Q2_K dequant — so even the
`UD-Q2_K_XL` k-quant fallback does not rescue it. Apples-to-apples for DeepSeek-V4 is
the NVFP4 vehicle; a true same-GGUF cross-engine number is only available on a
Qwen3/dense k-quant both engines already load. **W8 KEEP-QUANT MEMORY ENABLER + NAME-MAP
landed (2026-07-29, `CLAIM-DEEPSEEK-V4-W8`):** the ~2-3-bit routed-expert encodings now
carry a keep-quant `vec_dot` — IQ2_XXS, **IQ3_XXS** and Q2_K (1:1 ports of ggml
`vec_dot_{iq2_xxs,iq3_xxs,q2_K}_q8_K_generic`), so `HasQuantDotKernel` is TRUE and the
loader keeps their blocks COMPRESSED (the memory enabler: ~91 GiB vs the ~316 GiB bf16
OOM). **HONEST CORRECTION:** the real `UD-IQ2_XXS` experts are IQ2_XXS (gate/up) +
**IQ3_XXS** (down) — NOT Q2_K (that is the `UD-Q2_K_XL` sibling); all three landed.
**HONEST finding:** there is NO CUDA keep-quant vec_dot for ANY k-quant (`kMatmulBTQuant`
is CPU-only) — on GB10 it runs on the 20 ARM cores against the unified pool. Gate
`test_ops_quant_dot` 19 cases / 130444 assertions, RED-first proven. The full 1328-tensor
`blk.N.*`→V4 name map now has **EXACT coverage (0 unmapped, 0 leftover)** against the real
manifest (shard GGUF headers read via HTTP-range — no 91 GB download; `scripts/check-dsv4-gguf-namemap.py`
rc=0). The V4 registry GGUF reject is now a PRECISE message (keep-quant + name-map landed;
W2b tower materialization pending). **The RUN did NOT execute** and was NOT faked: it is
blocked on the unimplemented **W2b** (materialize the keep-quant blocks into the DeepseekV4
towers via the name map) — a genuine code brick, then the 91 GB download + GB10 greedy gen.
**W8-final entrypoint wiring landed + gated (2026-07-29, `CLAIM-DEEPSEEK-V4-W8`, base
`376e186b`):** with W2b landed, the top-level GGUF dispatch now recognizes a `deepseek4` file —
`DeepseekV4HfConfigFromGguf` maps `general.architecture=deepseek4`→the registered
`DeepseekV4ForCausalLM` (republishing geometry into `config.raw` for the parse hook) and
`LoadedEngine::FromModelDir` routes it via `HfConfigFromGgufDispatch` (was qwen-only). Gate
`test_deepseek_v4_gguf_load` 6/6·168 (CPU `-Werror`-clean), qwen path byte-neutral. **But the real
single-Spark RUN is still BLOCKED — now on a CODE residual, not download/box, and was NOT attempted
(it would OOM-reboot the box):** the DeepSeek-V4 forward composes off the FULLY-DEQUANTIZED f32
`weights.host` tower, and `LoadDeepseekV4FromGguf` builds that host tower unconditionally (every
routed expert `HostVec`→f32) ≈ ~24 GiB/layer × 43 ≈ **~1.0 TiB** f32, past the 119 GiB pool by
~layer 5; the keep-quant `weights.gguf` tower (~91 GiB) is built but never read by the forward. A
real run needs the forward rewired onto the CIQ `kMatmulBTQuant` keep-quant blocks + the host dequant
gated off (was named residual W2c). No tokens generated (not faked); DGX left as found.
**W2c landed (2026-07-29, `CLAIM-DEEPSEEK-V4-W2C`): the OOM-infeasibility is FIXED.**
`LoadDeepseekV4FromGguf` no longer f32-expands the big MLA/MoE/lm_head weights (only the
small norms/embed/MHC/DSA/hash tensors dequant); a new `DeepseekV4ForwardGguf` runs the
SAME composition with the 512-wide MLA linears + the 256 routed/shared expert GEMMs +
lm_head consuming the COMPRESSED `weights.gguf` blocks in place via `vt::MatmulBT`→the CPU
`kMatmulBTQuant` CIQ GEMM (no per-layer f32 expansion), and `DeepseekV4Model::Forward`
gates on `has_gguf_weights` (the safetensors/NVFP4 + tiny-synthetic host path stays
byte-identical). Gate `test_deepseek_v4_gguf_load` 7/7·185 (CPU Release `-Werror`-clean):
the keep-quant forward RUNS finite+deterministic, keep-quant(Q8_0)==dequant(bf16) RelL2
0.0116 (< 0.05 near-tie), RED-first (a no-sink miswire diverges, and a load that rebuilds
the f32 tower fails a load-time `VT_CHECK` + the host<gguf-bytes assertion). Memory-bound
asserted: at tiny shape host 23,980 B vs keep-quant 141,676 B; projected to the real config
the 256 routed experts alone are ~1032 GiB f32 (OOM-reboots the 119 GiB pool) vs the
keep-quant `UD-IQ2_XXS` ~91 GiB + small host < 3 GiB = memory-FEASIBLE on ONE GB10. The
real 91 GB run (download + GB10 generate + benchmark) stays the operational W8-run, now
memory-feasible. SACRED-inert: only the V4 forward/loader + its test changed; the W3-W6
primitive tests (the correctness oracle) are untouched and still pass.
**W8-run ATTEMPTED 2026-07-29 — HARDWARE-BLOCKED (not run, nothing faked):** with the code
complete + memory-feasible @ `2936ff70`, the operational run could not start because the
**DGX GB10 (`dgx.casa`) is OFFLINE** (ping 5/5 loss, ARP FAILED, SSH no-route; gateway/Thor/
mac-mini up ⇒ DGX-specific, down or crashed). No out-of-band wake path from the dev box.
The DeepSeek-V4 row stays SPIKE (no run, no benchmark). Resume: bring the DGX back online and
re-run the W8-run recipe (free box → download ~91 GB `UD-IQ2_XXS` → keep-quant load [assert
resident ≈ 91-94 GiB, the W2c `VT_CHECK` must not fire] → greedy generate → self-consistency +
coherence gate → TPOT/throughput/peak-resident benchmark). No code change needed.
**W8-run apples-to-apples oracle SELECTED + ds4-file LOADABILITY PROVEN (2026-07-29, base
`edf68c91`, NOT pushed):** the cross-engine oracle is now **antirez/ds4** (DwarfStar,
`make cuda-spark` on GB10), and the shared apples-to-apples vehicle is ds4's `q2-imatrix`
GGUF (`antirez/deepseek-v4-gguf`, a single 80.7 GB file — routed experts IQ2_XXS gate/up +
Q2_K down, Q8_0 attn/shared/out, F16 embed — all keep-quant types we already support).
Verified via HF HTTP-range (NO 80 GB download) that our `blk.N.*` name-map covers ds4's file
EXACTLY (1328/1328, 0 unmapped, 0 leftover — byte-identical tensor-name set to unsloth's).
Found + FIXED three real-file loader gaps that would throw or zero the model before a token:
(1) `compress_ratios` length 44 (43 layers + a trailing MTP entry) vs our strict
`==block_count` — relaxed to `>=` + prefix-truncate; (2) the clamp limit was read only from
the scalar `swiglu_clamp`, but ds4 ships the per-layer array `swiglu_clamp_exp` (all 10.0) and
a 0 limit ZEROES every expert (ClampedSwiGLU min(gate,0)·clamp(up,0,0)) — array fallback added;
(3) the hash `ffn_gate_tid2eid` is ggml I32 (type 26), unhandled in `DequantGgufRowToF32` — I32
case added. Gate `test_deepseek_v4_gguf_load` **8/8·288** (CPU Release `-Werror`-clean) with a
new ds4-flavor case reproducing all three quirks (load + keep-quant forward finite +
deterministic); RED-first proven (disabling the I32 case throws "unsupported ggml type 26").
The GB10 run itself did NOT execute (nothing faked): it is blocked on the GPU flock held by a
concurrent agent, two long builds (ds4 `cuda-spark` + our aarch64 CUDA), the 80 GB download,
and a greedy driver (the engine's incremental paged decode is incompatible with V4's stateless
keep-quant recompute — a manual greedy loop over `DeepseekV4ForwardGguf` is required; note
`Tokenizer::FromGguf` also rejects ds4's `pre='joyai-llm'`, so the token cross-check must inject
ds4's token ids). Row stays SPIKE.
**W8-run EXECUTED on the DGX GB10 (2026-07-29, base `858b0b15`, NOT pushed) — our engine LOADS
the real 158 B model but the forward HARD-FAILS at the real geometry; ds4 reference captured:**
the 80.7 GB file was downloaded + integrity-verified (86,720,111,488 B == HTTP Content-Length),
ds4 was built (`make cuda-spark`), our CPU engine + a new greedy driver (`examples/deepseek_v4_gen`)
were built, and both were run on the SAME file. **ds4 (GB10 GPU)** loads 80.76 GiB in ~20–29 s and
greedily generates coherent on-topic text ("We need to answer: "The capital of France is". This is
a straightforward") at **prefill 358 tok/s, decode 16.5 tok/s** (ctx=1024, `ds4-bench`). **Our engine**
LOADS the keep-quant tower on the real model (43 layers, 256 experts, vocab 129280, all 1328 tensors
accounted, `has_gguf_weights=1`, ~78 s) at **PEAK RESIDENT 116.2 GiB** — it FITS under the 119 GiB
pool but is ~32 GiB ABOVE the projected ~84 GiB, because `OwnGgufQuantBlocks` is called WITHOUT the
`mmap_src` arg so it COPIES ~81 GiB of blocks into owned vectors while the mmap pages stay resident
(fix: mmap-VIEW the blocks like `qwen3_5_gguf_weights.cpp`). **The forward then HARD-FAILS** with
`deepseek-v4 keep-quant GEMM: weight shape mismatch: want [N=512,K=4096] got [1024,4096]`
(`deepseek_v4.cpp:234`) at **layer 2 (the first DSA compressor layer)**: the W3–W7 forward, gated only
at tiny synthetic shape, conflates `head_dim`=512 with the DSA compressor's distinct 1024-dim output
(the real `attn_compressor_gate`/`attn_compressor_kv` are `[1024,4096]`). Layers 0–1 pass; layer 2
crashes. NO coherent tokens generated → row STAYS SPIKE. **Remaining brick:** rework the MLA/DSA
forward to the real DeepSeek-V4 geometry (compressor 1024-dim; audit indexer + grouped output-LoRA at
`o_groups=8`/`nh=64`/`q_lora_rank=1024`) against a reference; plus the mmap-view load fix. The
download + keep-quant load + tokenizer + greedy driver + ds4 oracle are DONE; the forward geometry is
the one blocker before a real single-Spark generation.
**W8-run.9 FIXED — COHERENT single-Spark generation; DeepSeek-V4 ADVANCES `SPIKE → ACTIVE` (2026-07-29,
base `eeef1695`, NOT pushed).** The isolated L00 experiment (aligned dump: ours' post-input-norm attn
input vs ds4's `attn_norm`) forked H1/H2: attention INPUT bit-exact (rel-L2 **0.0000**), KV correct
(0.018), but the **q operand rel-L2 0.9646** → the MLA q projection. Root cause: our forward OMITTED the
**per-head query RMS-norm** (ds4 `head_rms_norm_inplace` after wq_b; the KV correctly gets only its
`attn_kv_a_norm`). Fix `DeepseekV4QHeadRmsNormInplace` (+ spec-anchored RED-first doctest) → L00 q
**0.9646→0.0013**, attn_out **0.5956→0.0175**, and the FULL 43-layer curve COLLAPSED to the keep-quant
floor (L33 **0.53→0.003**, L34 **6.02→0.003**, MAX 0.0334). Plus the second confirmed fix, the
**inverse-RoPE on the attention output** (ds4 `rope(heads,inverse=true)`, identity at pos 0, load-bearing
for pos>0). Greedy generation is now **"The capital of France is Paris.<｜end▁of▁sentence｜>"** (correct +
EOS), self-consistent; `test_deepseek_v4_gguf_load` 10/10·403. CPU-tier decode ~3.3 s/tok / peak 85.8 GiB
(ds4 GPU: 16.5 tok/s). Gated by coherence + self-consistency + the ds4-oracle per-layer diff (no vLLM V4
GGUF plugin ⇒ not vLLM-token-exact). Named residuals: GPU-expert dispatch (CPU-tier), DSA-sparse ctx>512
(dense-fallback, exact for short gen), paged-engine integration. Row `ACTIVE`.

**W8-run.10 — GPU-expert wiring (`--gpu`) + apples-to-apples vs ds4 (2026-07-29, base `20d8ccfa`).** Experts
now run on the GB10: a CUDA queue routes every keep-quant expert/MLA/lm_head GEMM to the `kMatmulBTQuant`
kCUDA provider (#195), reading unified-memory weight blocks in place (memory-safe; no device copy). GPU
genuinely used (nvidia-smi 38–50% util, compute-app `deepseek-v4-gen` during the run); GPU path
byte-identical to CPU; CPU default unchanged (`test_deepseek_v4_gguf_load` 10/10·403). Real table (same 80.7
GB ds4 file): ds4 GPU prefill 325.9 / decode 16.3–16.6; ours `--gpu` prefill 6.26 / decode 0.68 (1.3–1.4×
over our CPU path). The residual gap is NOT expert placement (fixed) but our **stateless full-recompute
driver (no KV cache)** + route-(a) host-orchestration/per-GEMM sync. Real speed path = `ForwardDevice`
(#183): resident on-device activations + KV cache + no per-GEMM sync — the only route to approach ds4's 16.5
tok/s. Route (a) not dressed as a win. Row stays `ACTIVE`; see docs/BENCHMARKS.md W8-run.10.
**ForwardDevice campaign Stage 1 — MLA latent KV cache + incremental decode (2026-07-29, base `fd9e191c`,
NOT pushed).** The single biggest decode lever, equivalence-gated + benchmarked. For the real dense-MLA run
the only cross-token state is the per-layer `deck` latent `[head_dim]` (num_key_value_heads=1; MHC manifold
per-token), so `DeepseekV4KvCache` (mirror of ds4 `raw_kv`) makes each decode step process ONE new token
against cached KV instead of re-running the whole forward over the growing context. `DeepseekV4ForwardGgufCached`
+ driver `--kv-cache` (rollback-able; default + `--gpu` unchanged). **Token-IDENTICAL** to full-recompute
("…Paris.", ids `11111 16 455 6102 294 8760 344 11111`); decode **0.68 → 4.79–5.35 tok/s (7.9×)**, ~1/3 of
ds4's 16.5; GPU used (nvidia-smi 29–46%, compute-app `deepseek-v4-gen`); peak 86.33 GiB. Gate
`test_deepseek_v4_gguf_load` **11/11·430** (prefill bit-identical, incremental token-identical, RED-first).
Stage 0 spec: `.agents/specs/deepseek-v4-forward-device.md`. Residual to ds4 = host-orchestration + per-GEMM
sync (Stages 2–3: device-resident activations + graphs). Row stays `ACTIVE`; see docs/BENCHMARKS.md.
**ForwardDevice campaign Stage 2 — decode profiler + deferred-sync drain batching: HONEST NEGATIVE RESULT
(2026-07-29, base `17ec33ea`, NOT pushed).** The scoped lever ("drop the per-GEMM host sync") is MEASURED
to NOT move decode. Added `VT_V4_PROF`, which split the 0.18s decode step into sync 0.073s / gemm-dispatch
0.074s / host-glue 0.034s. Batched the stream drains (21 routed+shared expert GEMMs/layer → 2; grouped
o-LoRA → 2; `defer_sync` + `DrainDevice`), byte-identical (`test_deepseek_v4_gguf_load` 11/11·430). Real
DGX: decode 0.20 s/tok (5.11 tok/s), token-identical ("…Paris."), util ~49%, **NO speedup** (sync
0.073→0.071s). CORRECTION: the "sync" bucket is the host *waiting for GPU compute* of ~1100 tiny T=1 GEMMs,
NOT sync-call overhead; both buckets scale with GEMM count, not sync-call count. Real lever (named,
corroborated by the Qwen3-Coder W7 decode-graph row): a **grouped MoE GEMM** (18 tiny expert matvecs → ~3
kernels) + a **decode CUDA graph** to kill launch tax. Recommend re-scoping Stage 2 to the grouped GEMM.
Rollback-able; NO speedup claimed. Row stays `ACTIVE`; see docs/BENCHMARKS.md.
**ForwardDevice campaign Stage 2 (re-scoped) — GROUPED keep-quant MoE GEMM: REAL SPEEDUP (2026-07-29, base
`05f6318c`, NOT pushed).** The lever the profiler proved. New vt op `kMatmulBTQuantGrouped` (mirrors
`kMoeGroupedGemmBf16`, extends the #195 `kMatmulBTQuant` kernel): collapses the 6 routed experts ×
{gate,up,down} = 18 tiny T=1 matvecs/layer into 3 grouped kernels (CUDA `QuantDotGemmGroupedKernel` =
same dot core, per-group weight-row index; CPU provider loops `MatmulBTQuant` per group = byte-identical
reference). `MoeBlock` routes the routed experts through it (shared stays per-expert); `VT_V4_GROUPED_MOE=0`
rolls back. **TOKEN-IDENTICAL** (grouped ON == OFF == "…Paris.", `11111 16 455 6102 294 8760 344 11111`);
gate `test_deepseek_v4_gguf_load` **12/12·531** (+grouped==per-expert byte-identical, RED-first permuted map).
DGX `--gpu --kv-cache`: **decode 5.83 tok/s (23-tok) / 6.60 (11-tok) — ~22% over Stage 1's 4.79** (equal
length); prefill 7.6–8.1; peak 86.33 GiB; ~35% of ds4's 16.5. nvidia-smi util did NOT rise (36%) — fewer/
larger kernels leave the GB10 idle on host launches, so decode is now clearly **host-launch-bound** →
motivates Stage 3 (decode CUDA graph, mirroring Qwen3-Coder W7). Row stays `ACTIVE`; see docs/BENCHMARKS.md.
**ForwardDevice campaign Stage 3 — DECODE CUDA GRAPH: BLOCKED (architectural finding; honest last residual)
(2026-07-29, base `3eb018df`, NOT pushed).** A decode CUDA graph CANNOT be built on the host-orchestrated
forward — no code change. The vt capture contract (`cuda_backend.cu:173-182`) demands pure async stream
work (no Synchronize / host<->device copies / malloc; fixed pointers). The dense/DFlash graphs capture a
device-resident `ForwardLayers` over persistent `DBuf`s (`qwen3_5.cpp:5820/5747`). DeepSeek-V4's GGUF
forward (`ForwardComposeImpl`, `device=false`) instead runs MHC/Sinkhorn, the attention QK/softmax/AV Dot
loop, rope, RMSNorms, the MoE router (which decides `expert_ids`), SwiGLU and combine ON THE HOST over
`std::vector`, between GPU GEMMs, reading each output on the host — no contiguous async device sequence to
capture; the host syncs abort capture; the #183 glue kernels Upload/Download+sync per call and there is no
device attention kernel. No host-node fallback exists. **Prerequisite:** a device-resident decode forward
(all glue as real parallel device kernels on persistent `DBuf`s, incl. a device MLA attention kernel),
mirroring `Qwen3_5DenseDecodeGraph` — a multi-brick port, NOT the #183 `<<<1,1>>>` correctness kernels.
**FINAL campaign result: decode 0.68 → 5.83 tok/s (~8.6×), token-identical, ~35% of ds4's 16.5** (Stage 1
KV cache 7.9× + Stage 2 grouped MoE GEMM +22%; both merged). Named residual = host-orchestration launch
overhead (util ~36%), removable only by the device-resident forward. Row `ACTIVE`; see docs/BENCHMARKS.md.
**Device-resident decode campaign — Brick 0 (scope) + Brick A (device MLA attention kernel) (2026-07-29,
base `59260579`, NOT pushed).** User commissioned the device-resident decode forward (unblocks the Stage-3
graph). Brick 0 = `.agents/specs/deepseek-v4-device-decode.md` (bricks A→B→C→D, DBuf layout, gates, honest
ceiling: a graph likely reaches ~10-13 tok/s but SHORT of ds4's 16.5 — fp8 KV + tuned MMQ are a named
follow-on). Brick A = first real device V4 forward kernel: `DecodeAttnKernel` (`cuda_deepseek_v4.cu`)
replaces the host QK/softmax-sink/AV Dot loop, reading the unified KV latent in place (num KV heads = 1),
preserving host `SoftmaxWithSink` order (only expf vs std::exp differs). Flag `VT_V4_DEVICE_ATTN` (default
OFF). Gate = correctness: CUDA unit `test_cuda_deepseek_v4` **12/12·412** (+device-vs-host RelL2<1e-5,
RED-first no_sink); `test_deepseek_v4_gguf_load` 12/12·531; real-model TOKEN-IDENTICAL (ON==OFF=="…Paris.").
Speed (not Brick A's gate): decode 6.23 tok/s (24-tok) vs 5.83 baseline (~7%, grows with ctx); util still
~35% (payoff at Brick D's graph); peak unchanged. Rollback-able (flag OFF default). Bricks B→C→D await
review. Row `ACTIVE`; see docs/BENCHMARKS.md.
**Device-resident decode campaign — Brick B increment 1: device clamped-SwiGLU (in place) (2026-07-29, base
`21191ce2`, NOT pushed).** Brick B reimplements the decode glue as real in-place device kernels (no
Upload/Download/Sync — the Brick-A pattern), landed in gated increments. Increment 1 = MoE clamped-SwiGLU:
`ClampedSwiGLUInPlaceLaunch` (reuses the tested #183 `ClampedSwiGLUKernel` on unified `gate_up`/`out`);
`DispClampedSwiGLU` routes to it under `VT_V4_DEVICE_GLUE=1` + CUDA (default OFF). CHARACTERIZED NEAR-TIE
(RelL2<1e-5, device `expf` vs host `std::exp` in the SiLU — stated, not bit-identical). Gate = correctness:
CUDA unit `test_cuda_deepseek_v4` **13/13·671** (+in-place==host RelL2<1e-5, RED-first clamp-limit);
`test_deepseek_v4_gguf_load` 12/12·531; real-model **TOKEN-IDENTICAL despite the near-tie** (`VT_V4_DEVICE_GLUE=1`,
and combined with `VT_V4_DEVICE_ATTN=1`, both = "…Paris."). Speed flat (6.12 tok/s attn+glue 24-tok vs 5.83;
util ~38% — payoff at Brick D). Remaining Brick-B glue (router, MHC pre/post/head/Sinkhorn, RMSNorm, RoPE,
combine) = next increments. Row `ACTIVE`; see docs/BENCHMARKS.md.
**Device-resident decode campaign — Brick B increment 2: MHC-post/head + router device glue; MhcPre FINDING
(2026-07-29, base `21191ce2`, commit `3046f087`, NOT pushed).** In-place launchers reusing the tested #183
kernels for MHC pre/post/head + router; `DispMhcPost`/`DispHcHead`/`DispRoute` route to them under
`VT_V4_DEVICE_GLUE`. **KEY FINDING (measured):** the #183 `MhcPreKernel`/`HcHeadKernel` are `<<<1,1>>>`
single-thread stubs — routing MhcPre (86 calls/step over 16K, one GPU thread) REGRESSED decode ~10× (0.59
vs 6.5 tok/s), so MhcPre stays HOST until a real PARALLEL MhcPre kernel exists (folded RMSNorm + 20-iter
Sinkhorn + gates + collapse — the hardest remaining piece). The Grid-launched glue (SwiGLU, router,
MHC-post, hc_head) IS on device + FLAT. Gates: CUDA `test_cuda_deepseek_v4` **14/14·749** (in-place ==
round-trip #183 bit-identical, RED-first); `test_deepseek_v4_gguf_load` 12/12·531; real model TOKEN-IDENTICAL
all-device-on "…Paris.", decode 6.38 tok/s (flat). **BRICK B NOT COMPLETE:** parallel glue done + gated;
remaining = parallel MhcPre kernel + RMSNorm + RoPE + MoE combine. Row `ACTIVE`; see docs/BENCHMARKS.md.
**Device-resident decode campaign — Brick B increment 3: PARALLEL MhcPre kernel + per-op-sync-tax finding
(2026-07-29, base `21191ce2`, commit `9bd51523`, NOT pushed).** The crux: `MhcPreParallelKernel` (one block,
256 threads over the hc·H=16K width; block-tree reductions in double; Sinkhorn+gates on thread 0 in host
order) replaces the `<<<1,1>>>` stub. CHARACTERIZED NEAR-TIE (width reductions reorder → RelL2<1e-3).
`DispMhcPre` re-enabled. Gates: CUDA `test_cuda_deepseek_v4` **14/14·736** (MhcPre parallel==round-trip
RelL2<1e-3, RED-first); `test_deepseek_v4_gguf_load` 12/12·531; real model **all device on (incl parallel
MhcPre): TOKEN-IDENTICAL "…Paris."**, decode 5.43 tok/s (24-tok). **FINDING — per-op SYNC tax, not poison:**
parallel MhcPre is fine (10× poison gone); the ~21% dip vs host-glue (5.43 vs 6.84) is the ~560 per-op
device-glue drains/step — inherent to Brick B's per-op-synced structure, removed by Brick C (drop syncs) +
Brick D (graph). No single-thread on the hot path. Remaining for full device-residency: device RMSNorm,
RoPE, MoE combine (can fold into Brick C). Row `ACTIVE`; see docs/BENCHMARKS.md.
**Device-resident decode campaign — Brick C part 1: folded-in device glue kernels (RMSNorm/RoPE/combine)
(2026-07-30, base `f2fff1ee`, commit `d68768b4`, NOT pushed).** The device kernels the resident assembly
needs: `RmsNormKernel` (parallel block reduction; has_w=false → per-head q-RMS), `RopeKernel` (per-row
sequential YaRN recurrence, inverse flips sin), `MoeCombineKernel` (Σ_a w_a·eo_a). Added to the seam
(`rms_norm`/`rope` in DsaDeviceKernels, `moe_combine` in MoeDeviceKernels); NOT wired into the forward yet
(they land in the resident assembly). ALL THREE CHARACTERIZED NEAR-TIES (stated): RMSNorm (reduction
reorder), RoPE (cos/sin lib), and combine (a new **FMA** finding — device fused multiply-add vs host
separate mul+add; my bit-identical assertion correctly failed at last-ULP → fixed to a near-tie). GATE:
CUDA `test_cuda_deepseek_v4` **15/15·1106** (RMSNorm/RoPE/combine == host RelL2<1e-5, RED-first each);
`test_deepseek_v4_gguf_load` 12/12·531. **Brick C part 2 (the SPEED part) REMAINS:** thread the T=1 decode
through persistent DBufs across the 43-layer stack, consume these + the Brick-A/B kernels resident, and drop
the ~560 per-op drains (sync only at the step-boundary logits read) — that recovers + exceeds 5.83. Row
`ACTIVE`; see docs/BENCHMARKS.md.
**Device-resident decode campaign — Brick C part 2: the resident T=1 decode assembly — CORRECT + token-
identical, but eager-SLOWER (honest launch-bound finding) (2026-07-30, base `24bddf15`, commit `3a635624`,
branch `deepseek-v4-resident-assembly`, NOT pushed).** `ForwardResidentDecodeGguf` runs the whole 43-layer
T=1 step as ONE async device chain over the unified buffers — every GEMM defers (GemmIntoKq/RowSliceInto/
GroupedInto), the small host primitives run on the Brick-C device kernels IN PLACE (q/kv/final RMSNorm + the
batched per-head q-RMS `rms_norm_rows`, dual+inverse RoPE, MHC pre/post/head, router, clamped-SwiGLU, combine,
decode_attn — none draining), routing stays RESIDENT (device router → i32 topk_ids the grouped GEMM consumes
+ topk_weights the combine consumes — no host gather), the KV append writes the new deck row into the cache
slot, async cudaMemcpyAsync for the grouped-GEMM broadcast + [gate|up] pairing. Flag `VT_V4_RESIDENT_DECODE=1`
(default OFF). New tiny glue: `rms_norm_rows` (batched nh=64 q-RMS in ONE launch). Fixed a guard bug (rejected
on config compress_ratios; the keep-quant run is dense `dsa_dense` regardless → resident never engaged on the
first run). GATES (DGX GB10, deepseek-v4-gen --gpu --kv-cache, real 80.7 GB, 24 decode): CUDA unit
**16/16·33877** (+rms_norm_rows case); `test_deepseek_v4_gguf_load` **12/12·531**; **resident ON == host OFF
TOKEN-IDENTICAL** ("…Paris.", exact 25 ids). **SPEED (honest): resident-eager 5.17 tok/s vs host 6.44 (−20%);
util 38%→55%.** Attribution: NOT a correctness bug (tokens identical), NOT the router-gate drains (bf16 gate →
43 CPU-GEMM drains ≈ 1.4%/step) — **launch/host-gap bound** (~1700 small device-kernel launches/step replace
fast host-ARM glue → ~45% GPU idle). The §4 "may improve" precondition, NOT the payoff — **Brick D (collapse
the ~1700 launches into ONE cudaGraphLaunch)** is the payoff (55%→~100% busy ≈ 1.8× → ~9 tok/s, would exceed
6.44), PROVIDED the bf16 router gate is first device-ified (the one remaining non-capturable host op). Rollback-
able (flag OFF). STOPPED for review before Brick D. Row `ACTIVE`; see docs/BENCHMARKS.md.
**Device-resident decode campaign — Brick D: decode CUDA graph — step 1 gated; step 2 infra gated, capture
BLOCKED on the model's Q8_0 weights (grounded finding; REVISES the Brick C attribution) (2026-07-30, base
`24bddf15`, branch `deepseek-v4-brick-d`, commits `3c262cea`+`f62de523`, NOT pushed).** STEP 1 = device router
gate (`RouterGateKernel`: [ne,H] BF16 gate × f32 x, sequential f32 + exact bf16 upcast, bit-identical to the
host CPU MatmulBT) → resident routing is device. Gate: CUDA unit **17/17·33919**; real model
resident==host **TOKEN-IDENTICAL**. STEP 2 = the graph: `decode_attn_g` (KV length from a DEVICE buffer,
attends cache[0..len)+deck_new, fixed shmem → one graph serves a growing context) + `V4Graph` (persistent
buffers, fixed-cap KV, cold→warm→capture→replay, deck_new→cache appended between replays on-stream; every
captured input a persistent buffer) + opaque holder on `DeepseekV4KvCache`; flag `VT_V4_DECODE_GRAPH=1`
(default OFF). Unit **18/18·34176** (decode_attn_g == eager decode_attn BIT-IDENTICAL + RED-first);
`test_deepseek_v4_gguf_load` 12/12·531. **BUT capture ABORTS: `matmul_bt_quant: keepquant CPU-fallback drain:
operation not permitted when stream is capturing`.** ROOT CAUSE (gguf dump): **345 Q8_0 tensors** — the MLA
projections (attn_q_a/attn_kv/wq_b), o-LoRA (wo_a/wo_b), shared experts, `output.weight` (the AProjQ8/SExpQ8/
OutQ8 in the filename). `IsCudaKeepQuantSupported` is FALSE for Q8_0 (only IQ2/IQ3/Q2_K…Q6_K have a CUDA
keep-quant GEMM), so each of ~15 Q8_0 GEMMs/layer CPU-fallbacks with a `cudaStreamSynchronize` — legal eager,
ILLEGAL during capture. **REVISES Brick C:** the eager-resident ~45% GPU-idle is dominated by these **~646
Q8_0 CPU-fallback GEMM drains/step** (MLA/shared/lm_head run on the ARM CPU), NOT device-launch overhead.
**UNBLOCK = a CUDA Q8_0 keep-quant GEMM** = the named "tuned expert GEMM" follow-on (a USER decision — not
started). Graph infra is COMPLETE + unit-gated + capture-hazard-safe behind a default-OFF flag; replays once
Q8_0 lands. FINAL honest table (real 80.7 GB, 24 decode): host 6.44 · eager-resident ~5.0-5.2 · graph BLOCKED
(Q8_0) · ds4 16.5. Rollback-able (both flags OFF). Row `ACTIVE`; see docs/BENCHMARKS.md.
**Device-resident decode campaign — the Q8_0 CUDA keep-quant GEMM (graph unblocker + dominant lever):
CAMPAIGN COMPLETE with an honest speed finding (2026-07-30, base `93e72203`, branch
`deepseek-v4-q8-cuda-gemm`, commit `fcf3003c`, NOT pushed).** Added an on-GPU Q8_0 path to
`cuda_quant_dot.cu` (both providers), moving the ~646 Q8_0 GEMMs/step (MLA/o-LoRA/shared/lm_head —
AProjQ8/SExpQ8/OutQ8) off the ARM CPU onto the GB10: `QuantizeQ8_0Kernel` (bit-exact port of the CPU
`QuantizeRowQ8_0`, 32-block Q8_0 activation) + `QuantDotGemmQ8_0Kernel`(+grouped) (warp/output, int core
bit-identical to `VecDotQ8_0Q8_0`, float scale reassociated — NMSE 5e-4 band); wired BEFORE the CPU-fallback
⇒ NO stream sync (the capture unblocker). GATE: `test_cuda_quant_dot` **2/2·105601** (Q8_0 CUDA==CPU
nmse≤1e-6 + f64≤5e-4); `test_cuda_deepseek_v4` 18/18; `test_deepseek_v4_gguf_load` 12/12; real 80.7 GB model
host + resident + **GRAPH all TOKEN-IDENTICAL** ("…Paris."); **THE DECODE GRAPH NOW CAPTURES + REPLAYS**
(0 errors, token-verified 24 steps). **FINAL honest table (2 stable runs, 24 decode): host 7.22 tok/s ·
eager-resident 5.64 · GRAPH 5.57 · ds4 16.5.** Q8_0-on-GPU lifted the HOST (shipped-default) path
6.44→7.22 (+12%) — the real win. **The resident+graph track does NOT beat host** (5.6 vs 7.22): Q8_0 on GPU
makes the resident path GPU-compute-bound (util 94-95%), the correctness-grade device glue does more GPU work
than the ARM-CPU glue, and the graph recovers nothing (already 94% busy). REVISES the thesis: eager-resident
is **glue-kernel-efficiency bound**, not launch-bound; the graph works + is token-identical (infra proven) but
is a speed dead-end on GB10 vs the fast ARM CPU. NAMED last mile to ds4's 16.5 (not started, user decision):
tune the device glue kernels, fp8 KV, and/or an async host path overlapping CPU glue with GPU GEMMs. Rollback-
able (device flags OFF; the Q8_0 GEMM is always-on + benefits every path). Row `ACTIVE`; see docs/BENCHMARKS.md.
**Device-resident decode campaign — GLUE-KERNEL TUNE (top-3 offenders): THE GRAPHED DECODE NOW BEATS HOST
(2026-07-30, base `f83c8065`, branch `deepseek-v4-glue-tune`, commit `e232146f`, NOT pushed).** After the
profile-first tunable-win verdict (glue 44% of GPU time), tuned the top-3: (1) `RouterGateKernel`
one-thread/expert → ONE WARP/expert — **18×** (7.5%→0.6%); (2) `MhcPreParallelKernel` (the #1 kernel,
`<<<1,256>>>` = 24 sequential block reductions in ONE SM) → split `MhcPreDotsKernel` (one block per mix dot →
concurrent, BIT-IDENTICAL) + `MhcPreFinishKernel` — **4.8×** (25.5%→7.5%); (3) `HcHeadKernel` `<<<1,1>>>` →
one-block parallel — **42×** (3.75 ms→88 µs/instance). GATE: `test_cuda_deepseek_v4` **18/18·34176** (tuned
kernels' equivalence cases stayed green); `test_deepseek_v4_gguf_load` 12/12·531; real 80.7 GB host + resident
+ graph all **TOKEN-IDENTICAL** ("…Paris."), graph 0 errors. **NEW TABLE (2 runs, 24 decode): host 7.20 ·
eager-resident 7.96 · GRAPH 7.92 · ds4 16.5 — THE GRAPHED DECODE BEATS HOST (+10%).** New nsys split: GEMM
~78% · GLUE ~21% (RoPE 7.3%, MhcPreFinish 6.0% incl. Sinkhorn floor, Route 4.7% — partly irreducible) · ATTN
~1.3% — the step is now GEMM-bound (full glue elimination caps ~10 tok/s; the true last mile to ds4 16.5 is
GEMM/quant microarch — fp8 KV, tuned MMQ). The H2H-memcpy lever is NOT on the critical path (util 95%,
beats host). RECOMMEND flipping the device-resident/graph default ON (fastest path; flags still default OFF
pending that call). Rollback-able. Row `ACTIVE`; see docs/BENCHMARKS.md.
**Device-resident decode campaign — DEFAULT FLIPPED: device-resident decode is the SHIPPED DEFAULT
(2026-07-30, base `73d4799c`, branch `deepseek-v4-resident-default`, commit `75c3697a`, NOT pushed).** Broadened
validation (real 80.7 GB, resident vs host, 4 prompts × 256 tok): P0 (factual) TOKEN-IDENTICAL (258 toks);
P1/P2/P3 (open-ended) diverge only at genuine near-tie positions into COHERENT, deterministic continuations
(bounded kernel noise can only flip host's ~tied top-2 → by construction near-ties, not bugs), 0 errors — the
ratified coherent-near-tie gate is met. Resident is ~1.8× the host path at a grown 256-tok context (7.8 vs 4.3).
FLIP: `ResidentDecodeEnabled` OFF→ON (`VT_V4_RESIDENT_DECODE=0` = rollback off-switch → host forward); the guard
is UNCHANGED so CPU / non-dense / T>1 (prefill) / no-KV still fall back to host (verified: prefill→host every
run; gguf_load 12/12). The CUDA graph stays OPT-IN default OFF (graph 7.92 ≈ eager-resident 7.96 — GPU-bound at
95%, the graph adds nothing on GB10 + carries the capture-hazard surface). GATE: `test_cuda_deepseek_v4`
**18/18·34176**; `test_deepseek_v4_gguf_load` **12/12·531**; real model new DEFAULT (resident) **8.01 tok/s** vs
`=0` (host) **7.24**, both "…Paris." token-identical. Shipped DeepSeek-V4 decode = device-resident ~7.96 tok/s
(~48% of ds4, GEMM-bound); last mile = fp8 KV + tuned MMQ (named residual). **★ MMQ LEVER 1 LANDED (2026-08-01, byte-exact): the ds4-gap decode kernel was MISATTRIBUTED — the prior campaign optimized the Q8_0 dense GEMV (already `__dp4a`-efficient), but the DOMINANT-FLOP routed-expert k-quant MoE GEMM (`DotQ4K`/`DotQ5K`/`DotQ3K`/`DotQ6K` in `cuda_quant_dot.cu`) was a VERBATIM SCALAR CPU port (256-B/lane `aux8` local-mem spill + scalar MAC, NO `__dp4a`) while `DotQ2K`/`DotIQ2XXS` were vectorized. Resolves the "94% util but slow" paradox (SMs busy on an inefficient kernel). `__dp4a`-vectorized `DotQ4K`+`DotQ5K` (the DeepSeek routed-expert quants; ref llama.cpp `vec_dot_q4_K_q8_1_impl_vmmq`): `test_cuda_quant_dot` BIT-FOR-BIT green (9 cases / 110432 assertions CUDA==CPU). ★ HONEST SCOPE CORRECTION: this is a GENERAL k-quant kernel win (helps Q4_K/Q5_K keep-quant models incl. the `DeepSeek-V4-Flash-MTP-Q4K` variant), but the ds4-BENCHMARK checkpoint is `IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8` — its routed experts are IQ2_XXS (gate/up) + Q2_K (down), both ALREADY `__dp4a`-vectorized (`DotIQ2XXS`/`DotQ2K`), with attn/shared/out Q8_0. So `DotQ4K`/`DotQ5K` are NOT on THIS checkpoint's decode hot path → NO ds4-gap speedup here (the lever-scan's Lever-1 premise was seeded by a wrong ground-context "routed experts Q4_K/Q5_K"; the actual checkpoint differs). The change is kept as a correct byte-exact general improvement, not a DeepSeek-benchmark closer.

(2026-08-01 nsys state and Q8_0-GEMV-lever framing superseded by the 2026-08-04 binding parity line; detail in the benchmark record.)
Row `ACTIVE`; see docs/BENCHMARKS.md.
**Last-mile campaign — Bricks 0 to 2 (2026-07-30): the keep-quant GEMM roofline profile, the `__dp4a`
vectorized-dequant matvec, the IQ2_XXS grid-lookup fix (`__constant__` to GLOBAL) and routed gate/up
activation preq-reuse. Every step bit-identical; decode 8.01 -> 10.02 tok/s.** Per-brick forensics,
profiler tables and the refuted framings live in
[.agents/benchmark-record.md](../.agents/benchmark-record.md).
**Last-mile campaign — Brick 3: Q8_0 GEMM vectorized dp4a dot — BIT-EXACT but NEAR-FLAT +0.5% (the Q8_0 matvec is
at the 34-byte-block MEMORY WALL) (2026-07-30, base `dd6cc93c`, branch `deepseek-v4-last-mile`, commit
`91e51aea`→amended, NOT pushed).** `QuantDotGemmQ8_0Kernel` (+grouped, ~45% of step at 63% of BW peak) read the 32
int8/block as 32 scattered int8 loads + a scalar-MAC loop. FIX (mirrors llama.cpp `vecdotq.cuh:vec_dot_q8_0_q8_1_impl`):
read 8 int32 + 8 `__dp4a`. The Q8_0 `qs` is at offset 2 in the 34-byte block → mis-aligned for a naked int32 load,
so `GetIntB2` (bit-exact port of `ggml-cuda/common.cuh:get_int_b2`, two uint16 loads) reconstructs the bytes →
`__dp4a` gets the same signed int8 lanes as `(int)qs[p]`. BIT-IDENTICAL: `test_cuda_quant_dot` q8_0 **2/2·105601
nmse≤1e-6 ZERO drift**; `test_cuda_deepseek_v4` 18/18·34176; `test_deepseek_v4_gguf_load` 13/13·631; real 80.7 GB
model resident-default TOKEN-IDENTICAL "…Paris." (ids byte-equal to host); SASS shows exactly 8 IDP in the Q8_0
float kernel (Iron-Law compile-proof). **RESULT (nsys, 50 tok) — HONEST NEAR-FLAT: the Q8_0 kernel moved only
2.182→2.149 s (−1.5%; med 30,112→29,472 ns) = 45.0%→44.6% ≈ 63%→~64% of peak. Decode 10.02 → 10.07 tok/s (+0.5%,
6 warm runs 10.05–10.09, non-overlapping vs Brick 2's 9.99–10.04; host `=0` 8.50→8.56) vs ds4 16.5 (~61% of ds4;
campaign host 6.44 → 10.07 = +56%).** FINDING: dp4a cut instructions but not the bottleneck — the kernel is
DRAM-BW-bound and the 34-byte-misaligned block precludes true 128-bit coalesced loads; reaching ~85% needs an
ALIGNED WEIGHT REPACK at load (numerics-neutral, but trades the in-place unified-memory mmap design). **THE
BIT-EXACT IN-KERNEL GEMM LEVERS ARE NOW LARGELY EXHAUSTED** (both grouped dequant kernels memory-bound after
B1/B1b; Q8_0 at its block wall after B3). Remaining: (a) aligned Q8_0 repack (numerics-neutral, design-changing);
(b) glue (Rope+MhcPreFinish+Route ≈ 23%); (c) numerics-delicate (dequant-cache, fp8-KV). At 10.07 ≈ 61% of ds4 vs
the ~13–15 bit-exact ceiling. STOPPED for review. Rollback `VT_V4_RESIDENT_DECODE=0`. Row `ACTIVE`; see docs/BENCHMARKS.md.
**Last-mile campaign — Brick 4: aligned Q8_0 weight repack for coalesced CUDA loads — a MEASURED NEGATIVE
(bit-exact, NO speedup, +4.7 GiB peak RSS); the coalesced-load hypothesis is REFUTED (2026-07-30, base `d1c63dc7`,
branch `deepseek-v4-last-mile`, commit `b8a3f691`→amended, NOT pushed).** Repacked the Q8_0 tower (6.146 GiB) at
load into `[all qs (16B-aligned) | all scales]` (`RepackQ8_0Cuda`) so a warp lane reads via aligned `int4` loads
(`QuantDotGemmQ8_0AlignedKernel`); memory-disciplined (copy off mmap + `DropSpanResidency`); opt-in
`VT_V4_Q8_0_ALIGN` (default OFF); `wo_a` excluded (row-sliced). BIT-EXACT: `test_cuda_quant_dot` **3/3·105841**
(new aligned==plain BYTE-IDENTICAL + RED-first); `test_cuda_deepseek_v4` 18/18; `test_deepseek_v4_gguf_load`
13/13; real 80.7 GB ALIGNED resident ids BYTE-EQUAL to plain, "…Paris.". **RESULT — NEGATIVE: decode 10.07 →
10.03 tok/s (FLAT). nsys: aligned kernel 1.671 s + still-plain wo_a 0.506 s = 2.177 s vs Brick 3's 2.149 s
(+1.3% WORSE). PEAK RSS: aligned 91.05 GiB vs plain 86.33 GiB = +4.7 GiB (safe, 28 GiB headroom; not net-flat).**
WHY: the aligned int4 loads didn't help + separating the scales added a scattered per-block gather. With Brick 3
(dp4a also no help), the Q8_0 matvec is NOT load/ALU/alignment-bound but LATENCY/OCCUPANCY-bound (1-warp/output,
~33k tiny launches/step). **THE BIT-EXACT IN-KERNEL GEMM LEVERS ARE EXHAUSTED.** RECOMMEND: do not ship — revert
(keep record) or merge default-OFF. Residual = a kernel-structure change (risky) OR fp8-KV/dequant-cache
(numerics-delicate) — a USER CALL. At 10.07 ≈ 61% of ds4 vs the ~13–15 bit-exact ceiling. STOPPED for review.
Rollback `VT_V4_Q8_0_ALIGN` default-OFF. Row `ACTIVE`; see docs/BENCHMARKS.md.
**W8-run (2): geometry FIXED — the forward now RUNS the real 158 B model end-to-end; generation still
INCOHERENT (2026-07-29, base `fba56f9b`, NOT pushed).** The layer-2 hard-fail is fixed: the DSA
compressor projects to `2*head_dim` (ds4 `coff=2`), not `head_dim`, so the real keep-quant run uses
DENSE MLA (EXACT for `seq ≤ index_topk=512`; the sparse compressor/indexer at real geometry is a named
residual), guarded by a new real-ish gate (`test_deepseek_v4_gguf_load` 9/9·388, RED-first). Also
landed: mmap-VIEW keep-quant load (PEAK RESIDENT **116.2 → 86.1 GiB**, byte-identical, load 78 s → 6 s
warm) and per-layer YaRN RoPE (compressed layers use base 160000 + freq_scale 1/16, ds4
`rope_tail_ext_inplace`). The forward now runs all 43 layers on the real model at ~1.4–1.7 s/tok / peak
86 GiB, but greedy output is a degenerate 2-cycle loop (`201 7249 465 7249…`) — INCOHERENT. A different
prompt gives different ids, so the body is input-responsive; the failure is a numerical/scaling fidelity
error (suspects: MoE routing/expert-scale/shared-expert, MHC Sinkhorn mixing, keep-quant IQ2_XXS/Q2_K
numerics), needing instrumented per-layer comparison vs the ds4 oracle. Ruled out: geometry, attention
scale/structure (match ds4), fp8 KV roundtrip, memory path, rope. ds4 reference (same file, GB10 GPU):
coherent, prefill 358 tok/s, decode 16.5 tok/s. Row stays SPIKE (no coherent generation).
**W3 attention primitives landed
(2026-07-28):** the genuinely-new-vs-V2/V3 math is ported as portable host references
and unit-gated — the DSA "Lightning Indexer" sparse top-k SELECTION (a weighted
multi-query logit with a load-bearing per-head ReLU, then a causal top-512 token
select), plus the two 512-wide-MLA output pieces V2/V3 lack: per-head attention-sink
softmax and grouped output-LoRA. `test_deepseek_v4_dsa` passes 13/13 (hand-derived
literal cases plus double-precision references, clean CPU `-Wall -Werror -Wextra`);
the gate is honest hand-case + structural review, not a dumped-oracle comparison
(the fixed-config 167B model cannot be built at a tiny shape). It is additive and
byte-neutral for DeepSeek-V2. SGLang v0.5.15 registers and implements
`DeepseekV4ForCausalLM` (the full DSA/MHC stack), so it is a viable second reference,
subject to the same single-GB10 memory limit. **W4 attention primitives landed
(2026-07-29):** the second half of the DSA stack is ported as portable host references
and unit-gated — the DSA COMPRESSOR forward (the softmax-weighted window pool that
compresses the KV-state window into one latent, computed per head-dim column, plus the
fused save-time position-embedding add and the RMSNorm) and the fp8_ds_mla KV-cache
state read/write layout (the compressed latent split into a 448-wide part quantized to
FP8 with per-64 UE8M0 power-of-two block scales and a 64-wide RoPE part stored bf16, at
a 576-byte token stride, plus the dequant read). `test_deepseek_v4_compressor` passes
12/12 (hand-derived literal cases plus double-precision references plus an independent
scale-byte recompute, RED-first proven); ported 1:1 from vLLM and cross-checked against
SGLang, additive and byte-neutral for DeepSeek-V2. **W5 Manifold Hyper-Connections
landed (2026-07-29):** the hardest V4 brick — the residual stream becomes a
`[tokens, hc_mult, hidden]` manifold mixed at every attn/ffn boundary by a
20-iteration Sinkhorn-normalized doubly-stochastic matrix, with the layer RMSNorms
folded in and a learned head that collapses the streams back to one hidden. Ported as
portable host references (the Sinkhorn, the mHC pre/post mixes, the hc_head collapse)
and unit-gated `test_deepseek_v4_mhc` 14/14 (hand-derived literals plus
from-first-principles double-precision references, RED-first proven on both the
iteration count and a normalization axis). The W0 scope's "no eager reference upstream"
premise is corrected: vLLM ships an eager PyTorch reference (`mhc/torch.py`) and four
upstream implementations agree on the Sinkhorn, so the port is grounded 1:1 and
cross-checked against an independent derivation. Additive and byte-neutral. **W6
sqrtsoftplus + hash-routed MoE landed (2026-07-29):** the three genuinely-new-vs-V2/V3
MoE pieces are ported as portable host references and unit-gated — the router score
function `sqrt(softplus(x))` (distinct from V2/V3's sigmoid/softmax), the router that
adds the correction bias for SELECTION only then either picks top-k or looks the
experts up directly in the `tid2eid` token-id→expert hash table (bypassing top-k),
gathering the weights from the UNBIASED scores, and the asymmetric clamped SwiGLU
expert activation (gate clamped max-only, up clamped both-sided). `test_deepseek_v4_moe`
passes 12/12 (hand-derived literals plus from-first-principles double-precision
references, RED-first proven on all three load-bearing levers: dropping the sqrt,
gathering weights from the biased scores, and symmetric-clamping the gate). The shared
grouped-GEMM / expert / shared-expert / NVFP4 machinery is REUSED, not re-ported (only
scoring + hash + clamp are net-new); MegaMoE is SM100-only so GB10 mirrors the
FusedMoE-fallback router. Additive and byte-neutral. **W7 forward assembly landed
(2026-07-29):** the `VT_CHECK(false, "W3-W8 pending")` stub is replaced by a real
`DeepseekV4Model::Forward` that composes the four landed host primitives (W3 DSA/MLA
seams, W4 compressor + fp8_ds_mla KV, W5 MHC + Sinkhorn, W6 sqrtsoftplus/hash MoE) into
an end-to-end logits producer on the CPU path at a small synthetic config — the brick
that finally makes V4 structurally runnable. It is a STRUCTURAL/composition gate
(`test_deepseek_v4_forward` 6/6·26: the layer interleave runs, the residual is a
`[T, hc_mult, H]` stream, the hash layers route by token id, the DSA sparse path selects
and the compressor pools, logits are finite/deterministic, and a deliberately-miswired
interleave changes the output — all-gated hash, skip-final-fold, no-sink), NOT a
real-checkpoint token gate. Honest 3-state: the CPU forward assembly at tiny shape is
DERIVED + BUILD-VERIFIED (structural); it does not claim V4 "runs" a real model. Additive
and byte-neutral (shared MLA/MoE + the W3-W6 primitive TUs untouched). The
remaining bricks — the device kernels (which reuse the existing grouped-GEMM) +
`ForwardDevice`, the real-checkpoint tower materialization, and the full strict model
gate (W8) — stay multi-Spark-blocked; the single-Spark IQ2_XXS-GGUF vehicle also needs
the V4-GGUF `blk.N.*` name map (download-blocked).
The
frontier families Kimi / MiniMax / GLM-latest are scoped for mechanical porting
in [a dedicated spike](../.agents/specs/sweep-kimi-minimax-glm-latest.md):
Kimi-Linear-48B is the one that fits GB10 (91.5 GiB) and is e2e-gateable, while
MiniMax-M2 (214.3 GiB fp8), Kimi-K2, MiniMax-M3 and GLM-5 remain hardware- or
dependency-blocked (honesty-pass only). **Kimi K3** (released 2026-07-27, after the
pin) is scoped **derive-and-ship** in [kimi-k3.md](../.agents/specs/kimi-k3.md): a
2.8T MoE whose text backbone is the Kimi-Linear KDA+MLA+MoE hybrid massively scaled
(H=7168, 93 layers = 69 KDA + 24 MLA, 896 experts, MXFP4 + MoonViT-V2 vision). It
reuses heavily (our GDN, DeepSeek MLA, DeepSeek MoE, the Qwen3.6-35B GDN-hybrid
skeleton); net-new = the KDA kernel delta, MXFP4 (group-32/e8m0), AttnRes
(report-only, unconfirmed) and the MoonViT-V2 tower. It **does not fit** one GB10
(~1.56 TB MXFP4, ~12× the 119 GiB pool) and is not in the pinned oracle, so there is
no on-box golden — like the beyond-vLLM CUDA bricks; the real signal is a primitive
gate on the fitting Kimi-Linear-48B proxy plus build-verify, with full K3
verification left to a multi-Spark / 16×H200-class box. The W2/W5 CPU scaffolding
is now **build-only** (2026-07-28): an additive registry stub, nested
text/vision/quant config descent, the 93-layer KDA/MLA + 896-expert MoE
text-backbone structural name-map (grounded in `kimi_linear.py`), a REFUSE-by-name
forward and an MXFP4-refuse loader, green on a CPU build with a 6/6 scaffold gate;
MXFP4 and the MoonViT-V2 tower stay not-yet-buildable
(deferred to the shared DeepSeek-V4 MXFP4 row and W7). The **KDA kernel delta**
itself now has a first buildable brick (2026-07-28, `KERNEL-KDA-DELTA`): the four
gated-linear-attention pieces plain GDN lacks — the per-channel `[H,D]` low-rank
decay (`f_a`/`f_b` bottleneck), the sigmoid-gated output norm
(`FusedRMSNormGated`), the three q/k/v short convs and the q/k L2-norm — are
ported as portable host references and unit-gated (`test_kimi_kda` 14/14, clean
CPU `-Wall -Werror -Wextra`) against hand-derived literal cases plus
double-precision references. KDA subclasses GDN, so its recurrence is reused and
the brick is additive (the Qwen3.6 GDN gate is byte-untouched); the gate is
host-reference + structural review, not a dumped-oracle comparison — the real
e2e gate stays the DGX-blocked Kimi-Linear-48B proxy, and the CUDA device kernel
is a named residual. Registry-metadata correction (2026-07-29): the outer
`KimiK3ForConditionalGeneration` wrapper's `_ModelInfo` now sets
`has_inner_state=false` (mirroring the Qwen3.5 hybrid-mm sibling wrappers, whose
inner language-model class carries the recurrent state), fixing the
`test_model_registry` invariant that every registered outer wrapper is
inner-state-free; `test_model_registry` 24/24. Test-golden sync (2026-07-29):
`test_model_loader_gguf`'s hardcoded supported-architectures string was likewise
stale (predated the breadth-sweep + frontier registrations); synced to include
DeepseekV4/Gemma4/KimiK3/Qwen3VL, `test_model_loader_gguf` 3/3 (was RED on main).
Env-doc hygiene (2026-07-29): `VLLM_PLUGINS` documented in
`docs/ENVIRONMENT.md`, `VLLM_GEMMA4_MM_DEBUG` classified kernel-internal;
`check-env-doc` rc=0. And the `check-device-leakage` DSR ratchet (RED from the
W7-device lane): the 8 `kCUDA` op-lookups in `deepseek_v4_device.cpp` (the
V4 device-forward resolver TU) are allowlisted with a reason + a FOLLOW-UP to thread the runner `DeviceType` through them (deferred,
needs a GB10 re-gate — DGX offline). #189 then moved the server TU into the
shared layer with 5 `VT_BENCH_PROFILE_CONTROL` guards (32→37), DSR-ALLOW'd per
site; baseline held at 32. Record checkers green on main.
The matrix opens with an
architecture-support checklist (a per-architecture status roll-up covering every
engaged model) that a CI checker keeps in lockstep with the detailed rows.

### Recent dense batch

From a **next-tier batch of recent dense text families**
([spike](../.agents/specs/sweep-recent-dense-batch.md)), the top three are now
implemented (correctness-complete or gating): **Granite-3**
(`GraniteForCausalLM`, `granite-3.3-2b-instruct`) is correctness-complete,
token-exact 16/16 vs the vLLM 0.25.0 oracle (Llama plus four scalar multipliers,
no new compute kernel); **Phi-3 / Phi-4** (`Phi3ForCausalLM`,
`Phi-4-mini-instruct`) is implemented and runs but is **not** recorded as a clean
pass: it scores 15/16, and although its LongRoPE cos/sin cache is bit-identical
to the oracle's, two positions on one prompt fall outside the near-tie band as a
cascade after an exact-tie divergence, so closing it needs a cascade-aware gate
or a strict check on the larger phi-4 (14B); **OLMo-3** (`Olmo3ForCausalLM`,
dual rope plus interleaved sliding window) is implemented and runs in our engine
but has no SACRED gate because the gate-time vLLM 0.25.0 oracle cannot run the
`OLMo-3-1025-7B` checkpoint (its transformers predates OLMo-3's per-layer-type
rope schema).

**Command-R / Cohere** (`CohereForCausalLM`, parallel-residual with a weight-only
LayerNorm and a logit scale) is now **implemented** (a faithful zero-new-kernel
port that compiles, links and self-registers) but has **no SACRED gate** yet:
every real small `CohereForCausalLM` checkpoint is HF-gated with no token on the
GPU box, the only ungated checkpoints are tiny-random (head_dim 8/2, outside the
validated attention path), and the GPU box is disk-full; the oracle was
run-verified at W0 (arch confirmed `CohereForCausalLM`, not
`Cohere2ForCausalLM`).

**Phi-1 / Phi-2** (`PhiForCausalLM`, `microsoft/phi-2`) is now
**correctness-complete** (token-exact 16/16 vs the vLLM 0.25.0 oracle: 9 strict +
7 bf16 near-ties, max gap 0.25 nats, 0 forward-divergent), the OLDER Microsoft
Phi architecture, DISTINCT from Phi-3/Phi-4: a GPT-J parallel-residual decoder
(one nn.LayerNorm with bias feeds both attention and MLP), biased q/k/v/dense
projections, partial NeoX RoPE, a non-gated NewGELU MLP and an untied biased
lm_head, all reusing landed ops (its `gelu_new` maps to the landed
`vt::GeluTanh`, so no new compute kernel), with an F16-checkpoint dtype-aware
loader mirroring vLLM's f16->bf16 cast.

**MiniCPM** (`MiniCPMForCausalLM`, `openbmb/MiniCPM-2B-sft-bf16`) is now
**correctness-complete** (token-exact 16/16 vs the vLLM 0.25.0 oracle: 10 strict
+ 6 bf16 near-ties, max gap 0.0 nats, 0 forward-divergent): the landed
Llama/Granite dense forward plus three scalars (a scale_emb embedding scale, a
scale_depth/sqrt(layers) scaled residual add on each sublayer, and a
dim_model_base logit scaling), tied embeddings, no new compute kernel. Its
checkpoint ships only as a pickle `.bin` with no safetensors and no native
transformers config, so the vehicle was prepared with a trusted torch
`.bin`->safetensors conversion of the official weights (both the oracle and our
engine read the identical bf16), and the gate feeds the oracle's exact prompt ids
(its SentencePiece tokenizer is a follow-up).

**MiniCPM3** (`MiniCPM3ForCausalLM`, `openbmb/MiniCPM3-4B`) is now
**correctness-complete** (token-exact 16/16 vs the vLLM 0.25.0 oracle: 13 strict
+ 3 bf16 near-ties, max gap 0.0 nats, 0 divergent): the first MLA-attention
MiniCPM, and the model that closes the non-trivial part of this tier. It is the
MiniCPM three-scalar skeleton with its attention swapped from GQA to
DeepSeek-style MLA, reusing the landed DeepSeek-V2 MLA block (weight absorption,
absorbed-MQA decode, materialized-MHA prefill) with only three faithful deltas: a
neox-style rope flag threaded through a new default-false field on the shared MLA
dims (so DeepSeek-V2 stays byte-identical, re-gated 8/8), a LongRoPE cos/sin
cache instead of YaRN, and a reuse-only zero-pad of the FlashAttention-2 MLA
prefill from head_dim 96 up to the already-compiled 128. No new compute kernel;
its `.bin`-only weights were converted via trusted torch like MiniCPM.

The remaining recent-dense families are the trivial tail only: **Yi**
(`YiForCausalLM`, a Llama alias) and **InternLM3** (`InternLM3ForCausalLM`,
InternLM2 plus a sliding window).

## Build and test lanes

Record hygiene (2026-08-08): a manifest/stub byte-preserves the monolithic log;
`check-state-record` range-gates bounded indexes and immutable events.

Public documentation restructure (2026-08-04): `docs/BENCHMARKS.md` had grown
into an append log of 11,405 lines and 171 claim-titled sections and was no
longer readable by users. It is now a keyed-table scoreboard, one row per
subject, updated in place; the full attempt record moved verbatim to the
append-only [.agents/benchmark-record.md](../.agents/benchmark-record.md), and
nothing was edited or deleted. A new [docs/FEATURES.md](FEATURES.md) carries the
public feature comparison against vLLM, SGLang and llama.cpp under the same
rules. Two gates keep both current: `scripts/check-public-doc-tables.py` (CI,
with mutation test `tests/scripts/test_check_public_doc_tables.py`) enforces the
required sections, the section/prose/character/table-cell budgets and the
no-em-dash house rule, and rejects the FIRST appended section rather than the
seventh; `scripts/check-doc-checkpoint.py` now also requires `docs/FEATURES.md`
to move whenever the feature, model, backend or quantization matrices do. When
sections have accumulated, `scripts/roll-benchmark-record.py --apply` relocates
them verbatim, reading its allowlist from the checker so the two cannot drift.
No engine code, no kernel, no measured number changed; the scoreboard carries
the corrected ds4 (~16.5) and Laguna (~43) denominators already on main.

Agent-record repair, triage and compaction (2026-08-04): structured state
replaces the unsound union log with ordered indexes and immutable events;
[.agents/NOW.md](../.agents/NOW.md) restores cold resume. The manual and checker
share one machine-readable contract. Only DeepSeek and Laguna remain in flight;
54 rows left `ACTIVE`, and 29 claims moved to
[completed/claims-era1.md](../.agents/completed/claims-era1-2026-08-04.md).
Backfill: 12 engine rows cite `TEST_CASE`, 19 model rows cite
`REGISTER_VLLM_MODEL`; 79 rows remain, 30 anchored. `upstream-inventory.py`
gates all 362 vLLM registry architectures (31 added).
Devices: 6/6 vLLM platforms covered; llama.cpp's 11 extra ggml backends are
spike-gated `BACKEND-GGML-*` rows (`ROAD-V1-D6`). Arch parity is checked both ways. **Role discipline
ENFORCES** since `44e8225c`: feature code reaches `main` only via a merged
`row/*` PR. **CI gap closed:** the two DIFF-scoped gates carried a
`cancel-in-progress` group contradicting their own comments, so two pushes had
them cancelled. The roadmap's 484-line chronology moved to
`completed/`, `AGENTS.md` went 697 to 286 lines with its directives verbatim in
[the archived directives](../.agents/completed/policy-directives-legacy.md), and THIS page gained a
shrink-only ratchet. The structured state record holds the detail. An operator/helper
protocol is ACCEPTED; W0-W5 LANDED, enforcement opt-in
([archived spec](../.agents/completed/operator-helper-protocol-legacy.md)). No engine code, no
kernel, no numbers changed.

Qwen3.5-4B direct-load remains speed-pending against the oracle built at the
actual parity pin. The 2026-08-05 post-upstream revalidation measured 0.9971x
total throughput and 1.1244x TPOT. Our throughput is 0.99997x the prior run;
the ratio moved because the current oracle denominator is 1.00089x its prior
run. Direct ON/OFF outputs remain 128/128 identical, so this is a benchmark
null result rather than a local regression.

One open lead is on record from the same profiling pass: cuBLASLt resolves
Ampere-class GEMM kernels on this Blackwell device. It is unmeasured and may be
a non-issue, since it lands on the axis we already pass.

Device-specific references in the device-agnostic layer stay at their ratchet
floor: capability questions ("is memory unified?") rather than device-type
tests.

Benchmark provenance is gated as well as measured: both the comparison and the
same-binary A/B harness refuse to start a leg on a GPU that is not idle, after
a cooldown rather than before it.

The ASan+UBSan and TSan CPU lanes both pass **333/333** on GCC 15.2 after
merging upstream `main`; ASan leak detection stays enabled. PR #28's hosted
failure was filesystem exhaustion, not
a sanitizer finding: sharing one internal instrumented image and using `-g1`
cut the ASan+UBSan tree from 93 GiB to 5.7 GiB; TSan occupies 1.9 GiB.
`VT_POOL_BYPASS=1` makes intentional scratch retention visible as real frees,
and the complete survey fixed one genuine minja context/callable ownership
cycle. Host sanitizers remain CPU-only; CUDA uses `compute-sanitizer`.

The plain GCC 15.2 `-Werror` build exposed two optimizer false positives. The
project-owned `std::vector<bool>` assignment now avoids the packed word-copy
path without suppression; the vendored minja diagnostic is demoted only for
`chat_template.cpp` under GNU. Every other diagnostic and unit remains fatal.
Sanitizer lanes alone omit `-Werror` because instrumentation triggers additional
libstdc++ range-analysis false positives.

CI concurrency follows gate scope. The diff-scoped documentation and commit
protocol jobs never cancel because no later run rechecks their commit range.
Tree-scoped jobs cancel superseded PR runs per ref; sanitizer groups also key on
the matrix lane so ASan+UBSan and TSan cannot cancel each other. Push workflows
key on SHA, while pull-request workflows deduplicate the same updated ref.

Known failing on discrete sm_120 (RTX 5070 Ti), pre-existing and not introduced
by the lanes: `test_cuda_ops` "CUDA matmul (cuBLASLt) matches CPU on odd sizes"
fails 11 of 442 elements on the bf16-in/bf16-out 17x31x13 case. Verified present
on a pristine build of the same commit. The project's development GPU is GB10 /
sm_121a, so this is an unattributed per-architecture numerics difference, not a
regression.

## Performance detail

**GDN recurrence output and z gate at the model dtype on every arm
(`GDN-MOE-BF16-OUT`, `GATING`,
[#1168](https://github.com/mudler/vllm.cpp/issues/1168)):** `GdnOutDType`
resolved bf16 from a dense checkpoint and f32 from a MoE one, so every MoE
checkpoint held the recurrence output `dcore`, the `z` gate and the gated-RMSNorm
weight at double width. vLLM branches on no model shape here. The shape term is
gone from the resolver and from packed-decode eligibility, and
`VT_GDN_OUT_BF16=0` is the f32 rollback for both arms now, not the dense one
alone ([spec](../.agents/specs/gdn-moe-bf16-out.md)).

**Nothing is measured and no GPU gate has run**, so no axis is claimed in either
direction. The CPU tier enters through `ModelRegistry::Forward` on a MoE config
and reads the dtypes off the tensors, in the default and the `=0` arm alike. The
35B correctness gate, the `315/315` and `235/235` engine counts, the same-binary
A/B and the `nsys` memory-format confirmation are owed to a GPU host. Dropping
the shape term reaches packed GDN decode on no MoE checkpoint either:
`in_proj_ba` has one writer, the dense loader
([#1169](https://github.com/mudler/vllm.cpp/issues/1169), owed).

**Local Qwen3.5-4B plain BF16 direct loader; throughput ahead, acceptance
`PENDING`, latency and VRAM open:** production `AsyncLLM` uses default-ON exact
`(sequence, 8-token chunk)` causal-conv dispatch. Graph-node `nsys` measures
**718.704→233.955 ms (3.072x)**, leaving 1.609x to vLLM.

ONE cross-engine result stands, the corrected three-repetition run: throughput
**6831.71 vs 6643.40 tok/s (1.0283x)**, host PSS lower; TTFT **1.0853x**, TPOT
**1.0165x**, E2E **1.0288x** slower, VRAM **+118.7 MiB** OPEN. The earlier
**1.021246x** is SUPERSEDED, not a rival: it predates the atomic wave
admission that removed the frontend confound. Disposition `PENDING`, matching
[the spec](../.agents/specs/sm120-qwen35-pareto-2026-08-09.md) — mutation
re-review, operator gate and real-GPU token identity are owed; the GDN
selectors stay default OFF. The 18-leg oracle runs were VOID.

This local 4B diagnostic does not establish 27B/35B support:
[exact-chunk outcome](bench-evidence/qwen35-4b-sm120-main-20260807.md).

**27B fp8 tower, packed GDN decode now REACHABLE (`PERF-GDN-PACKED-BRIDGE`,
#365):** selection PROVEN 48/48 on `nvidia`@`0893e160`. Decode c1 at
`input_len=16` reads `0.977x -> 0.984x` against the pin — INDICATIVE, not
binding: the arms were not interleaved and did not share a background state.
Only the two structural kernel terms (-0.400, -0.131 ms/step) exceed the
untouched control's own +0.262 drift. Tokens move on neither 27B checkpoint,
which establishes no gross defect at ~50x coarser sensitivity than the
perturbation introduced, not equivalence. **That indicative reading is now
superseded by an interleaved same-binary A/B** on the 1024/128 workload at the
pin and a pinned clock: decode **1.007x** (c1), **1.012x** (c4), 1.027x (c16,
#577-exposed); TTFT **1.048x** (c1), **1.041x** (c4). Both toggles stay DEFAULT
OFF; flipping a default is a separate change with its own token gate
([spec](../.agents/specs/perf-gdn-packed-bridge.md),
[record](../.agents/benchmark-record.md)).

**Binding parity at the pin (2026-08-13), 1024/128, n=3, arms interleaved,
clocks flat at 2184 MHz:** 27B decode TPOT **0.976x** (c1) and **0.946x** (c4),
prefill TTFT 0.944x (c1); 35B decode TPOT **0.995x** (c1) and **0.946x** (c4),
prefill TTFT 0.920x (c1) and 0.849x (c4). 27B c16 is **VOID, not a number**: our
arm completed 93 of 96 requests where the pin completed 96 of 96 and the three
missing are the SLOWEST, because our SSE keepalive (`VT_SERVER_SSE_PING_S`,
then-default 15 s, now OFF per #931) was ENABLED on every leg
([#577](https://github.com/mudler/vllm.cpp/issues/577)). 35B c16 TPOT and TTFT
are NOT ESTABLISHED (6.8% and 15.0% leg spread). Forensics in
[the benchmark record](../.agents/benchmark-record.md).

There is no front-page race clip yet; when one is produced it will follow the
LocalAI house style (side-by-side, identical output, honest measured ratios).

## Backend detail

Gemma4/ROCm env split: public `VT_GEMMA4_EXPERT_VRAM_MB` caps expert LRU in positive MiB (unset/0 unlimited); `VT_SERVER_MAX_{PROMPT_CHARS,NEW_TOKENS}` guard requests at 200000/4096 (0 disables); nine inherited tuning toggles are internal. No runtime/perf change.

`BACKEND-TENSTORRENT`: `ACTIVE`: OPT-125m strict 6/6 on Blackhole; 17 ops. Qwen3 is wired; its full 16x16 gate and speed remain pending.

`BACKEND-TENSTORRENT-MISTRAL`: `ACTIVE`: Mistral-7B-v0.3 gated on a Blackhole P150, 16/16 prompts (12/16 strict token-exact, 4/16 inside the near-tie band, 0 forward-divergent), max gap 0.062 nats. `MistralForCausalLM` is allowlisted by exact match, so `Mistral3ForConditionalGeneration` (#387, unported) still falls through. Correctness only -- no speed claim.

`BACKEND-TENSTORRENT-TRACE-RUNNER`: `SPIKE`: NO-GO for T=1 decode capture. Measured `to_vector` abort. Decode capture moved to `BACKEND-TENSTORRENT-HOST-FREE-FORWARD`. Prefill capture still unaudited.

`BACKEND-TENSTORRENT-HOST-FREE-FORWARD`: `ACTIVE`: env-gated `VT_TT_HOST_FREE_DECODE` decode-graph capture. Implementer P150 run of Qwen3-0.6B, 80 tokens: 79 replays, no hang, 5.8x vs eager, 22/22 vs the per-step-copy baseline. Default path inert. Operator gate and full-engine golden still owed. A new batch after the first capture is refused.

`ENG-CUDAGRAPH-DEDUP`: `ACTIVE`: env-gated `VT_CUDA_GRAPH_DEDUP` graph-executable dedup — one `cudaGraphExec` per captured TOPOLOGY instead of one per padded decode bucket per model. The GB10 A/B of 2026-08-18 split: byte-identical 10/10, [#1184](https://github.com/mudler/vllm.cpp/issues/1184) gone, and the fold never engaged because the shipped key carries the padded batch dimension.

`ENG-CUDAGRAPH-DEDUP` coarse key: the [#1226](https://github.com/mudler/vllm.cpp/issues/1226) hypothesis is CONFIRMED on GB10. Drop the launch dimensions and every bucket folds — 3 graphs to 2 execs, 2 to 1, 2 to 1 — with `cudaGraphExecUpdate` probed once per fold and `refused=0`. It is opt-in, default OFF, and PR #1232 has NOT landed, so nothing on `main` folds today. No memory or speed number: clocks were unpinned and nobody measured bytes saved.

**Platform SELECTION is the one non-additive site, and is now gated.** A
platform missing from `CurrentPlatform()`'s hardcoded walk registers and answers
correctly but is NEVER selected, with no compiler diagnostic. `test_platform`
now gates that every `DeviceType` is in the walk and CPU is last.

**The device-scratch pool is now ONE POOL PER DEVICE (`POOL-DEVICE-KEY`, #516).**
It was a process-wide free list keyed by byte size class with no device in the
key, so in a mixed-backend process a block allocated through one backend was
handed to the next caller of that class on another: a `cudaMalloc` block reaching
a CPU forward SIGSEGVs host-side with `compute-sanitizer` clean, and a host block
reaching a CUDA forward returned a uniform `0x7fff0000` quiet NaN. A pool is now
bound to a backend, `Pool(b)` is the only spelling, every operation refuses a
foreign backend, and the two per-caller workarounds are deleted. `test_device_pool`
gates it without a GPU; `VT_POOL_BYPASS`/`VT_POOL_EXACT` keep their meanings and
the suite is green under both.

**CUDA architectures.** The runtime-gated production arch is GB10 `sm_121a`
(every gate model, every benchmark). A build-supported cross-family fan-out
(`sm_80/86/87/89`, `sm_90a`, `sm_100a/103a`, `sm_110`) compiles single-arch,
portable-kernels-only (fp8/fp4/CUTLASS/Marlin/FA2 fast paths resolve EMPTY).
**As of 2026-07-27, `sm_110` is RUNTIME-VERIFIED on real silicon — the FIRST
non-GB10 runtime proof.** vllm.cpp was built natively for `sm_110` on an NVIDIA
Jetson Thor board (aarch64, JetPack R38, nvcc 13.0, `compute_cap=11.0`; cutlass
absent and not needed), Release `-Werror` 0 warnings, 16 TUs of real `sm_110`
SASS. It ran the Llama-3.2-1B paged-engine
greedy gate and was **STRICT token-exact 12/16 prompts (192/192 tokens) vs the
committed vLLM oracle golden** (every vLLM-deterministic prompt),
**15/16 bit-identical to the GB10 `sm_121a` anchor**; the remaining 4/16 are the
ratified bf16 near-tie prompts (gap 0.000 nats). Scope covers only the portable
bf16 path that ran; `sm_110` fp8/fp4/CUTLASS remain DERIVED/NOT-YET. Other
fan-out boards are build-supported only (no board here). Repro and evidence: [docs/BENCHMARKS.md](BENCHMARKS.md),
[.agents/backend-matrix.md](../.agents/backend-matrix.md) `BACKEND-CUDA-SM110`.

**GDN Triton-AOT cubins are vendored per-arch (2026-07-28).** The GDN AOT set
(packed decode + the delta_h/chunk_o FLA kernels) is vendored for
`sm_80/86/89/90a/100a` as well as `sm_121a`; `cuobjdump` shows per-target SASS,
and `sm_121a` is byte-unchanged (SACRED gate intact).

**`VLLM_CPP_TRITON` defaults ON for every CUDA arch (#219).** `ON` when
`VLLM_CPP_CUDA` is `ON`, `VLLM_CPP_TRITON_REGEN` is `OFF`, and every vendored
tree passes the builder's drift/staleness checks; `OFF` otherwise, naming the
condition. Arch COUNT is not a condition.

**RISK of that default, DISCLOSED NOT MITIGATED.** It selects all five
non-`sm_121a` trees unasked, yet only `sm_121a` is RUNTIME-VERIFIED; the rest
stay **DERIVED+BUILD-VERIFIED (testing-welcome)** — no such board here runs a
GDN model — and `sm_80` has open
[#193](https://github.com/mudler/vllm.cpp/issues/193) (crashes, wrong GDN
output) reported at Triton OFF, so not caused by this. `-DVLLM_CPP_TRITON=OFF`
restores the hand kernels. Specs:
[per-arch](../.agents/specs/triton-aot-per-arch.md),
[default-on](../.agents/specs/triton-aot-default-on.md).

**As of 2026-07-28, `sm_87` is also RUNTIME-VERIFIED (portable bf16 SYNC path) on
real silicon — the SECOND non-GB10 runtime proof.** vllm.cpp was built
portable-only for `sm_87` on an NVIDIA Jetson AGX Orin (Tegra R36.4.3 / JetPack 6,
aarch64; the integrated GPU self-reports `sm_87`, unified memory; toolchain
detail in the matrix row). All fp8/fp4/CUTLASS/FA2 fast paths resolve EMPTY on
`sm_87` (Ampere: bf16 + int8; no cutlass). It ran the
Llama-3.2-1B paged-engine greedy gate and was **13/16 prompts
STRICT token-exact vs the committed vLLM 0.25.0 oracle golden, 16/16 under the
near-tie distributional gate, 0 forward-divergent** (exceeding Thor's 12/16), plus
`test_cuda_backend`/`test_cuda_ops` (461 assertions of real on-device kernel
execution). One honest `sm_87` bug surfaced: the DEFAULT async runner path crashes
on the first forward with an illegal memory access, so RUNTIME-VERIFIED is scoped
to the portable bf16 **synchronous** path (`VT_ASYNC_RUNNER=0`); the async runner
on `sm_87` is a tracked unblock item. Repro and evidence:
[docs/BENCHMARKS.md](BENCHMARKS.md),
[.agents/backend-matrix.md](../.agents/backend-matrix.md) `BACKEND-CUDA-SM087`.

As of 2026-07-28, the datacenter-Blackwell `sm_100a` fan-out gained its first
FAST-PATH body: the **NVFP4 tcgen05 block-scaled GEMM is BUILD-VERIFIED**
(`DERIVED+BUILD-VERIFIED, testing-welcome`). A faithful 1:1 type-port of vLLM's
`Fp4GemmSm100` (`ArchTag=Sm100` + `KernelScheduleAuto` — CUTLASS 4.5.0 selects the
5th-gen tcgen05 collective) compiles single-arch `100a` `-Werror`-clean and
`cuobjdump` shows a real `sm_100a` cubin; its own `cutlass-nvfp4-sm100` cell
gates it, so the GB10 `sm_121a` gate build is byte-unchanged. The native consumer `mma.sync kind::mxf4nvf4` fp4
path stays `sm_12x`-only (it does not port to sm_100's tcgen05). **No B200/sm_100
board ran it**: a green compile + SASS is not execution evidence; the other
sm_100 fast paths (CUTLASS C3x FP8, MoE, MXFP4, MLA) remain scoped.

As of 2026-07-28, the Hopper `sm_90a` fan-out gained its first FAST-PATH body: the
**CUTLASS C3x FP8 (W8A8) scaled-mm GEMM is BUILD-VERIFIED**
(`DERIVED+BUILD-VERIFIED, testing-welcome`). A faithful 1:1 type-port of vLLM's
`cutlass_3x_gemm_sm90_fp8` (`ArchTag=Sm90` + `KernelTmaWarpSpecialized*FP8FastAccum`
— CUTLASS 4.5.0 emits the 4th-gen wgmma/TMA warp-specialized collective) compiles
single-arch `90a` `-Werror`-clean and `cuobjdump` shows a real `sm_90a` cubin
(ptxas names `wgmma.mma_async` in it); its own `scaledmm-c3x-sm90` cell gates it,
so the GB10 `sm_121a` gate build is byte-unchanged. The consumer `sm_12x` FP8 scaled-mm
body (`ArchTag=Sm120`) is untouched. **No H100/H200/sm_90 board ran it** — a green
compile + SASS is not execution evidence; the sm90 int8/blockwise C3x legs and
the other Hopper fast paths (FA3, Machete, CUTLASS MoE) remain scoped.

As of 2026-07-28, the datacenter-Blackwell `sm_100a` fan-out gained its second
FAST-PATH body (after the DC1 NVFP4 tcgen05 GEMM): the **CUTLASS C3x FP8 (W8A8)
scaled-mm tcgen05 GEMM is BUILD-VERIFIED** (`DERIVED+BUILD-VERIFIED,
testing-welcome`) — the intersection of DC1's tcgen05 arch and the Hopper DC2 C3x
fp8 kernel. A faithful 1:1 type-port of vLLM's `cutlass_3x_gemm_sm100_fp8`
(`ArchTag=Sm100` + `KernelScheduleAuto`/`EpilogueScheduleAuto` — CUTLASS 4.5.0
selects the 5th-gen **tcgen05** warp-specialized collective, a 2SM
`ClusterShape<_2,_2,_1>` default) compiles single-arch `100a` on nvcc 13.0
`-Werror all-warnings` 0 warnings and `cuobjdump` shows a real `sm_100a` cubin (the
emitted kernels are `Sm100TmaUmmaWarpSpecialized` naming `SM100_MMA_F8F6F4`,
`SM100_TMA_2SM_LOAD` and `SM100_TMEM_LOAD`; SASS carries `LDTM`/`tmem`
tensor-memory ops); it is gated by its own `scaledmm-c3x-sm100` feature cell
(enabled only for `100a`, so the GB10 `sm_121a` gate build is byte-unchanged). The
consumer `sm_12x` FP8 body (`ArchTag=Sm120`) and the DC1 sm100 NVFP4 leg are
untouched. **No B200/sm_100 board ran it** — a green compile + SASS is not
execution evidence, not runtime support, and not vLLM-competitive; the sm100
int8/blockwise C3x legs and the other datacenter fast paths (CUTLASS MoE Sm100,
MXFP4, CUTLASS MLA) remain scoped.

A separate **beyond-vLLM breadth lane** targets the older NVIDIA arches vLLM
DROPS but llama.cpp still runs (Pascal/Volta/Turing). Its first brick landed
2026-07-28: **the Turing `sm_75` attention TU is BUILD-VERIFIED.** The bf16-WMMA
prefill kernels (bf16 tensor-core fragments are Ampere+ only) are now guarded
`#if __CUDA_ARCH__ >= 800`, so `cuda_paged_attn.cu` compiles `-Werror` 0
warnings at single-arch `75` on nvcc 13.0 and `cuobjdump` shows a real `sm_75`
cubin — the TU falls back to the existing scalar CUDA-core attention path (the
llama.cpp fp16 fast body is a later brick). On `sm_80`+ the guard is a no-op
(preprocessor identity), so the GB10 gate build is byte-identical. This is
**DERIVED+BUILD-VERIFIED (testing-welcome), NOT runtime** — no Turing board ran
it; there is no vLLM oracle on Turing (vLLM dropped it), so correctness is
referenced against llama.cpp-on-card plus a portable cross-check. **A 2026-08-06 audit of all 20 unconditionally-built CUDA TUs at `sm_75` now
measures 20 PASS / 0 FAIL** (W1b guards, W1c selector gating); the
full-library LINK + SASS proof is owed — see below. Volta/Pascal
need a CUDA `<13` toolkit (nvcc 13 rejects `sm_70`/`sm_60`/`sm_61`). See
[.agents/backend-matrix.md](../.agents/backend-matrix.md) `BACKEND-CUDA-SM075`.

**Metal (Apple Silicon), indicative, not binding.** Two models run end to end
and pass correctness (OPT-125m, Qwen3-0.6B); 18 of 75 ops are native, the rest
fall back to CPU on unified memory. Kernel work including mma prefill attention
(4.3x), a vectorised decode V accumulation (+1.66%) and vectorised prefill
attention staging (+0.50%) puts warm b=1 throughput at about 95.9% of MLX-LM by default, and 97.6% with the optional MLX GEMM provider
shape-gated to prefill (`-DVLLM_CPP_MLX=ON`), which is a numerics-deviating
configuration. Both figures are against an interleaved 6-run MLX-LM baseline;
an earlier 99.1% claim used a 2-run baseline containing an outlier. With MLX
gated, prefill is 1.5% AHEAD of MLX-LM and the entire remaining deficit is
decode. The GEMV runs at 83% of memory peak; decode attention runs 2.6x slower
per byte than the GEMV because at b=1 it gets 8x fewer threads in flight, and
three separate attempts to exploit that (flash-decoding, GQA grouping, wider
threadgroups) have each measured worse. Bisecting the GEMM puts it at 97% of MLX's own
(mma issue rate 3.91 TFLOP/s), so the residual is NOT GEMM-led: it is spread
across prefill attention, small prefill kernels and decode, none dominant. The
decode share is 93% weight streaming at 83% of the part's memory peak, so the
remaining items sit against roofs rather than against defects: GEMV at 83% of
memory peak, GEMM at 97% of MLX's, and a re-test of GEMV memory-level
parallelism under the paired harness confirmed the earlier exclusion at -1.03%.
The decode residual is non-GEMV overhead (2.7 ms/token against MLX's implied
2.0), so the next lever is a fused qk-norm-RoPE attention
preamble. The dense recipe (kAttnQkNormRope) is composite-only on every backend
(fast_op = kNoFastOp), so this meant writing the kernel rather than porting one.
That kernel now exists and is validated on Metal (worst element error 1.4e-06 vs
the composite); the default bf16 path is now routed through it
(gated on the rope cache), worth a measured +0.35% and taking warm b=1 throughput
to about 96.0-96.4% of MLX-LM, with decode at 98.0%. The largest remaining lead
is prefill attention, which at 547 GFLOP/s is still 5.2x slower per FLOP than
this device's own GEMM; its online softmax now runs one simdgroup per query
row rather than one thread, worth -16% on the kernel and +0.19% end to end. Its
remaining 4.4x FLOP-rate gap to this device's own GEMM has no open structural
idea left: BK deepening is blocked by a bf16-P precision constraint and eliding
the identity softmax rescale measured no gain. Prefill's
small-kernel bucket is ~36 ms against MLX-LM's implied ~25, so improving our own
architecture there caps out near 97% rather than parity. A spike comparing MLX's
lazy-graph execution model against ours found our matmul chain already +1.5%
ahead of MLX's own ceiling, so adopting MLX's execution model would not help
decode; the residual is small-op fusion, which is ours to do. The V-accumulation win came from BISECTING the attention
kernel once a paired ABBA harness with cooldown made 0.2% differences
measurable: the V loop moved 2 bytes per lane per load where the score loop
moved 8, giving 29 GB/s against 64 on identical traffic. That also explains why
five earlier decode hypotheses (fusion, split-K, KV layout, threadgroup width,
barrier depth) were each refuted — the limiter was bytes per instruction in one
loop, not parallelism or layout. An optional MLX GEMM provider is available via
`-DVLLM_CPP_MLX=ON` and currently measures net slower than our own kernels. Full
per-lever chronology: [docs/BENCHMARKS.md](BENCHMARKS.md) and
[.agents/specs/metal-dispatch-attribution.md](../.agents/specs/metal-dispatch-attribution.md).
The MLX-enabled Darwin build remains **build-verification pending**: MLX is now
an imported CMake dependency whose public include directory is explicitly
`SYSTEM`, and the provider additionally scopes AppleClang's
`-Wgnu-folding-constant` suppression to the MLX public includes. Third-party
header diagnostics stay outside project `-Werror`, while vllm.cpp warnings
remain fatal. Darwin CI is the binding AppleClang gate.

**CUDA architectures.** The production target is GB10/`sm_121a` (runtime-gated,
both gate models token-exact + at/above vLLM speed). The arch-additivity
framework makes adding a CUDA architecture a data edit; `sm_120a`, `sm_90a`,
Ampere/Ada `sm_80/86/87/89` and datacenter Blackwell `sm_100a/103a` are
**BUILD-supported, portable-kernels-only** (they compile `-Werror`-clean and emit
real per-arch SASS on dgx, but no such board runs here, so none is runtime
support). The **Ampere major-8 fast-path bring-up is now SPIKED** as a
derive-and-ship campaign
([.agents/specs/cuda-arch-ampere-fastpath.md](../.agents/specs/cuda-arch-ampere-fastpath.md)):
the fast-path kernel bodies vLLM builds for Ampere (FA2, Marlin int4 W4A16 +
fp8-input, AllSpark W8A16, CUTLASS scaled-mm C2x) will be ported 1:1, build-verified,
and shipped **labeled** DERIVED+BUILD-VERIFIED (testing-welcome) — a build is never
a runtime claim. AGX Orin (`sm_87`) is the one reachable non-GB10 board and the
sole target that reaches RUNTIME-VERIFIED, gating the first non-GB10 runtime proof
token-exact vs the vLLM 0.25.0 oracle. **FA2 Ampere (WA-1) has LANDED
DERIVED+BUILD-VERIFIED (testing-welcome) — the `ROAD-V1-D1-CUDA` first brick:** the
`fa2` FEATURE-TABLE cell was widened to `8.0,8.6,8.7,8.9,12.0a,12.1a`
and the FA2 build gate decoupled from the sm_12x NVFP4 flag onto arch-independent
CUTLASS-header availability; single-arch `87` and `80` dgx builds are `-Werror`
0-warn with `cuobjdump`-proven real `sm_87`/`sm_80`/`sm_86` FA2 cubins (`86`
direct 2026-08-06 7/7 TUs; `89` inherits), and `sm_121a` is unchanged (NVFP4 GEMM
still ON) — verified by a re-run gate. NO Ampere board ran it here. Missing
CUTLASS on an FA2-capable arch now WARNS instead of silently dropping FA2. AllSpark and
scaled-mm C2x are new bodies; Marlin needs its int4/fp8 instantiations. The whole CUDA-arch
fast-path + beyond-vLLM breadth effort is tracked on the roadmap as
`ROAD-V1-D1-CUDA` (derive-and-ship + a public tested/untested signal matrix), with
AGX Orin (sm_87) and NVIDIA Thor (Blackwell) as the reachable runtime-gate boards.

**Datacenter CUDA arches (Hopper `sm_90a`, Blackwell `sm_100/103/110`) — build-only, fast-path SPIKED.** These arches compile today as **portable-kernels-only** (`build-only`; every wgmma/tcgen05 fast-path FEATURE-TABLE cell resolves EMPTY, no board here) — see [.agents/backend-matrix.md](../.agents/backend-matrix.md). The FRAMEWORK (arch-additivity seams) is done; the FAST-PATH kernel bodies are now scoped for **derive-and-ship** (the llama.cpp model) in [.agents/specs/cuda-arch-datacenter-fastpath.md](../.agents/specs/cuda-arch-datacenter-fastpath.md). The CUTLASS C3x / NVFP4-tcgen05 / grouped-MoE / MLA kernels are CUTLASS template instantiations (the wgmma/tcgen05 MMA lives in cutlass 4.5.0, selected by `ArchTag`), so they are 1:1-portable and BUILD-VERIFIABLE here via a single-arch `90a`/`100a` cross-compile + `cuobjdump` SASS proof, with NO Hopper/B200 board. Such kernels will ship LABELED `DERIVED+BUILD-VERIFIED (testing-welcome)` — a faithful port with a compile+SASS proof, HONESTLY untested (a green link is not execution evidence); cloud-GPU token-exact + every-axis runs upgrade the label to runtime-verified. DeepGEMM (runtime JIT/autotune) is the one path that is `not-yet-buildable / needs-real-port` and ships no fake.

**CUDA arch breadth beyond vLLM (Pascal / Volta / Turing).** These
are older NVIDIA arches vLLM DROPS but llama.cpp still supports, so this is a
"support more than vLLM" lane with llama.cpp as both the kernel source and the
competitor floor. Our portable attention uses bf16 tensor-core WMMA, which does
not exist before Ampere, so these arches need the bf16-WMMA path compile-guarded
(and, for a fast path, a new fp16/non-tensor-core kernel body ported 1:1 from
llama.cpp's `fattn-tile`/`fattn-vec`). The lane is derive-and-ship, and its first
brick has LANDED: the **Turing (`sm_75`, e.g. the cloud-common T4) attention TU is
BUILD-VERIFIED** (2026-07-28, `CLAIM-CUDA-TURING-SM75`). The bf16-WMMA prefill
kernels are now guarded `#if __CUDA_ARCH__ >= 800`, so `cuda_paged_attn.cu`
compiles `-Werror` 0-warn at single-arch `75` on our current CUDA 13 toolkit and
`cuobjdump` shows a real `sm_75` cubin — the TU falls back to the existing scalar
CUDA-core attention path (the fp16 fast body is a separate later brick). On
`sm_80`+ the guard is a no-op (byte-identical GB10 build). It ships labeled
"derived, build-verified, not hardware-tested here, community testing welcome" —
no Turing board ran it; a green compile + SASS is not execution evidence.

**Scope correction (2026-08-06): that was ONE translation unit, not a library
build**, and this page previously read as the latter. A full compile audit of all
20 unconditionally-built CUDA TUs at `sm_75` (nvcc 13.0.88) first measured 18 PASS
/ 2 FAIL and **now measures 20/20 PASS, 0 errors and 0 warnings**. The two
residuals (`cuda_gdn.cu`, `cuda_matmul_nvfp4.cu`) carried bf16 WMMA fragments plus
a harder second class, `wmma::precision::tf32` — also Ampere+, but failing at the
type-alias definition rather than at a use site, so a body-only guard does not
compile. Both are now wrapped `#if __CUDA_ARCH__ >= 800` with a `__trap()`
fallback, along with the helpers only those bodies use. All eight fast-path
feature cells resolve DISABLED at `75`.

**The three WMMA selectors are arch-gated too, which the compile guards alone did
not cover:** each predicate was host-side only (shape, dtype, env), so a
pre-Ampere board would still have selected a `__trap()` body — a build error
turned into a runtime crash. All three now require `sm_major >= 8` and fall
through to paths that already exist: the portable CUDA-core flash (attention), the
sequential scan (GDN), and the naive/tiled/split-K kernels (NVFP4/MoE). Each fails
safe if the device capability is unreadable. **GB10 is unaffected by measurement**
— the three TUs compile 0-warn at `sm_121a` with byte-identical SASS.

**No `sm_75` library link exists yet** — a per-TU compile sweep is not a link, and
no Turing, Volta or Pascal board has run any of this. bf16 needs no fp16 model
path here: there are zero bf16 *arithmetic* intrinsics in the CUDA tree
(convert-on-load, compute in fp32), so models stay bf16 and only WMMA fragment
instantiation is Ampere-gated. **Volta (`sm_70`, V100) and Pascal are
not-yet-buildable** — CUDA 13 dropped their code generation and no 12.x toolkit is
provisioned here; the fix list transfers to Volta by construction, the SASS proof
does not. There is no vLLM oracle on these cards, so real correctness testing
means llama.cpp on the same card plus a newer-card/CPU cross-check; nothing is
runtime-verified yet.

## Serving and API notes

- **Published recipe commands reach model load (`SERVE-RECIPE-ARGS`, `ACTIVE`,
  issue #606, `.agents/specs/serve-recipe-args.md`).** `vllm-serve` rejects any
  argument it does not recognise, which is the right default and was also why
  copy-pasting an official `vllm serve` line aborted before a weight was read:
  over `vllm-project/recipes` @ `86c7777a` (157 model recipes),
  `--enable-auto-tool-choice` appears in 89 and `--trust-remote-code` in 82, and
  neither means anything to this engine. Both are now **accepted and inert**,
  each announcing itself and the reason it does nothing at startup. The list is
  **enumerated, not a catch-all**: any other unrecognised flag still aborts with
  the same message, including flags inert only because the capability is missing
  (`--tensor-parallel-size` and the other parallelism flags), and upstream's
  validation still fires — `--enable-auto-tool-choice` with
  `--tool-call-parser none` is refused, mirroring
  `vllm/entrypoints/openai/cli_args.py:395`. Not yet reviewed or gate-rerun by
  the operator; `--language-model-only` was excluded from this list because it
  is a real capability, not a no-op — and as of 2026-08-14 (#607 wave L2) it is
  **implemented** rather than accepted-and-inert, alongside
  `--limit-mm-per-prompt`. Both set `vllm::MultiModalConfig` and are ENFORCED: a
  server started with `--language-model-only` answers a multimodal request with
  HTTP 400 `At most 0 image(s) may be provided in one prompt.`, which is what
  upstream's flag does. The 43 recipes that pass it now reach model load. It
  does **not** free memory yet — nothing gates vision-tower construction on the
  limits (wave L3, owed with a measured RSS reduction), so the flag must not be
  described as a VRAM knob until that lands.
- **Surface coverage (ONE SURFACE, `ARCH-ONE-SURFACE`,
  `.agents/specs/surface-coverage-2026-08-07.md`).** 21/30 text archs
  on-framework; the recurring defect (a capability in a per-model CLI) is in seven
  lanes. Guard `scripts/check-surface-coverage.py` (two axes, preflight + CI):
  every `examples/*` unit is a client of `include/vllm.h` or tracked to a fold
  row; every `FEATURES.md` C-ABI capability names an entry point or is tracked.
  ROW 8 ABI-v14 device selection is behavior-complete; #139 repairs #136's
  shared-layer DSR 39->32 through registry/name resolution (`kcuda=0`) with the
  baseline unchanged. CPU: selector 2/2·11; semantic execution guard 52/52
  (CTest config/enabled, CI/preflight, manifest integrity). CUDA A/B remains.
- **Automatic prefix caching (APC)** is on by default for dense models (hybrid /
  GDN and attention-free default off, mirroring vLLM), and it now has an
  end-to-end cache-ON gate on `Qwen/Qwen3-4B` (a shared common prefix reused
  across requests). A cache hit never changes the output: cache-ON and cache-OFF
  produce token-identical greedy decodes (differing only where the bf16 model
  itself is a genuine near-tie, exactly as vLLM's own greedy does), and both
  match vLLM's own greedy under teacher-forcing. The prefix cache measurably hits
  (0.807 hit rate on that workload) and pays off: on a cached request the time to
  first token drops about 1.76x versus the cold first request (70 ms to 40 ms on
  the GB10), because the shared prefix's attention is not recomputed. Hit-rate
  statistics are counted per vLLM's own counters. Block-hash extra keys
  (multimodal hash, LoRA adapter name, `cache_salt`) are folded into the block
  hash exactly as vLLM does, so requests differing only in those never share a
  cache block (proven no-false-share). The `cache_salt` OpenAI request field is
  not surfaced yet. A Prometheus `/metrics` endpoint is available (opt-in): it
  exposes vLLM's `vllm:*` metric names, help text, types, buckets and
  `{model_name, engine}` label schema in text format 0.0.4, gated against vLLM's
  own scrape spec. It now serves live per-step values, not just the schema: each
  engine step folds the scheduler snapshot (running/waiting counts, KV-cache
  usage, prefix-cache queries and hits) and the iteration stats (prompt and
  generation token counts, request-success counts, and the time-to-first-token,
  inter-token-latency and end-to-end latency histograms) into the registry,
  matching vLLM's own mapping. The scheduler also records per-request
  QUEUED/SCHEDULED/PREEMPTED engine-core events (1:1 with vLLM, behind the same
  stats gate), which the frontend folds into the per-request queue, prefill and
  inference timing histograms and the preemption counter, so those series now
  carry real durations rather than staying at zero. Both frontends record: the
  synchronous engine and the asynchronous one the HTTP server serves from, each
  behaviourally CPU-gated. The remaining work is the chat/completion
  response-body timing surface and the config-gated metric families (speculative
  decoding, KV connector, multimodal cache, LoRA).
- **SGLang RadixAttention behavior parity** — scoped 2026-07-27, W1+W2 flags now
  IMPLEMENTED (CPU-gated). SGLang's radix-tree prefix cache is functionally
  equivalent to our block-hash APC (both do automatic longest-prefix KV sharing
  with LRU eviction; the only delta is token/page vs block sharing granularity,
  which is bounded and never changes the output), so RadixAttention is fused into
  APC and `--enable-radix-attention` / `--disable-radix-attention` are aliases for
  the existing prefix-cache toggle (also reachable via the C-ABI tri-state
  `vllm_model_params.enable_prefix_caching`, ABI v7), not a second cache. SGLang's
  overlap scheduler is likewise already our async scheduler. The one
  genuinely-distinct SGLang behavior — cache-aware admission that reorders the
  waiting queue by longest prefix match — is implemented behind
  `--schedule-policy=lpm` (`SchedulerPolicy::kLPM`, output-neutral: it changes only
  which request is admitted first, never any request's tokens; default stays
  `fcfs`). SGLang's in-batch prefix-collision de-prioritization (SW2) is now also
  implemented inside the `lpm` path (a later request that would collide on the
  same not-yet-cached prefix as an earlier waiting request is sorted behind the
  non-colliding requests), still output-neutral. Note: this de-prioritization
  changes only admission order here — our prefix cache already caches a request's
  blocks at allocation time, so a second same-step request sharing an uncached
  prefix already reuses the first's blocks (the redundant-prefill this avoids in
  SGLang, whose cache updates only after the forward pass, does not occur for us).
  Jump-forward constrained decoding (SW3) now has its SAFE subset implemented: an
  opt-in driver (env `VT_ENABLE_JUMP_FORWARD`, default off) emits a grammar-forced
  token WITHOUT a model step ONLY where the grammar leaves exactly one valid token
  at a non-accepting state — provably byte-identical to normal per-token
  constrained decode (the constrained sampler has a single finite-logit token
  there), so no re-tokenization is needed. The general case (a forced byte run
  with several possible tokenizations, which SGLang handles by re-tokenizing and
  rolling back the boundary token) is deliberately not jumped, and wiring the
  driver into the live decode loop (jumped tokens need their KV recomputed) is a
  named residual — so the flag stays off by default. The radix eviction-policy
  knob (SW4) stays a named/deferred opt-in. See
  `.agents/specs/sglang-radixattention.md`.
- **SGLang-compatible knobs are now first-class, documented toggles** (2026-07-28,
  `CLAIM-SGLANG-ABI-DOCS`, reconciled to ABI v10). LPM scheduling and jump-forward
  are exposed on the C++ library API AND the C ABI (not env-var/internal-only),
  joining the already-exposed prefix-caching/RadixAttention and
  custom-logits-processor knobs. LPM is selected through the ABI v9 **string** field
  `vllm_model_params.scheduling_policy` (`"fcfs"`/`"priority"`/`"lpm"`); jump-forward
  through the new ABI **v10** tri-state field `vllm_model_params.enable_jump_forward`
  (`0`=default/`1`=on/`2`=off). The server has `--scheduling-policy lpm` and
  `--enable-jump-forward` / `--disable-jump-forward`. All default to today's
  behavior — an all-zero config is byte-identical to before ABI v10, and
  `vllm_abi_version()` returns 10. `VT_ENABLE_JUMP_FORWARD` is retained as an env
  override (wins when set). The user-facing enablement guide for all four behaviors
  is `docs/SGLANG-COMPAT.md` (engineering record: `.agents/specs/sglang-enablement.md`).
- **KV persistence to disk / CPU offload** is built (CPU and disk tiers,
  identity-checked blocks, a size-budgeted disk tier) and wired opt-in into the
  scheduler through an abstract `KVConnector` ABI selected by a
  `KVTransferConfig` (a disk-offload connector and the LMCache `lm://` client,
  over the same seam), off by default. The LMCache `lm://` client interoperates
  with a real vLLM+LMCache peer (its cache keys are byte-for-byte identical to
  LMCache's own `ChunkedTokenDatabase`, chunk 256, vLLM's `sha256_cbor`, and a KV
  prefix a real peer wrote is found and loaded back byte-identically over the
  wire), and it is now proven END TO END inside a real generation loop: with the
  connector ON against a live `lmcache.v1.server`, an OPT-125m run loads a
  previously stored prefix's KV and shortcuts prefill, generating tokens
  BIT-IDENTICAL to the connector-OFF cold run (verified both after an in-process
  restart and from a genuinely cold second process). The worker-side KV
  store/load is implemented **for the LMCache connector only**: the CPU/disk
  `OffloadingConnector` implements the scheduler half alone (it shortcuts prefill
  for matched blocks, but nothing moves their bytes into the KV pages), so the
  engine now REFUSES to wire it at construction, on every device, naming the
  connector and the device, rather than serving output computed over KV that was
  never written. Both connectors are selectable from the server with
  `--kv-transfer-config '<json>'`, mirroring vLLM's own flag and JSON shape. The
  disk connector's worker half remains unimplemented and is the named remaining
  step. Usage guide: [docs/KV-OFFLOAD.md](KV-OFFLOAD.md).
- **KV-cache events** (the stream external KV routers and prefix-cache-aware load
  balancers subscribe to) are built and off by default. When enabled, the
  prefix-cache block pool emits `BlockStored`, `BlockRemoved`, and
  `AllBlocksCleared` events at the store, evict, and reset sites, and they
  serialize to a `msgpack` payload that is byte-for-byte identical to vLLM's
  `msgspec` encoding (checked against captures for both the default
  int-truncated block-hash form and the raw-bytes form). Event generation and the
  wire payload go through an abstract publisher seam that mirrors vLLM's
  `--kv-events-config`. The live ZMQ socket transport (the actual publish to an
  external router, with its replay buffer and per-rank ports) is not wired yet,
  so today the events can be generated and encoded but not yet streamed over a
  socket; the seam is built so a real ZMQ publisher drops in without touching the
  emission sites or the encoder. Because it is off by default, the prefix-cache
  path is byte-identical to before.
- **Tool calling** covers the dialect set in the capability table above,
  including the Qwen3-Coder XML parser (`qwen3_coder` / `qwen3_xml` / `mimo`).
  Selection over HTTP is `--tool-call-parser` and `--reasoning-parser`, mirroring
  vLLM's own flag names, alongside the C ABI's
  `vllm_model_params.tool_parser` / `.reasoning_parser`. Both flags keep the
  server's previous behaviour as their default (`hermes`, reasoning off), take
  `auto` to opt into the same chat-template detection the C ABI uses, take `none`
  to disable, and abort startup listing every registered name if given an unknown
  one.
- **Engine configuration through the C ABI** reached parity with the bundled
  server's flags at **ABI v9** (2026-07-28). `vllm_model_params` now carries
  `max_num_batched_tokens` (chunked-prefill token budget), `scheduling_policy`
  (`fcfs`/`priority`/`lpm`), and `kv_transfer_config` (the external KV connector /
  LMCache JSON, connector name validated against `KVConnectorFactory` at load),
  and honours `tokenizer_config_path`, which had been declared since v1 and never
  read. Every addition is inert at its default, so a v8 caller that zero-fills the
  struct growth gets the byte-identical pre-v9 engine. Malformed
  `speculative_config` / `kv_transfer_config` documents report
  `VLLM_ERR_INVALID_ARGUMENT` rather than `VLLM_ERR_MODEL_LOAD` (the contract
  `vllm.h` has documented since v6). Driver: an embedder (the LocalAI vllm-cpp
  backend) could not expose LMCache or the prefill budget in a model config.
- **The stable C ABI contract is now spiked and current at ABI v10**
  (`SERVE-C-ABI`, `CLAIM-SERVE-C-ABI-SPIKE`). The accepted
  [contract](../.agents/specs/c-api-library.md) inventories ownership,
  no-throw/error mapping, concurrency, versioning and packaging gates. The row
  remains `ANCHOR-BACKFILL`: the next CPU-only bricks are all-19-symbol `dlsym`
  coverage, historical-layout compatibility fixtures, allocation-failure
  no-throw proof, and lifetime sanitizer stress.
- **DFlash speculative decoding from a GGUF DRAFT WORKS end to end on the GB10
  release target** (`SPEC-DFLASH-GGUF`, `DONE` for correctness on both axes,
  speed `PENDING` by design,
  [spike](../.agents/specs/gguf-dflash-draft.md)). A DFlash draft packaged as a
  single `dflash`-arch GGUF now loads AND generates against a safetensors target
  (axis A): config from the `dflash.*` KVs, weights through a resolver that
  reuses the existing safetensors `LoadQwen3DFlash` body, a `.gguf` branch in the
  draft-path resolution, and the target-shared embedding and lm_head still taken
  from the target, as the GGUF contract intends.
  **Binding result, MEASURED 2026-07-29 on a production-configured build**
  (CUTLASS + FlashAttention-2 + vendored Triton-AOT, sm_121a, proven three ways;
  earlier numbers came from a build carrying none of those and are void): on the
  Qwen3.6-27B NVFP4 safetensors target, a DFlash draft read from an UNQUANTIZED
  `BF16` GGUF is **the same draft as the z-lab safetensors one down to the
  count** - token-for-token identical greedy continuations AND exactly equal
  accept counts, 47/96 at 48 tokens (reproduced) and 27/64 and 15/144 at 24
  tokens on two prompts, k=16, concurrency 1. The published `Q4_K_M` draft emits
  the same tokens and costs at most ONE acceptance (46/112 against 47/96 at 48
  tokens, zero delta at 24).
  **That one acceptance is ordinary 4-bit cost, established rather than
  assumed.** It stood RED against an exact bar until 2026-07-29, and what closed
  it was finding out what the bar was measuring, not relaxing it. Loaded through
  the SAME code path the `BF16` GGUF is BYTE-IDENTICAL to the safetensors draft
  on all 58 tensors (a CPU gate, `tests/vllm/models/test_qwen3_dflash_gguf.cpp`,
  302/302, functionally RED against the `Q4_K_M` file); our Q4_K/Q6_K dequant is
  bit-equal to `gguf-py`'s reference on the real tensors; the published ladder's
  weight error is monotone and uniform (BF16 0, Q8_0 5.6e-3, Q6_K 1.85e-2, Q5_K
  3.85e-2, Q4_K_M 7.6e-2 mean relative, no outlier tensor); the only numeric
  config difference is `rms_norm_eps` at 2.5e-9 relative. The end-to-end run then
  closed it: restoring the draft's precision and nothing else moves the count
  back to exactly 47/96. So the accept-count bar is now EXACT across formats and
  BANDED across quantization levels (`|d_accepted| <= 2`, `|d_proposed| <= k*2`,
  both derived from the measurement and mutation-proved non-vacuous at band 0),
  with the arm selected by reading the draft file's ggml types rather than by a
  flag.
  Three conventions this had to undo, all invisible to shape and name checks:
  `dflash.target_layers` is stored `+1`-offset; the draft's RMSNorm weights are
  RAW (its converter class does not inherit the Qwen3Next `+1` shift, so unlike
  the trunk and the MTP head they must NOT be un-shifted, which points the
  opposite way); and `vocab_size`, which the GGUF contract deliberately omits,
  must be back-filled from the target or the draft's shared embedding table is
  empty and the first proposal throws. Only the last survived the load-level
  tests, and only generating found it.
  **Axis B (the TARGET is a GGUF too) now WORKS as well.** `SharedHeadSource`
  replaces the safetensors-typed parameter that used to make the draft's shared
  embedding and lm_head unavailable from a GGUF - which is why the GGUF path
  refused dflash outright - and serves both tensors out of either container. On
  the Qwen3.6-27B NVFP4 **GGUF** target with the same `Q4_K_M` GGUF draft, the
  DFlash-ON continuation is token-for-token identical to that same target's
  spec-OFF and the drafter is alive at 14/160 accepted.
  Re-measured 2026-07-29 on the production build and BROADENED from one prompt to
  three, strict-form green on all of them: "The capital of France is" IDENTICAL
  with acceptance 14/160, "Write a Python function that reverses a string:"
  IDENTICAL with 24/64, "Photosynthesis is the process by which" IDENTICAL with
  15/128, all exit 0.
  Two things are worth stating plainly rather than rounding off. The spike's
  highest-ranked risk - the draft scoring with a head dequantized out of a
  quantized target - does not exist on this asset: the GGUF stores
  `token_embd`/`output` as BF16 beside its NVFP4 body, byte-identical to the
  safetensors sibling of the same quantization run, so the read is verbatim. And
  acceptance against the safetensors target is PROMPT-DEPENDENT rather than
  uniformly lower: 14/160 against 15/144 on the first prompt, but EQUAL at 24/64
  with IDENTICAL DFlash-ON streams on the second. That belongs to the GGUF
  target's compute path, not to the shared head, because there is no NVFP4 GGUF
  GEMM yet, so the GGUF target expands to bf16 while the safetensors target runs
  the true W4A4 kernels; the two containers' own spec-OFF streams diverge at
  index 4 on the first prompt and index 16 on the second, with no speculation
  anywhere, and the DFlash-ON pair tracks that.
  Operational note for anyone re-running the axis-B arm: it peaks at ~81 GiB RSS
  against GB10's 119 GiB unified pool, so a run launched on top of a large page
  cache can fail with `vt cuda: embedding: out of memory`. That is contention,
  not a result; on a settled box the arm reproduces its numbers exactly.
  **Both axes are now gated and closed** (gates 1-5 and 7 of the spike), so the
  row moved `PARTIAL` -> `DONE` for correctness in closing commit `c62f2fa3`.
  What remains open is deliberately NOT this row's: no speed A/B is owed until a
  native NVFP4 GGUF GEMM exists to make the two target containers comparable, and
  `docs/BENCHMARKS.md` carries that as an explicit `PENDING` with its
  reproduction command.
- **MTP speculative decoding from a GGUF target WORKS, on CPU and on the GB10
  release target** (`SPEC-MTP-GGUF`, `DONE`,
  [spike](../.agents/specs/gguf-mtp-spec-decode.md)). The head loads from a
  head-carrying GGUF - `HfConfigFromGguf` republishes the depth it already read
  from `<arch>.nextn_predict_layers`, and `LoadQwen3_5MTPFromGguf` reads the
  `nextn` block with the trunk loader's own helpers, so it inherits the GGUF
  (w+1) norm storage, the quantization routing and the torch [N, K] shape order.
  A GGUF exported without the head is refused, naming that as the reason.
  **Both correctness gates pass: spec-ON output is token-identical to spec-OFF**
  on CPU against a real llama.cpp-converted Qwen3.5-2B, and on the GPU against
  the 35B A3B NVFP4 GGUF with 13 drafts proposed / 11 accepted. **RE-ANCHORED
  2026-07-29** to a production-configured build (CUTLASS + FlashAttention-2 +
  vendored Triton-AOT, sm_121a; the original close-out ran without any of them,
  so its evidence was not trustworthy): dgx GB10 under `flock`, 3/3 cases, 10/10
  assertions, exit 0, spec-ON token-identical to spec-OFF, **13 proposed / 11
  accepted, unchanged**, 90.26 GiB peak RSS, 7m13.59s. Acceptance parity is measured
  against the safetensors sibling of the same quantization run: 12 proposed / 11
  accepted there vs 13 / 11 here. Cross-format token agreement is NOT
  APPLICABLE - it needs an F16/F32 GGUF sibling and every head-carrying export
  on hand is quantized. Throughput is the one open item and is deliberately
  PENDING, see [docs/BENCHMARKS.md](BENCHMARKS.md). Closing commit `edf91449`.
  Two things had held this row open and both are now settled by measurement.
  The build was believed to be `CMAKE_CUDA_ARCHITECTURES=75` on an sm_121 GB10:
  that came from `CMakeCache.txt`, which records the CUDA compiler's own probe
  default and is shadowed by the normal variable the project sets, so it never
  described the compiled code. The real flags and every emitted cubin are
  `sm_121a`. And the GPU arm's 24 tokens differed from the CPU arm's: that was
  read as a near-tie, and the 2026-07-29 re-measurement shows it was an artifact
  of the defective build. **There is no CPU-versus-GPU token delta on a
  production build**: both devices emit the same 24 tokens, because with CUTLASS
  and the Triton-AOT GDN kernels in, the GPU takes the CPU's side of the 0.06-nat
  near-tie at position 1 (GPU rank1 `11` -0.763897 over rank2 `13` -0.824083; CPU
  rank1 `11` -0.765499 over rank2 `13` -0.830374, bit-identical to before, as
  expected for CUDA-only kernels). Zero exact ties in either arm, minimum margins
  0.060 and 0.065 nats. The near-tie is real and unchanged in size; only which
  side the GPU landed on moved, and the earlier cascade narrative is retracted.
- **The 35B A3B NVFP4 SAFETENSORS target has EXACT logit ties, and its
  speculative token identity does not hold there** (open, surfaced by
  `SPEC-MTP-GGUF`'s gate, owned by `SPEC-MTP`). Pointed at the safetensors
  sibling of the same quantization run, the same concurrency-1 gate shows
  spec-ON inserting one token relative to spec-OFF, and two runs of spec-OFF
  itself disagreeing. Both divergences land exactly on positions where the top
  candidates carry BIT-IDENTICAL logprobs: that arm has three such exact ties in
  24 tokens, because keeping NVFP4 packed puts its logits on a 1/16 grid where
  distinct tokens collide. The GGUF arm, which expands to bf16, has ZERO exact
  ties over the same 24 tokens (minimum margin 0.048 nats) and reproduces its
  sequence every run - so the GGUF result stands and this is a quantized-GEMM
  determinism question, not a speculative-decoding logic one. Not root-caused
  beyond the tie measurement (one prompt, two runs).
- **Speculative decoding on CPU corrupted the target's own state, and is FIXED**
  (`CPU-SPEC-DIVERGENCE`). `qwen3_5.cpp:3616` sized the GDN state gather/scatter
  row by `(Kw-1)` while the speculative persistent row is `(Kw-1)+num_spec`, so
  the contiguous row helpers mis-strode the cache slot and every channel past the
  first, corrupting post-prefill recurrent state; speculation therefore changed
  the target's own greedy output even when no draft was accepted. Fixed with a
  stride-aware per-channel state copy used only when the two widths differ, so
  every non-speculative path stays byte-identical. CPU-only in effect - the
  fp16/bf16 cache arm routes through the `GdnStateGather`/`Scatter` ops, so CUDA
  was never exposed and no GPU result is affected or retracted. It survived
  because there was no CPU spec-decode gate anywhere in the tree; there is one
  now.
- `/health` reports process liveness rather than a full engine-health probe.
- **Speculative decoding** ships user-facing via `--speculative-config` (OpenAI
  server, example CLI, and C ABI v6), unset by default and byte-identical to the
  non-speculative engine when unset. **MTP (k=1)** on the Qwen3.5/3.6 gate
  checkpoints (those shipping an `mtp.*` head) is token-exact to both our own
  spec-off output and vLLM's own MTP greedy at concurrency 1, and about 1.04x
  faster than vLLM's spec-on decode; the concurrent (multi-request) path shares a
  scheduler step with ordinary prefills over a bit-exact GDN split-merge. Honest
  caveat: above concurrency 1 the 27B greedy output is not bit-stable across
  batch shapes, so exact spec-on/spec-off token agreement is a concurrency-1
  property. **Block-diffusion DFlash** (method `dflash`, the z-lab Qwen3.6-27B
  draft over the 27B NVFP4 target, wired through the same `--speculative-config`
  with an external draft `model` path) is correctness-complete (ratified
  near-tie) and at or above vLLM throughput at concurrency 1 (29.32 vs 29.24
  tok/s, 1.003x), built D0 through D14 on the vLLM 0.26.0.dev0 stack (which
  resolves vllm#40898) and recorded DONE across the engine, model and kernel
  matrices. **ngram** (method `ngram`) is the draft-free suffix-match proposer,
  reusing the same verify/reject loop with no draft model; it is token-exact vs
  vLLM's own ngram on repetitive workloads (27B, 5/5 prompts, every draft
  accepted) and off by default (byte-identical). Measured A/Bs and the D0-D14
  chronology: [docs/BENCHMARKS.md](BENCHMARKS.md); usage guide:
  [docs/SPECULATIVE-DECODING.md](SPECULATIVE-DECODING.md).
- **LoRA and multi-GPU** are not supported yet. **Multimodal** is
  correctness-complete on a single-sequence eager path and the top roadmap
  priority; the OpenAI-server wiring has advanced through its CPU bricks —
  brick 1 (`CLAIM-MM-SERVING-W1`) parses the OpenAI multimodal content-part array
  and routes the decoded media through the existing processors, and brick 2
  (`CLAIM-MM-SERVING-W2`) carries the result into the engine (additive
  `add_request(MultiModalInputs)`/`generate` overloads on both engines via
  `InputProcessor::process_inputs_mm`, chat-template placeholder-string helpers,
  and a default-inert serving_chat multimodal seam), but the closing end-to-end
  GPU gate (`MM-SERVE-E2E` — the model tokenizer/processor seam body plus the mm
  forward) is still pending, so it is not yet servable end-to-end. Image-to-text is strict
  token-exact 32/32 vs the vLLM 0.25.0 oracle on both Qwen3-VL-4B and our own
  gate model Qwen3.6-27B (`Qwen3_5ForConditionalGeneration`); video-to-text works
  end-to-end on both (Qwen3-VL-4B near-tie-robust, Qwen3.6-27B strict 32/32); and
  audio-to-text works end-to-end on Voxtral-Mini-3B (near-tie-robust, decoder
  proven token-exact 48/48), the first audio understanding in the tree. The
  pipeline is a pure C++ port end-to-end: the Qwen3-VL image/video processors and
  the Whisper-class audio processor are bit/byte-identical to the oracle, the
  Qwen3-VL vision tower and the Whisper encoder tower are proven faithful within
  the bf16 envelope, and the MRoPE/DeepStack text-backbone contracts are
  unit-exact. Concurrency-1 speed is measured against vLLM 0.25.0's graphed
  config: on Qwen3.6-27B image the decode is at parity and the vision tower,
  after a warp-scoped online-softmax attention kernel plus a resident-weight
  load, is 148 ms/image (0.59x of vLLM's eager encode, faster) with the strict
  image/video gates still 32/32. On Voxtral audio the decode is now graph-captured
  and measured at 60.9 ms per token (vs 61.7 ms eager, a small real
  non-overlapping win), against vLLM's graphed 40.8 ms, so the audio gap is about
  1.49x: the removable per-step launch overhead turned out to be only about 1.25%
  of the per-token time, so the residual is per-step compute (kernel) efficiency,
  not launch overhead. That residual is now fully attributed: nsys shows all of it
  is the naive scalar paged-attention decode kernel (723 us per call over 30
  layers, about 120x the KV memory floor). vLLM runs decode attention on
  flash-attention varlen, which our tree already ships and enables by default for
  this head shape; Voxtral only missed it because the decode driver allocates one
  KV block of size 444, not a multiple of 16, which disqualifies the fast path.
  Rounding that to a multiple of 16 routes decode through the flash kernel and
  reaches 38.2 ms per token, a 36% win that beats vLLM's 40.8 ms, and the
  resulting sequence teacher-forces vLLM with zero divergences (a valid greedy
  branch). It is a bf16 near-tie ceiling: it changes which side of an exact tie is
  taken, so it no longer reproduces the committed near-tie golden byte-for-byte
  and is not adopted here; the byte-exact scalar path is kept at 14/14 with the
  golden unchanged, and the win is one line plus a golden regeneration away.
  On-GPU greedy sampling is in place on both mm decode loops. The open work is
  batched/graphed mm serving, adopting the audio decode win, and higher-concurrency
  throughput. Design and status:
  [.agents/specs/multimodal-track.md](../.agents/specs/multimodal-track.md),
  [.agents/specs/audio-track.md](../.agents/specs/audio-track.md), and
  [.agents/specs/multimodal-speed.md](../.agents/specs/multimodal-speed.md).
- **Environment variables.** A handful of behavior-changing knobs (CPU thread
  count, prefix-cache hash seed, LMCache host/port, the default-on async and CUDA
  fast paths and their rollbacks, the portable-reference bisect switch, GGUF
  loading behavior, the Vulkan device selector) are documented in
  [docs/ENVIRONMENT.md](ENVIRONMENT.md). A CI check
  (`scripts/check-env-doc.py`) keeps that reference honest: any new production env
  var must be documented or explicitly classified kernel-internal.

## Verification and parity

Every model is gated token-for-token against the vLLM 0.25.0 oracle (the
gate-time pin, re-validated bit-identical on the current 0.26.0.dev0 pin) on the
same workload, and every change that could affect correctness or performance is
compared apples-to-apples against vLLM with both numbers and the ratio recorded.
Behavioral CPU tests run under CTest; CUDA correctness, sanitizer, trace, and
performance evidence is recorded per feature rather than inferred from source.
The active protocol is in the [verification procedure](../.agents/verification.md). A CI check
(`scripts/check-fusion-consistency.py`) additionally keeps model forwards routing
their fusable add+RMSNorm glue through the portable fusion catalog rather than
hand-fusing it. A sibling CI check
(`scripts/check-runner-routing-consistency.py`, wired into the same `agent-record`
job, 2026-08-02) enforces the THIRD "MUST route through" seam — the decode/runtime
path — over every `REGISTER_VLLM_MODEL` with THREE sub-checks: **(a)** the default
`gather_logits` forward returns device-resident logits
(`ForwardLogits.on_device()==true`, off the on-GPU sampler), **(b)** no private
`*GenerateCore` host generate loop off the runner, and **(c) bf16-resident
activations** (added 2026-08-03) — the decode keeps its residual/activation stream in
bf16 **device** `DBuf`s (the shared `dense_attn::AttnBlock` glue), NOT a hand-rolled
private `std::vector<float>` host residual stream cast to bf16 before every projection
(`CastBf16`/`GemmBf16`); a model that declares >=`MIN_F32_RESID`(=3) private f32
residual-stream buffers AND routes through neither the shared bf16 attn preamble nor a
bf16-`DBuf` residual is flagged **F32_STREAM** (drift). It resolves the `.forward` hook
through its `SomeModel::ForwardDevice` delegate and `using`-aliases (so a HOST-stub
`ForwardDevice` is caught while a `VT_CHECK(false)` refuse stub is skipped) and
classifies all 27 registrations green: **on (a)/(b)** 24 device-resident, 2 off-runner
on `scripts/runner-routing-allowlist.txt` (`laguna`, `qwen3_vl`), 1 refuse stub;
**on (c)** 24 bf16-resident, 2 f32-stream on the SIBLING
`scripts/runner-bf16-activation-allowlist.txt` (`laguna`, `deepseek_v4`), 1 refuse stub.
The two axes are ORTHOGONAL and carry SEPARATE allowlists because their membership
differs: `deepseek_v4` returns device-resident logits (clean on (a)) yet computes its
whole decode in an f32 host vector stream (drift on (c)); `qwen3_vl` is HOST on (a)/(b)
yet its per-token transformer decode residual is a bf16 `DBuf` (clean on (c) — its text
decode is already routed onto DBufs). A shared flat list would silently cross-suppress a
regression on the other axis. The gate would fail a NEW model returning `HostLogits`, or
a NEW model hand-rolling an f32 host stream, without an allowlist entry; removing an
(a)/(b) entry after the model returns device-resident logits, or a (c) entry after its
decode residual routes onto the shared bf16 `AttnBlock`/`DBuf` glue, is the gate closing.
**The Gemma family (Gemma-1/2/3) was MIGRATED off the drift
allowlist 2026-07-30 (Tier-B2 of the cross-arch fold plan):** their
residual-carrying gemma-(1+w) input/pre-ffn/final norms now route through
`vt::FusedChain(kFusedAddRmsNorm)` behind `FusedChainAdoptEnabled()` (byte-exact
`else` fallback), and the sandwich post-norms (no residual add) correctly stay
standalone — so the allowlist shrinks to `glm4`/`phi3`. In the SAME wave (Tier-C1)
all four text Gemma MLPs fold onto a new shared GeGLU gate-up method
(`layers::UnquantizedMlpGateUpGeluMethod`, the GeluAndMul sibling of the SwiGLU
`UnquantizedMlpGateUpMethod` A1 landed) — one merged `[2I,H]` descriptor family now
serves BOTH activation families. Gemma is the strongest reuse proof because it has
REAL committed goldens: see docs/BENCHMARKS.md for the DGX SACRED-gate results.

**The 27B SACRED gate needs the production kernel build, and says so.** The 27B
W4A4 gate (`tests/parity/test_qwen27_paged_engine.cpp`) reproduces the oracle's
production continuation only with BOTH the CUTLASS sm120a NVFP4 fp4xfp4 GEMM
(`-DVLLM_CPP_CUTLASS_DIR=<cutlass 4.5.0+>`) and the vendored Triton-AOT GDN
kernels; without either, the engine falls to the emulation-grade fp4 GEMM / hand
GDN recurrence, takes the other side of the tok6 whitespace near-tie and emits
`greedy_ids_emulation.npy`. Measured 2026-07-29 on GB10 sm_121a from ONE tree at
`d4492c03`: CUTLASS off + Triton off 234/235, CUTLASS on + Triton off 233/235,
both on **235/235**. The gate refuses to run in a build missing either and names
the configuration; Triton is now default-ON (#219). Status UNCHANGED and GREEN.

### Feature-gap map vs pinned vLLM 0.26 (2026-07-28)

A whole-surface sweep of pinned vLLM `555967922` (0.26.0.dev0) against our
matrices ranked what vllm.cpp is MISSING: 8 HIGH, ~19 MED, ~16 LOW gaps, each
grounded in vLLM `file:line`. The material HIGH-priority misses (common,
single-box, user-facing) are LoRA / multi-LoRA runtime, the
pooling/embedding/classify/rerank task class, AWQ + GPTQ native quantized
compute, the xgrammar structured-output backend, fp8 KV cache, and reasoning
parsers. One medium gap has no tracked row yet (generic draft-model + Medusa
speculative decode); the offline Batch API (`SERVE-BATCH-API`) and the plugin
system (`ENG-PLUGIN-SYSTEM`) were picked up 2026-07-29. vLLM has
removed prompt adapters, so that is not a gap. The full prioritized list lives
in [`.agents/specs/vllm-feature-gap-analysis.md`](../.agents/specs/vllm-feature-gap-analysis.md);
this is analysis only, no capability state changed.

The CPU CTest suite is green (0 real regressions). A 2026-07-27 hygiene pass
triaged the previously-red tests as stale assertions or `-j` contention, not
regressions: the model-registry count assertion tracks the 24 registered
architectures, the GGUF loader rejection case uses a genuinely-unregistered arch,
the DFlash proposer test supplies the now-required `"model"` key, and the two
HTTP-server tests run under CTest `RUN_SERIAL` so a CPU-hog concurrent test can no
longer starve their accept thread.

**env-doc hygiene (2026-07-29):** allowlisted `VT_SPEC_TRACE` (a per-block
spec-decode acceptance *trace*, off by default, output-neutral — `runner.cpp`
`spec_trace`) on `scripts/env-doc-allowlist.txt`; introduced by the concurrent
DFlash-GGUF session and left `check-env-doc` RED on main. No behavior change.

**fusion-consistency gate-test repair (2026-07-30):** VOID, no behavior change and
no teeth weakened. `tests/scripts/test_check_fusion_consistency.py`'s mutation test
hardcoded `gemma2` as its in-tree known-drift example, so the Tier-B2 fold broke it
by doing what the gate wants (Gemma adopted `vt::FusedChain`, its allowlist entries
were retired, and the assertion went looking for a model that no longer drifts). It
was RED on clean main before this change. The example is now DERIVED from the
allowlist: emptying the allowlist must expose every in-tree hand-fusing model, and
the allowlist must suppress at least one model the detector really sees, so the next
migration closes the gate instead of breaking the test.
`scripts/check-fusion-consistency.py` itself is unchanged; `tests/scripts` 87/87.

**Landing-page restructure + project logo (2026-07-30):** DOCUMENTATION-ONLY, no
capability state moved and no number changed. The thesis is now stated as the
project's actual ambition (smallest, fastest, every feature people want, with vLLM
as the PROOF rather than the identity) instead of "a C++ port of vLLM", the
llama.cpp and MLX-LM comparisons carry their real per-axis numbers alongside the
vLLM grid, and a Credits section attributes vLLM / SGLang / llama.cpp / MLX / the
kernel upstreams and records that this tree does NOT use ggml (own portable `vt::`
runtime, GGUF read natively) without disparaging it. **Two accuracy fixes landed with
it.** (1) The footprint claim was STALE: the README said "9.4 GiB venv vs one 10 MiB
binary" while the committed spec (`benchmarks/demo/footprint_gb10.json`, measured
2026-07-28 on the same GB10) and the rendered figure both say **9318.6 MiB (9.1 GiB)
vs a 65.78 MiB stripped CUDA server binary**, i.e. ~140x, not ~960x; every instance is
now the measured number. (2) The concurrency ratios are now read against the gated
**0.5% non-tail run-to-run noise band**: c2-c32 (0.7%-1.7%) are presented as a TIE
with the c1 4.5% as the one win outside noise, llama.cpp decode 0.97x as parity inside
its own spread, and the Metal 2.4% as REAL and outside both spreads (ours 0.12%,
MLX-LM's 0.34% over 6 interleaved runs) rather than waved off as noise. README.md becomes a landing page
(logo lockup, scoped badges, the six-axis concurrency grid, the footprint figure,
and a features block); its reference tables move to the new
[docs/USAGE.md](USAGE.md) (CLI, endpoints, server flags, C ABI, C++) and
[docs/BUILD.md](BUILD.md) (per-backend recipes, every CMake option, hardware and
quantization state), and the 26-row architecture matrix plus the full feature list
become `<details>`. The sections `scripts/check-readme-structure.py` requires are
retained as compact linking sections, so the AGENTS.md landing-page contract holds
with the tables elsewhere (README 23k -> 19.1k chars against the 30k budget). The
logo ("Steps": the V1 scheduler timeline, bars crossing step boundaries) is
rendered by `scripts/make-logo.py` from one source-of-truth geometry, reusing the
`benchmarks/demo/concurrency_race.py` palette. Honest-numbers rule held: the Metal
row states prefill 1.5% ahead of MLX-LM with warm total 97.6% MLX-gated / 95.9%
default and the 18-of-75-ops caveat, and the performance badge is scoped to the
Qwen3.6-27B gate model rather than implying every architecture beats vLLM.

## Upstream sync 2026-07-30 (`555967922..e04a30a77`, 198 commits)

Bounded mechanical sync cycle. Enumerated + tier-classified the 198-commit delta
(ranked queue: [.agents/specs/upstream-sync-2026-07-30.md](../.agents/specs/upstream-sync-2026-07-30.md);
report: [.agents/sync/2026-07-30-e04a30a.md](../.agents/sync/2026-07-30-e04a30a.md)).
**Ported (T1, CPU-gated):** vllm#49754 per-request `stream_interval` sampling
param, mirrored into `SamplingParams`, the output-processor clamp, and both
OpenAI request paths (only raises the engine `--stream-interval`, clamped up;
first/final chunks always emitted). Gates: `test_output_processor` 11/11,
`test_sampling_params` 9/9, `test_openai_protocol` 28/28 (CPU). **Parity pin
UNCHANGED** at `555967922` (subset landed); the T1 correctness queue (48245
preemption-underflow, 50297 P/D-race, 49343 eagle-draft) and the T0 RMSNorm
perf/determinism pair (49750, 48391 — GPU runtime gate) are the ranked
carry-over. vllm#49030 (video temporal padding) was found ALREADY-CORRECT in
`qwen3vl_processor.cpp` (our ceil formula equals the fix); no port.

**Increment 2 (2026-07-30, T1-correctness triage).** Ported the portable core of
rank 3, **vllm#49343** (EAGLE draft `max_position_embeddings` clamp): the pure
config helper `SpeculativeConfig::MaybeOverrideDraftMaxPositionEmbeddings`
(`include/vllm/config/speculative.h`) — raise an eagle/eagle3 draft's
`max_position_embeddings` up to the target's `max_model_len` (never lower;
prevents the draft rotary `cos_sin_cache` OOB gather our `RotaryEmbeddingBase`
would take). Gate: `test_speculative_draft_max_position_embeddings` **4/4** (CPU,
RED-first verified — the two raise/override-flag assertions fail with the clamp
neutered). We do not yet LOAD eagle/eagle3 drafts (draft-config resolution is
deferred, EAGLE3 is T2), so there is no live call site yet — this is the tested
clamp the eagle loader wires in when it lands; the 2 upstream model-integration
tests (real `ModelConfig(EAGLE3_DRAFT/AR_MODEL)`) are SKIPPED (need HF-download +
draft ModelConfig machinery), recorded in porting-inventory.
Ranks **1-2 SKIPPED** as unported-code-path, not force-fitted: **vllm#48245**
(preemption `num_output_placeholders` underflow) and **vllm#50297** (P/D
preemption race) both rebuild the async stale-output machinery around
`num_in_flight_tokens` / `num_stale_output_tokens` / `drop_stale_output` + the
`reset_prefix_cache(reset_running)` scheduler force-preempt + a `skipped_waiting`
queue + (50297) the KV-connector producer/consumer `requires_kv_delivery` role —
NONE of which this tree carries (our `async_tokens_to_discard` is never set, the
force-preempt path is unported, the async engine has no PP concurrent-batch
window, and KV P/D transport is N-A by sync policy). The underflow they fix
cannot occur here; porting the prerequisite machinery is a scoped feature port,
requeued in the sync doc. Parity pin still UNCHANGED at `555967922`.

**Increment 3 (2026-07-30, quant/config triage — verified SKIP-all).** Worked the
quant-loader + config-validation slice of the ranked queue (ranks 6-8). All three
are SKIPPED as unported-code-path after source verification against the real
upstream diffs — not force-fitted, no invented code. **vllm#49483** (compressed-
tensors: prioritize fused-name match over class match in `find_matched_target`)
reorders two matching strategies inside `find_matched_target`
(`vllm/model_executor/layers/quantization/compressed_tensors/utils.py:145-150`); our
tree has NO such generalized matcher — compressed-tensors schemes are resolved
per-projection by direct tensor-name probing (`IsCtNvfp4Projection` /
`LoadCtNvfp4W4A16` in `include/vllm/model_executor/models/dense_weight_loaders.h`
check `has(proj + ".weight_packed")` and honor the config `ignore` list inline),
so there is no targets-list walk, no class-name substring match, and no
fused-vs-class ordering to fix. The bug is structurally impossible here.
**vllm#48589** (INC `extra_config` layer-name suffix match) adds an `endswith`
fallback to the Intel-Neural-Compressor config parser
(`vllm/model_executor/layers/quantization/inc/config_parser.py`); we carry no INC
quant path (Intel/XPU, N-A bucket) — our only `extra_config` is the KV-connector's,
an unrelated map. **vllm#49134** (reject contradictory custom-op directives)
validates `CompilationConfig.custom_ops` (`+op`/`-op`/`all`/`none`), a
torch.compile/Inductor plumbing config we do not carry (our fusion catalog is a
portable `vt::FusedChain` applied by declaration, not a runtime custom-op
enable/disable list). The T0 RMSNorm pair (49750 uncontiguous-residual perf, 48391
batch-invariance) stays QUEUED, GPU-gated: our `RmsNormRowKernel`
(`src/vt/cuda/cuda_ops.cu:62`) indexes contiguously (`row*h`, no `input_stride`),
so 49750's residual-stride relaxation is a multi-backend structural change with no
current strided-residual caller, and 48391 depends on the `vllm_is_batch_invariant`
determinism subsystem we do not carry — neither is a clean 1:1 mirror. No code
landed; the queue + records were refreshed. Parity pin UNCHANGED at `555967922`.

**Best-GEMM-path research (2026-07-30):** `.agents/specs/best-gemm-path-2026-07-30.md` — best-in-class GEMM path per regime (fp8/full-tensor/low-bit × decode/prefill) across vLLM+SGLang+ds4+llama.cpp + FusedChain-mapping verdict. Two actionable findings: (1) **fp16 dense inputs THROW** (`cuda_matmul.cu:220-227`) — a breadth gap, one-branch cuBLASLt fix; (2) **nvfp4 W4A4 M=1 decode is mis-routed to the prefill tensor-core tactic on our production 27B/35B gate models** — a real decode perf hole, dp4a-matvec fix. `Fp4GemmSm120` confirmed LIVE (not dormant). No behavior change (research/spec only).

**nvfp4 W4A4 M=1 decode "mis-route" — MEASURED NON-ISSUE (2026-07-30, best-GEMM-path rank-2 CLOSED).**
DGX-measured (Iron Law): at M=1 the cutlass autotuner ALREADY selects swap-AB small-M tactics, and
the weight-dominant 27B W4A4 projections run FLAT across M at the HBM copy roofline (gate_up 258
GB/s, down 283 GB/s) — decode is memory-bound, so a dp4a matvec reads the same bytes at the same
bandwidth and cannot raise tok/s. NO production code changed; SACRED `test_ops_nvfp4_fp4` 30/30·27006.
The gate models' W4A4 decode is at the achievable roofline. See `.agents/specs/nvfp4-w4a4-m1-decode-measured-2026-07-30.md`.

**DeepSeek-V4-Flash Brick 6 — MoE fusion, +5.2% decode (2026-07-30, `6795717f`, MERGED).** Corrects the Brick-4 "bit-exact GEMM levers exhausted" framing: that was the GEMM *mainloop*; the CROSS-kernel fusion lever (ds4's real edge) was still open. Fused routed-MoE gate+up+silu into ONE kernel (mirrors ds4 `moe_gate_up_mid`) — **BIT-EXACT** (byte-identical 49-token sequence, real 80.7 GB model), decode **10.27 → 10.80 tok/s (~65% of ds4 16.5)**, kernel instances −12.2%. `VT_V4_FUSED_MOE=0` rollback. NEXT (Brick 7): fuse the un-fused per-step glue (Rope 119.9ms + MhcPreFinish 98.7ms + Route 77.4ms) — the LARGE remaining ds4 gap. Row stays `ACTIVE`.

**DeepSeek-V4-Flash Brick 7 — fused per-head RMSNorm+RoPE + parallelized RoPE tail, +5.5% decode (2026-07-30, branch `brick7-fused-rope`, commit `<this>`, NOT pushed).** Attacks Brick 6's top glue (`RopeKernel`). New `NormRopeRowsKernel` (ds4 `head_rms_norm_rope_tail_kernel`/`dsv4_qkv_rms_norm_rows_kv_rope_kernel`) folds the {`rms_norm_rows`;`rope`} launch pair into ONE block-per-row kernel and splits the RoPE pairs across threads — **BIT-EXACT** (the `theta *= theta_scale` recurrence is a left-fold, so pair p via p sequential mults reproduces the exact double product; unit case `norm_rope_rows == split` BYTE-IDENTICAL, real 80.7 GB model `VT_V4_FUSED_ROPE=1` vs `=0` byte-identical 49-token sequence). Wired into BOTH the eager forward AND the captured `V4Graph::Step` (the graph path was a separate code path — the first eager-only build measured FLAT, an RED-first catch). Decode **10.82 → 11.41 tok/s (+5.5%, non-overlapping bands, ~69% of ds4 16.5)**; nsys: RoPE glue GPU time 129.1 → 63.9 ms (**−50.5%**) + ~1204 rms_norm launches folded; PEAK RSS 86.68 GiB unchanged. `VT_V4_FUSED_ROPE=0` rollback. NEXT: activation-quant fusion into the GEMM prologue (`QuantizeQ8K` launch-bound, now the top glue at 7.6%) + MhcPre reduction glue. Row stays `ACTIVE`.

**Cross-arch merged-GEMM fold plan (2026-07-30, research-only, `.agents/specs/arch-fusion-fold-plan-2026-07-30.md`).** Read-only survey of ALL implemented forwards (~60 fold sites) for the merged-GEMM + fused-glue patterns, ranking how to fold every arch onto ONE shared descriptor family + the FusedChain glue catalog — the "use it consistently" rollout so DeepSeek's fused-kernel wins stop being DeepSeek-only. **Headline: the merged-GEMM weight layouts already exist almost everywhere; the gap is launch/epilogue granularity + seam routing, not weight marshaling — most high-value folds inherit an already-tuned kernel for ZERO new kernel code.** Tier A (bit-exact, free-kernel, wide reuse) first: A1 bf16-SwiGLU-MLP seam (5 archs: OLMo-2/Granite/StableLM/dflash/deepseek_v2), A2 MLA A-proj 3→1 merge, A3 keep-quant grouped-MoE → `kMatmulBTQuantGrouped`, A4 bf16 grouped-MoE → promote ds4 `moe_gate_up_swiglu` to shared `kMoeGateUpSwiGLU`, A5 MLA norm-rope → `kFusedNormRope`. Feeds the generalization front (shared ops + descriptor). End state: a new dense/MoE arch is "born fused" (kimi_k3 the proof — inherits A2/A4/A5 with zero model-specific work). No code changed (research/spec only).

**Keep-quant fused MoE → SHARED op + declarative merged-GEMM descriptor (2026-07-30, `bb864b0d`, MERGED).** The extensibility answer to "do other archs gain from what we do?" — the DeepSeek-V4 Brick-6 fused routed-MoE **gate+up+SwiGLU** kernel, previously an arch-private `MoeDeviceKernels::moe_gate_up_swiglu` function pointer, is PROMOTED to a first-class shared `vt::` op `OpId::kMoeGateUpSwiGLUGrouped`, MIRRORING `kMatmulBTQuantGrouped` (enum + typedef + `vt::MoeGateUpSwiGLUGrouped` host shim + CUDA registration of the **SAME** kernel via an append-only registrar [BIT-IDENTICAL move] + a CPU composite golden = two `MatmulBTQuantGrouped` + clamped-SwiGLU). Plus the DECLARATIVE DESCRIPTOR `vt::MergedGemmGroup` (`include/vt/merged_gemm.h`) — the contraction-tier analog of `FusedRecipe`: N GEMMs SHARING operand A + a fused epilogue opcode, selected by (dtype, arity, epilogue), realized end-to-end by `kKeepQuantGateUpSwiGLU` with the byte-exact Tier-0 composite (`arity`× `MatmulBTQuantGrouped` + epilogue) as the portable fallback — same tiering as `fused_recipe.h`. Any keep-quant MoE arch now inherits the tuned single-launch kernel by calling ONE op; the inheritable win is the MEASURED Brick-6 **+5.2% decode** (`6795717f`). CPU gate `test_deepseek_v4_gguf_load` **15/15·931** (2 new RED-first cases: fused == composite byte-identical; descriptor fast==Tier-0 byte-identical). DGX: DeepSeek CUDA regressions GREEN + bit-exact (promotion is a bit-identical move). `deepseek_v4.cpp` UNTOUCHED (Front B owns it for Brick 8). **NAMED RESIDUAL:** a fresh second-model e2e decode-tok/s A/B is BLOCKED — DeepSeek is the only in-tree keep-quant grouped-MoE model (qwen3_5/Coder MoE are fp4/bf16), there is no K-quant MoE checkpoint on the box (root 96% full) and no host K-quant encoder to synthesize one, so the win is proven at the kernel level (shared op + Brick-6 measurement) + framework level (descriptor byte-exact across tiers). Spec `.agents/specs/keepquant-shared-ops-2026-07-30.md`.
**DeepSeek-V4-Flash Brick 8 — FUSE QuantizeQ8K into the grouped-GEMM prologue: BIT-EXACT but a MEASURED NEGATIVE (−22%); hypothesis REFUTED, code REVERTED (2026-07-30, branch `brick8-fused-quant`, records-only, NOT pushed).** Brick 7 named `QuantizeQ8K` (7.6%) the top glue and called it launch-bound (lever-2). Brick 8 folded the activation-quant into the two decode grouped kernels' prologue (fused gate+up+SwiGLU + down), quantizing one row/block cooperatively into a shared tile (`VT_V4_FUSED_QUANT`, both eager + graph). **BIT-EXACT** (`test_cuda_quant_dot` 4/4·106909 incl. a new grouped fused-quant nmse≤1e-6 case; `test_cuda_deepseek_v4` 19/19·66949; `test_deepseek_v4_gguf_load` 13/13·631; real 80.7 GB `=1` vs `=0` byte-identical 50-token "…Paris…"). **RESULT: decode 11.46 → 8.91 tok/s (−22.3%, 4 interleaved warm runs, non-overlapping).** nsys (grounded): the fused path removes `QuantizeQ8K` (125M ns) but the grouped kernels balloon 2.5–4.9× (+630M ns net) — folding the quant in forces THOUSANDS of GEMM blocks (down ≈6144/layer) to re-quantize their row behind a `__syncthreads` barrier, vs the ONCE-per-row standalone quant whose L2-cached Q8_K output all blocks read; "launch-bound" doesn't hold under the cudagraph. **REFUTED, no salvage** (per-block re-quant is intrinsic to the fusion); code reverted, production default unchanged (11.41). Same mechanism refutes fusing `QuantizeQ8_0`. CORRECTED next lever: the Q8_0 matvec `QuantDotGemmQ8_0Kernel` is now 49.3% of the step (Brick-3/4 latency/occupancy-bound) → a kernel-STRUCTURE change (multi-output-per-warp/persistent), NOT a fusion; then MhcPre/Route reduction glue (near-tie). Row stays `ACTIVE`.

**DeepSeek-V4 ds4-gap STUDY — 16.5 IS reachable, ranked lever plan (2026-07-30, research/profiling only, `.agents/specs/deepseek-ds4-gap-lever-plan-2026-07-30.md`).** Fresh nsys-diff (ours 85.35 ms/step = 11.2 t/s vs ds4 54.86 ms = 17.2 t/s; gap 30.5 ms, 98.5% GPU-bound; roofline ours 42–45% vs ds4 65–70% of 240 GB/s). **No irreducible gap** — the 30.5 ms is fully attributed to specific kernels, each with a ds4 file:line precedent. RANKED LEVERS (measured-%-anchored): **L1 dense Q8_0 preq-prologue fusion** (~9 ms; a MEASURED still-present 9.1 ms activation-quant DRAM round-trip — DISTINCT from the refuted Brick 8, which fused into the grouped-MoE kernel; near-tie; → ~13.5 t/s, the go/no-go), **L3 route→warp-topk** (6.5 ms, BIT-EXACT, → ~14.5), **L4 QKV+norm+RoPE+KV fusion** (4.0 ms, near-tie, → ~15.3), **L2 Q8_0 sub-warp GEMV tiling** (speculative occupancy lever, short-K only, 5–8 ms → ~15.8–16.5). Measurement REJECTED as wishful: MHC (measured TIE), routed-MoE (we WIN +2.6 ms via Brick 6), attention/fp8-KV (0.6 ms short-ctx), stream-overlap (98.5% GPU-bound). Honest band: **15.3 t/s sure (L1+L3+L4 bit-exact/near-tie), 16.5–17.2 if L2 tiling lands.** Implementation follows the gated sequence with explicit stop conditions. Row stays `ACTIVE`.

**Tier-A1 fold - 5 bf16 SwiGLU-MLP archs onto the shared gate-up seam `layers::UnquantizedMlpGateUpMethod` (2026-07-30).** OLMo-2, StableLM, qwen3_dflash, deepseek_v2 (dense MLP + shared-expert epilogue) and Granite now route their MLP through one tuned method and inherit the nvfp4 `GateUpFusedMarlinD` fused arm for free; the op sequence is byte-for-byte identical, no loader concat. Gated on the DGX production stack: `test_linear_method` 4/4 (RED-first composite golden) and the OLMo-2 SACRED paged-engine gate 1/1 - 13/16 STRICT token-exact + 3/16 near-tie, identical to the pre-fold result, with the anchor-drift REQUIRE against the committed pre-fold ids passing, so the fold is a proven byte-exact no-op on the real GPU forward. **Honest scope: only OLMo-2 has a committed golden**; Granite and StableLM goldens were never captured, so those tests link, load-verify and skip rather than running a vs-vLLM token gate, and dflash + deepseek_v2 are build-verified plus CPU-composite-covered only. Run detail in [.agents/benchmark-record.md](../.agents/benchmark-record.md) and [.agents/specs/arch-fusion-fold-plan-2026-07-30.md](../.agents/specs/arch-fusion-fold-plan-2026-07-30.md); next tiers A2 (MLA A-proj 3->1) / A4 (bf16 grouped-MoE).

**Q8_0 GEMV sudo-ncu root cause (2026-07-30, measurement-only, `.agents/specs/ds4-q8-ncu-2026-07-30.md`).** sudo ncu (now unblocked) on our `QuantDotGemmQ8_0Kernel`: **LATENCY-bound** (long-scoreboard stall 54.4) at 71.9% occupancy, L1 hit 96.7% (34-byte over-fetch L1-absorbed → not the DRAM bottleneck), L2 10.7%. Explains why Bricks 4/11/12 (alignment/occupancy/launch-count) were all flat — the axis is memory-LATENCY, not those. The one untried lever = ILP (multiple output-rows/thread → independent load streams). ds4 ncu side not captured (graph-decode needs --graph-profiling). No production code changed.

**ds4 Q8_0 GEMV — ds4 SIDE ncu captured + DIFFed (2026-07-31, measurement-only, `.agents/specs/ds4-q8-raw-mechanism-2026-07-30.md`).** Closes the residual of `ds4-q8-ncu-2026-07-30` (ds4 side never captured). sudo ncu `--replay-mode application --graph-profiling node` on ds4's `matmul_q8_0_preq_warp8_kernel` (the 1:1 kernel) on the byte-identical 6.15 GiB Q8_0 tower. **DIFF (ds4 vs ours): long-scoreboard 17.2 vs 54.4, regs 56 vs 39, occupancy 60.3% vs 71.9%, L2-hit 2.6% vs 10.7%, bytes/sector 1.70 vs 2.0.** Mechanism = **register-resident memory-level parallelism** (longSB inversely tracks regs across 3 kernels; occupancy non-monotone). Task's L2-residency + coalescing hypotheses both **REFUTED** (ds4 is LOWER/worse on both). VERDICT (A): a portable, bit-exact lever exists — deepen our single-row load pipeline (intra-row multi-block prefetch at ~56 regs), DISTINCT from the failed Brick-13 multi-ROW ILP; but bounded (~+1.5-2 t/s), does NOT overtake ds4's raw 16.5 alone. No production code. Box OOM-rebooted once on `--replay-mode kernel` (fixed → application replay). Decision surfaced to the user.

**Hygiene (2026-07-31): `VT_MLA_FUSED_NORM_ROPE` allowlisted (`88ae8fdf`)** — the A5 MLA `kFusedNormRope` fold rollback A/B switch (kernel-internal, no behavior change). Recorded here to pair with the env-doc allowlist entry.

**Tier-A4 fold — bf16 grouped-MoE gate+up+SwiGLU onto a SHARED fused op `vt::MoeGroupedGemmBf16GateUpSilu` (`kMoeGroupedGemmBf16GateUpSilu`) (2026-07-31, branch `fold/a4-bf16-grouped-moe` off `d05fff2d`, NOT pushed).** Last A-tier of the cross-arch merged-GEMM fold plan (`arch-fusion-fold-plan-2026-07-30.md` §A4). **OPEN-QUESTION RESOLVED (§3 gap 3):** qwen3_coder ALREADY routes its bf16 grouped MoE through task #90's fast `vt::MoeGroupedGemmBf16` (`qwen3_moe.cpp` `RunMoeBlock` → `qwen3_5.cpp` `MoeBlock` → `MoeBlockBf16Cuda`, default-ON), NOT the `qwen3_5.cpp:4256` reference loop — so A4 for coder is a launch-reduction/consistency fold, not a perf-gap fix. Both bf16 consumers (qwen3_coder `MoeBlockBf16Cuda`, deepseek_v2 `MoeBlock`) now replace their `{gate grouped-GEMM (f32); up grouped-GEMM (f32); MoeSiluMul}` triplet with ONE `vt::MoeGroupedGemmBf16GateUpSilu` call. The new op is BIT-IDENTICAL to that composite in EVERY regime by construction: decode/non-WMMA reuses the exact `MoeGroupedGemmBf16NaiveSplitK` partials + a fused reduce+SwiGLU launch (5→3 launches, drops the two f32 [P,I] reduce round-trips); prefill/WMMA reuses `LaunchGroupedBf16` twice + the identical silu-mul (byte-for-byte the old three ops). **HONEST DEVIATION:** the bf16 grouped path uses per-expert weight-POINTER ARRAYS `[E] i64` + a pair→token row_map, NOT the contiguous `[E*N,K]` + broadcast-act convention the keep-quant `kMoeGateUpSwiGLUGrouped` takes, so the bf16 arm is a SIBLING op (bf16 twin, documented in `merged_gemm.h`) rather than literally the same OpId. Route-weight stays in `moe_combine` post-down (invariant preserved); no loader change (both archs already resident-pack the per-expert bf16 gate/up towers). Default-ON behind `VT_MOE_BF16_FUSED_GATEUP` (`=0` = composite everywhere, same-binary rollback; allowlisted). **EMPIRICAL GATES (DGX GB10 sm_121a, RelWithDebInfo + cutlass-4.5.0 + triton).** A/B unit `test_ops_moe_grouped_bf16_gate_up_silu` PASS — fused == `{2× vt::MoeGroupedGemmBf16 (f32) + vt::MoeSiluMul}` BYTE-IDENTICAL (RED-first memcmp) across the naive(splits==1), split-K-decode, and WMMA-decode/prefill regimes; sibling regression `test_ops_moe_grouped_bf16` PASS. **`test_deepseek_v2_paged_engine` SACRED 8/8 UNCHANGED** (223 assertions; 5/8 STRICT + 3/8 near-tie, 92/128 tokens strict, max 0.25 nats @ prompt[3] tok=9, 0 forward-divergent — the near-tie prompts/tokens/gaps are IDENTICAL to the A2/A5 baseline ⇒ end-to-end BIT-EXACT), plus `test_deepseek_v2_load`/`_forward` PASS. **`test_qwen3coder_paged_engine` vLLM-0.25.0 golden 6/6 PASS** (138 assertions; 5/6 STRICT + 1/6 near-tie, 0 forward-divergent). Shared-TU (`qwen3_5.cpp`) inertness canary `test_qwen3_32b_nvfp4a16_paged_engine` 6/6 (142 assertions, 4/6 STRICT = 67/96 tok, max 0.062 nats) + `test_qwen27_paged_forward` PASS (27B/35B MoE take the fp4 path, never `MoeBlockBf16Cuda`; ops.h change is additive-before-`kCount`). **coder speed non-regressed by construction** (decode fuses to fewer launches with the SAME grouped-GEMM kernels/accumulation; prefill/c4+ decode is the byte-identical composite) — a launch-count/consistency fold, no isolated tok/s A/B claimed (mirrors the A2/A5 disposition; the same-binary rollback flag is available for an A/B). See docs/BENCHMARKS.md. A-tier of the fold plan now COMPLETE (A1/A2/A3-carry/A4/A5); remaining rollout: Gemma B1 (deferred not-bit-exact), C2/C3 vision towers, D/E tiers.

**Laguna-S-2.1 W4 CUDA-verify (2026-07-31, `.agents/specs/laguna-s21-w4-cuda-verify-2026-07-31.md`).** The full CUDA fast-path build of the W1-W3 tree is CLEAN (1059/1059, 0 warn; all 4 laguna TUs). BUT the real-GGUF forward is NOT yet built: `LoadLagunaFromGguf` is a `VT_CHECK(false)` stub, `LagunaModel::Forward` is an f32 reference fed only synthetic weights, no run entrypoint — so the engine cannot yet run Laguna on real bytes (this is W5, not W4). Reference established: llama.cpp Poolside-fork runs the UD-Q4_K GGUF COHERENTLY ("…Paris." @ 27.8 tok/s). Two W5 blockers scoped: a multi-shard GGUF reader (3 shards) + disk (build tree ~137GiB alongside the 69GiB model). Row stays `ACTIVE` (NOT runnable). W5 fully scoped + de-risked in the spec.

**Laguna-XS-2.1-NVFP4 long-context decode levers LANDED + MEASURED on GB10 (2026-08-03, `CLAIM-LAGUNA-LONGCTX-LEVERS`).** Two source-verified levers that diverge from vLLM by ~0 at the 6-token benchmark but grow LINEARLY with context, now measured on the two-length (256 & 2048) slope. **Lever B — window-bounded SWA reads (`VT_LAGUNA_SWA_WINDOW`, default ON, BYTE-EXACT):** the four `DecodeAttnGqa*` decode kernels stream only the ~512-row sliding window instead of the full grown deck (vLLM bounds the read; `laguna.py:412`) — byte-exact (only fully-masked tiles skipped; verified off-GPU over 18,600 configs AND token-IDENTICAL `=1` vs `=0` on GB10). **MEASURED −0.30 ms/step at ~2k context** (GB10, `VT_LAGUNA_DECODE_GRAPH=1`, drop_caches, two-length nsys diff); ~0 at ≤512 as predicted (the deck fits the window) — so the slope is real and grows with context. Ships DEFAULT-ON (byte-exact + measured win). **Lever A — bf16 paged KV (`VT_LAGUNA_KV_BF16`, default OFF/opt-in, near-tie UNRATIFIED):** halves the decode-attn KV DRAM to match vLLM's bf16 KV via parallel bf16 storage + on-append RNE cast + in-register widen; f32 path byte-identical when off. Measured a distributional near-tie on the slope but the argmax-flip risk that sank the prior short-context attempt (`CLAIM-LAGUNA-KV-ATTN-BF16`) is NOT retired to ratification, so it stays **default-OFF (opt-in only)** pending a strict near-tie sign-off. See BENCHMARKS `CLAIM-LAGUNA-LONGCTX-LEVERS`.

**D3 GPU gate CLOSED (2026-07-31, GB10).** OLMo-2 full-width qk-norm fold empirically confirmed: olmo2 16/16 + fused_chain 10/10 + qwen3/27B canaries unchanged. See BENCHMARKS.

**A3 fold scoping (2026-07-31).** qwen3_5 GGUF keep-quant grouped MoE: premise bit-exact (ResidentWeight keeps weights quantized → the per-expert loop already runs the ds4 vec_dot). Remaining-work list SUPERSEDED by the W2/W3a/W3b LANDED entries below. See `.agents/specs/arch-fusion-fold-plan-2026-07-30.md` A3.

**Laguna W8 embed-fix LANDED + GATED (2026-07-31).** `LagunaEmbed` gathers only the T needed rows instead of converting the whole 1.23 GB embed table to f32 every token. GATED on real UD-Q4_K_XL GGUF (GB10, `--gpu`): TOKEN-IDENTICAL to the W5/W6 golden + decode **0.66 → 0.17 s/tok (3.9×; 1.5 → 5.9 tok/s)**. Bit-exact by construction. See `.agents/specs/laguna-s21-w7-speed-2026-07-31.md` §W8.

**Laguna W9 grouped-expert MoE LANDED + GATED (2026-07-31).** The 30 un-grouped per-expert GEMV launches/step fold onto the shared `vt::MatmulBTQuantGrouped` (3×top_k → 3 launches/token, no loader change). Same-binary A/B on real UD-Q4_K_XL (GB10, `--gpu`): grouped==per-expert BYTE-IDENTICAL (md5 `754728c6`, == W6 golden) + decode **0.18 → 0.13 s/tok (1.38×)**. Cumulative with W8: **0.66 → 0.13 s/tok (5.1×, 1.5 → 7.7 tok/s)**. See spec §W9.

_(Laguna W8-W11 decode-speed campaign: cumulative 0.66 → 0.13 s/tok. W10's host-sync and W11's GPU-compute-bound readings, and W11's demotion of device-residency, are ALL superseded by the 2026-08-04 binding result above. See `.agents/state.md`, spec §W10/§W11.)_

_(Laguna NVFP4-Marlin decode-isolated (2026-08-01, `CLAIM-LAGUNA-DECODE-SYNC`, BENCHMARKS): the ~10 tok/s Marlin-default path's 1.9× gap to vLLM 18.8 is HOST-SYNC — 60-tok−20-tok nsys÷40 = **~432 `cudaStreamSynchronize` + ~188 `cudaMemcpyAsync` per decode token, ZERO malloc/free** (the 72k allocs are all the one-time load-repack; DBuf pool recycles). Byte-exact lever = device-resident decode (kills the memcpy) + decode CUDA-graph (collapses 432 syncs→1), vLLM's mechanism, gated on golden ids UNCHANGED. Executable plan: `.agents/specs/laguna-device-resident-decode.md`; same "Brick D" as DeepSeek task #228. Unlike the GGUF W11 path (GPU-compute-bound), the NVFP4-Marlin path IS host-sync-bound — the Marlin MoE kernel is vLLM's own, so compute is at parity.)_

_(Laguna decode KV+attention → shared-framework port, bf16-KV slice (2026-08-02, `CLAIM-LAGUNA-KV-ATTN-BF16`, BENCHMARKS): scoped the tractable slice (private f32 KV → **bf16**, vLLM's own paged-KV dtype, halving the memory-bound decode-attn DRAM bytes, inside Laguna's own `DecodeAttnGqa*` kernels — SACRED 27B/35B byte-untouched; full `AttnBlock` port OWED — the shared FA2 pure-decode throws on `window_size` and has no gate epilogue, so it needs per-layer sliding-window + a softplus-head-gate hook grown additively). Built clean on GB10 (7 TUs, 0 warn) + DGX-gated on `laguna-xs-nvfp4` (decode-graph path). **RESULT: WASH + near-tie prefix BREAK — reverted, NOT landed.** Speed: bf16 3.93s vs base 3.84s decode/127 (within noise) — did NOT move the residual (at XS ~130-tok context the KV read is a small share; `lm_head_gemv`+Marlin MoE dominate, confirming the residual is GEMV/MoE not attention). Correctness: bf16 flips the argmax at decode step 2 (268→22345) so it no longer shares vLLM's first-20 prefix — the device regime is not bit-identical to vLLM, so a near-tie flips even moving TOWARD vLLM's dtype. Default stays f32 KV (base golden prefix + ~33 tok/s, re-verified after restore). See `.agents/specs/laguna-kv-attn-port-2026-08-02.md`.)_

**qwen3_5 A3 keep-quant grouped-MoE — LANDED, COMPLETE (2026-07-31).** W2/W3a: keep-quant experts load as one stacked [E*out,in] tower + byte-exact `ExpertMlpKq` slice forward via a ResidentWeight slice (CUDA `needs_weight_staging`=TRUE). W3b: that loop collapses to 3 grouped `vt::MatmulBTQuantGrouped` launches, `VT_QWEN35_GROUPED_MOE` default-ON, sharing Laguna W9's descriptor — DGX APEX-Compact grouped(=1) vs per-expert(=0) byte-identical and strict-passing the golden; APEX-Balanced 2nd-case all-zeros is PRE-EXISTING (tasks #50/#65). **That gate was CUDA-only — see the CPU P0 below.** W1, the down-GEMM bf16-out hazard and two non-landing attempts are superseded; see the record + `.agents/specs/qwen35-a3-grouped-moe-2026-07-31.md`.

**CPU grouped keep-quant GEMM was f32-ONLY — P0 FIXED (`QUANT-GGUF-CIQ-GEMM`, 2026-08-06).** `KqGrouped` (W3b) first passed `vt::MatmulBTQuantGrouped` a **bf16** activation; the CPU provider addressed both row views as `float*` with a hardcoded `kF32` activation row, so bf16/f16 rows were mis-strode and mis-decoded: CPU GGUF 35B decode was all-token-0 from `b4f5610a` on, while CUDA stayed byte-exact (it honours `act.dtype`) Fixed at `src/vt/cpu/cpu_quant_gemm.cpp:220-268`: rows addressed by `SizeOf(act.dtype)`/`SizeOf(out.dtype)`; `repacked`/`q8_0_aligned` now propagate onto the slice (CIQ-G7 mode). Gated per dtype in `tests/vt/test_ops_quant_dot.cpp`.

**Records-only repairs; no capability change.** 2026-07-31: six `DONE` rows re-pointed at closure commits. 2026-08-06 audit: all 10 stale `ACTIVE` rows → `READY`; 11 claims retired, 9 amended. That reconciliation is now a STANDING GATE, wired only after the record was repaired so it never had to be relaxed to pass: `scripts/audit-live-rows.py --check` fails on any `ACTIVE` row with no branch and no commit behind it, and runs in `agent-preflight.sh` and CI with its mutation suite. A red gate means the record drifted, never that the gate is too strict.

**That gate's KNOWN LIMIT, recorded not fixed (2026-08-06).** Its evidence rule is a commit-MESSAGE mention with no code-touch filter, and the repair's own commits name 9 of the 10 vacated IDs, so it cannot re-detect those ten after merge. The fix (filter on code-touching commits) reddens 8 records-only-backed rows today, so it needs its own decision; follow-up in `.agents/state.md`. `.agents/workflow.md` now states the `ACTIVE` precondition the gate enforces.

**Shared pure-dense decode CUDA-graph — `Qwen3DenseModel`, 5 registrations (2026-08-02).** Added `Qwen3DenseDecodeGraph` (opt-in `VLLM_CPP_QWEN3_DENSE_DECODE_GRAPH`), the pure-dense sibling of the shipped `Qwen3MoeDecodeGraph`, driving the SHARED `Qwen3DenseModel` forward so `Qwen3ForCausalLM` + `LlamaForCausalLM` + `InternLM3ForCausalLM` + `MistralForCausalLM` + `InternLM2ForCausalLM` all pick up a captured decode graph at once (Llama/Mistral/InternLM2 alias `Qwen3DenseModel`). The eager forward is split into `EmbedInto` (outside capture) + `ForwardLayers` (the captured region); same cold→warm→capture→replay state machine + padded-batch capture set as the siblings. dgx GB10 SACRED near-tie gate (`test_qwen3_paged_engine`) with the graph ON: Qwen3-0.6B + Qwen3-4B both 16/16 PASS, 0 forward-divergent, BYTE-IDENTICAL to the eager (graph-OFF) baseline (captured S=1, 239 replays/checkpoint). Default OFF keeps the eager path byte-identical (SACRED default gates untouched). Rigorous steady-state decode tok/s (eager vs graphed) OWED. See commit.

**CI repair — CPU build leg + two static-gate holes (2026-08-02).** Three jobs were RED on `main` (`ca5c7adc`); no lifecycle, support status or forward path changed.
* `build-test-cpu` + `sanitize-cpu` failed at the **Build** step: `tests/vt/test_cuda_quant_dot.cpp` included `<cuda_runtime.h>` and declared the Brick 12 `MatmulQ8_0{Pair,GroupDiag}Cuda` externs unconditionally, so the CPU-only and sanitizer legs (which force `VLLM_CPP_CUDA=OFF`) could not compile it at all. The include and the two A/B gates that call those `.cu` symbols DIRECTLY are now `#ifdef VLLM_CPP_CUDA` (the in-tree pattern from `test_ops_nvfp4_fp4.cpp` / `test_dropin_abi.cpp`); the file's other 7 cases still skip cleanly and the CPU leg now builds, links and runs them 7/7.
* With Build unblocked, the CPU **Test** step became reachable for the first time since Brick 12 and exposed 2 pre-existing RED tests that no gate could see while the build was down — both cases of a test lagging a DELIBERATE change: `test_model_loader_gguf` asserts the supported-architectures error string but was never given `LagunaForCausalLM` when Laguna registered; and `test_kimi_k3_scaffold` still asserted `has_inner_state == true` after `4d1be010` deliberately set it FALSE (the outer `ConditionalGeneration` wrapper mirrors the Qwen3.5 hybrid-mm convention — the recurrent/conv state belongs to the inner `KimiLinearForCausalLM`). That commit updated `test_model_registry` and missed this second assertion. Both tests corrected to match the shipped registrations; no production behavior changed.
* `agent-record` runner-routing was RED on a stale expectation that was HIDING a real hole. `deepseek_v4` builds its device-resident carrier in its own `WrapV4DeviceLogits` instead of the shared `WrapDeviceLogits`, so `check-runner-routing-consistency.py` matched NEITHER seam and classified it `NONE` — and `NONE` is not an error state, so the model dropped OUT of the HOST-drift check while the gate still reported green. A hole that silently EXEMPTS a model is worse than a red gate, so the classifier now follows ONE hop into a file-local `ForwardLogits`-returning helper (`classify_with_helpers`). ds4 reads `DEVICE`, its now-decorative allowlist entry is retired (3 → 2 allowlisted), and the tree holds **0 models in the `NONE` bucket**. Two mutation tests added — the live-tree case plus a synthetic both-ways check that a HOST helper is NOT laundered into DEVICE; suite 18/18.
* `device-leakage` DSR ratchet was RED at 42 vs baseline 32. `laguna_device.cpp`'s 2 `kCUDA` lookups are ALLOWLISTED: it is the same resolver-TU shape as the already-allowlisted `deepseek_v4_device.cpp` and shares its deferred follow-up (thread the runner `DeviceType` through both). The 8 new `#ifdef VT_MARLIN_NVFP4` sites in `laguna.cpp` / `laguna_weights.cpp` are **TRACKED DEBT, not a repair**: each carries a `DSR-ALLOW(S1)` marker naming the specific repair it owes. The **baseline is UNCHANGED at 32** (never raised) and the 8 exemptions print loudly on every CI run. HONEST LIMITATION: the 2 `laguna_weights.cpp` sites gate a host-only bf16 row concat with no device dependency of its own — the cheapest genuine repair (ask `LagunaDeviceKernelsAvailable()` at runtime instead) — but it moves weight loading and needs the GB10 Laguna re-gate, which was not available for this change.

<!-- decode/runtime framework-routing (2026-08-02) -->
- **Decode/runtime framework-routing** — `GATING`. Third "MUST route through" seam codified (AGENTS.md) + CI-enforced (`check-runner-routing-consistency.py`, green: 27 models, 3 allowlisted). Landed: Laguna split-K attention + w13-fusion (82.7→83.2% of vLLM decode); shared `Qwen3DenseDecodeGraph` (5 registrations, GB10 token-exact 16/16); Qwen3VL + DeepSeek-V4 routed to on-device logits + runner entry. Pending: Laguna gate-up fold onto the shared MLP seam (allowlisted) + its full KV/attention port onto `AttnBlock` (the kernel-level residual to vLLM parity); CUDA re-gates for Qwen3VL/DS4 owed.

**Darwin Laguna build repair (2026-08-03).** The Marlin-only
`LagunaMarlinMoeEnabled` helper now shares the `VT_MARLIN_NVFP4` feature guard
used by every call site, so non-CUDA AppleClang builds do not emit
`-Wunused-function` under `-Werror`. A static regression gate enforces that
boundary; runtime behavior and Laguna's lifecycle state are unchanged.
The next run also guarded Voxtral's GCC-only `-Wstringop-overflow` suppression
out of Clang, where it was fatal. Its Go `go-m1cpu` diagnostics were nonfatal
and outside this repo.

**Darwin Qwen3.5 build repair (2026-08-16).** The MoE layout refusal lambda no
longer captures its namespace-scope help string. Apple Clang diagnosed that
redundant capture as `-Wunused-lambda-capture`, and the project promotes the
warning to an error. The lambda can still read the namespace-scope name
directly. Runtime behavior and the Qwen3.5 lifecycle state are unchanged.

**Agent onboarding:** [session](../.agents/specs/session-onboarding.md) +
[entry](../.agents/specs/developer-agent-protocol-entrypoint.md) implemented;
documentation-only.
