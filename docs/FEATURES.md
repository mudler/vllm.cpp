# Features <!-- ENG-RELEASE-WINDOWS: state=ACTIVE publication=pending artifact=unpublished -->

What vllm.cpp supports, next to the engines it is measured against. This page is
a **keyed table**: one row per feature, kept current. It is not a changelog.

For measured speed see [BENCHMARKS.md](BENCHMARKS.md); for per-capability
lifecycle conventions see [Project status](../README.md#project-status); for
the agent-facing parity inventory with upstream file references see
[.agents/feature-matrix.md](../.agents/feature-matrix.md).

**Legend.** ✅ supported and gated. ◐ partial, usable with named gaps. ☐ not yet.
n/a means the feature does not apply to that engine's design.

Reference versions: vLLM 0.26.0.dev0, SGLang v0.5.15, llama.cpp `b10451`,
MLX-LM as of 2026-07. Competitor columns describe what those projects ship, and
are our reading of their documented behavior, not measurements.

## At a glance

| | vllm.cpp | vLLM | SGLang | llama.cpp |
|---|---|---|---|---|
| Language | C++20 | Python + CUDA | Python + CUDA | C/C++ |
| Runtime deps | none | PyTorch | PyTorch | none |
| Install size | **66 MiB** | 9.1 GiB | comparable to vLLM | comparable to us |
| Embeddable behind a C ABI | ✅ | ☐ | ☐ | ✅ |
| Weight formats | Safetensors + GGUF | Safetensors | Safetensors | GGUF |
| Correctness gate | token-exact vs vLLM | reference | own | own |
| Architectures | 40 registered, 27 gated | 130+ | 100+ | 100+ |
| Downloadable server binaries | ✅ v0.0.2: eight indexed archives with checksums, provenance, manifests, and SBOMs. Windows ZIP downloads do not exist; native CPU/Vulkan lanes await hosted runtime, dry-run, prerelease, and authenticated audit gates | ✅ wheels/containers | ✅ wheels/containers | ✅ host-specific binaries |
| Native Windows builds | ◐ CPU/Vulkan: `/MT /W4 /WX`, central `NOMINMAX`, UTF-8, aligned allocation, C++20 `std::numbers` pi, runtime ISA dispatch. Local closure includes the float-domain DeepSeek probe; hosted compile/runtime/release pending | ✅ | ✅ | ✅ |

## Serving and scheduling

| Feature | vllm.cpp | vLLM | SGLang | llama.cpp |
|---|---|---|---|---|
| Continuous batching | ✅ | ✅ | ✅ | ◐ |
| Chunked prefill | ✅ | ✅ | ✅ | ☐ |
| Automatic prefix caching | ✅ | ✅ | ✅ (radix) | ◐ |
| Preemption and recompute | ✅ | ✅ | ✅ | ☐ |
| Priority scheduling | ◐ gating (`--scheduling-policy priority` reaches the server; scheduler-unit tests only, no engine-level priority-vs-FCFS gate exists yet, #534) | ✅ | ✅ | ☐ |
| LPM cache-aware admission | ✅ | ☐ | ✅ | ☐ |
| In-batch prefix de-prioritization | ✅ | ☐ | ✅ | ☐ |
| Async / overlap scheduling | ✅ default on (UAF-safe drain; device token-ids mirror on gate + classic-dense; the decode graph declines while the mirror is live (#323 fix, eager fallback); opt-in `VT_ASYNC_EXECUTOR` out-of-capture H2D staging) | ✅ | ✅ | ☐ |
| CUDA graph decode capture | ◐ per-family | ✅ | ✅ | ✅ |
| Partial-prefill concurrency | ☐ | ✅ | ✅ | ☐ |
| Cascade attention | ☐ | ✅ | ◐ | ☐ |

## KV cache and memory

| Feature | vllm.cpp | vLLM | SGLang | llama.cpp |
|---|---|---|---|---|
| Block-paged KV with refcount and LRU evict | ✅ | ✅ | ✅ | ◐ |
| Hybrid KV groups (full attention + GDN/Mamba) | ◐ GDN gate activation resolved from the checkpoint's `output_gate_type` (silu/swish/sigmoid; anything else refused at load, #489) | ✅ | ◐ | ◐ |
| Sliding-window and chunked-local attention | ◐ | ✅ | ✅ | ✅ |
| fp8 KV cache | ◐ `--kv-cache-dtype fp8` halves the block, so a fixed `--kv-cache-memory` buys 2x the blocks and the DEFAULT 256-block path halves the pool bytes instead. Costs the bf16-native FA-2/WMMA/vector kernels (net UNMEASURED). 16 archs, MLA, the C ABI are refused before any write; only 1 arch names fp8 back. CUDA UNRUN ([spec](../.agents/specs/fp8-kv-cache.md)) | ✅ | ✅ | ✅ |
| KV offload to host memory | ✅ | ✅ | ✅ | ☐ |
| External KV provider ABI (LMCache) | ☐ | ✅ | ◐ | ☐ |
| KV events (block create / evict publish) | ◐ no transport | ✅ | ☐ | ☐ |
| Prefix-cache matching unit | ◐ resolver only | ✅ | ☐ | ☐ |
| Compute directly on quantized blocks | ✅ | ☐ | ☐ | ✅ |
| Scratch allocator keyed by device (two backends, one process) | ✅ since [#516](https://github.com/mudler/vllm.cpp/issues/516); a pool is bound to one backend and refuses any other, and a backend with no registered platform is refused rather than given another's residency cap | ✅ device is field 0 of the allocation handle | ✅ | ✅ |
| Automatic memory sizing (no hand-tuned budget) | ☐ hand-typed block count | ☐ percent, hand-tuned | ☐ | ◐ |
| Memory cap with a pre-flight error instead of an OOM | ☐ | ◐ KV pool only | ◐ | ☐ |
| Routed-expert weight streaming from disk | ◐ default OFF (`VT_MOE_EXPERT_STREAM=1`), keep-quant/keep-f16 towers (#1378); bounded slot cache; refuses unfittable slices by name. c1-c4 capacity, not throughput. CPU; staging device DECODES, token gate FAILS (#1299) | ☐ blanket `cpu_offload_gb`, not expert-granular | ☐ | ◐ mmap only |

## Quantization and weight formats

| Format | vllm.cpp | vLLM | SGLang | llama.cpp |
|---|---|---|---|---|
| NVFP4 (W4A4 and W4A16 Marlin) | ✅ | ✅ | ✅ | ✅ in GGUF, not safetensors (#979). Was wrongly ☐: `GGML_TYPE_NVFP4 = 40` (`ggml.h:430`), CUDA MMQ and the ModelOpt repacking converter are UPSTREAM at pin `b10451`, the sm_121a GEMMs fork-local |
| NVFP4 dense sinks take vLLM's dense Marlin, not the single-expert MoE route | ✅ `VT_MARLIN_DENSE` (single projection, `efa6e40d`) + `VT_MARLIN_DENSE_PAIR` (fused shared-expert gate_up), both default-ON; the pair sink measured +1.31% at c8 / +1.38% at c4 on 35B-A3B, SACRED 315/315 + 235/235 | ☐ | ☐ | ☐ |
| Dense W4A16 MLP runs ONE merged `gate_up` Marlin GEMM (vLLM's `MergedColumnParallelLinear` topology) | ✅ `VT_DENSE_MARLIN_GATEUP`, **default ON** (opt out `=0`): the A/B measured +2.12% c1 / +1.70% c8 on the 27B, arms separated, tokens identical (#365). Replaces the split pair's 193 Marlin calls/step vs the oracle's 129 | ✅ | ☐ | ☐ |
| NVFP4 shared-expert `down_proj` kept bf16 (no f32 round-trip) | ✅ `VT_SHARED_DOWN_BF16` default-ON; bit-identical (both consumers widen bf16 in-kernel and re-round on store), SACRED 315/315 + 235/235 on BOTH arms with unchanged assertion counts; +2.05% c8 / +0.79% c4 on 35B-A3B | ☐ | ☐ | ☐ |
| NVFP4 `lm_head` kept packed (no dequant at load) | ✅ `VT_LMHEAD_FP4` default-ON, #213; CUDA-gated on `nvidia`@`0893e160` (continuations byte-identical packed vs dequant, 235/235; RSS -1.70 GiB on CUDA, owed a re-measure; a no-fp4-GEMM backend keeps one bf16 operand too) | ✅ | ☐ | ☐ |
| GGUF k-quants and i-quants | ✅ (CPU grouped keep-quant MoE bf16 regression in `b4f5610a` fixed 2026-08-06). **CPU quant compute is ISA-tiered:** Arm has i8mm + repack; x86_64 portable-only, MEASURED open on every axis (CIQ `G5`, #433) | ☐ | ☐ | ✅ |
| GGUF F16 weights kept resident as F16 (no BF16 promotion) | ✅ `VT_GGUF_KEEP_F16` default-ON (CPU), the f16 GEMM computes on it directly. Default settled 2026-08-17, a memory-for-speed trade: 1.05 GiB less peak RSS for ~9% prefill and ~1.4% decode, tokens identical. `0` opts out | ☐ | ☐ | ✅ `ggml_vec_dot_f16` |
| GGUF is a TWO-engine comparison at these pins (#979) | ✅ text-only `qwen35`, no `clip` projector (#821) | ☐ REMOVED from the tree in `6635279d8`, now an unpinned out-of-tree `vllm-gguf-plugin` | ☐ full stack present, `qwen3_5` unreachable behind FOUR blockers, and the load path has NO completeness guard so a clean-looking load proves nothing | ✅ native, `LLM_ARCH_QWEN35` |
| AWQ | ◐ CPU dequant | ✅ | ✅ | ☐ |
| GPTQ | ◐ CPU dequant | ✅ | ✅ | ☐ |
| MXFP4 compressed-tensors | ◐ W4A16 Marlin, mem 2.63x less. gate_up FUSION + decode-graph default-ON; #44 3/3, 32B 6/6. **`VT_MARLIN_DENSE` DEFAULT-ON** (`KERNEL-MARLIN-DENSE-EXEC`): dense marlin 48-CTA, byte-faithful, beats MoE (c8 0.969) | ✅ | ✅ | ☐ |
| Compressed-tensors `mixed-precision` (`config_groups`, ordered regex `targets`) | ◐ scheme read from the config, never from a dtype probe; an arm with no loader is REFUSED BY NAME. On `unsloth/Qwen3.8-27B-NVFP4` the W4A4 group loads, the FP8 W8A8 group and the `kv_cache_scheme` are refused (#821) | ✅ | ✅ | ☐ |
| ModelOpt `MIXED_PRECISION` (`quantized_layers`, exact module names) | ◐ the declared `quant_algo` is CROSS-CHECKED against the shipped operand names at load; a disagreement, an unimplemented algo or an unknown operand family is REFUSED BY NAME. `r0b0tlab/...-MTP-sm121` loads (#821) | ✅ | ✅ | ☐ |
| ModelOpt `MIXED_PRECISION`: the cross-check moves no arm | ✅ `nvidia/Qwen3.6-27B-NVFP4` @`0893e160` is NOT refused, and one case says so: same format, wildcard exclusions, an `input_scale` on every NVFP4 module, a `kv_cache_scheme` shipping zero scales (#821) | ✅ | ✅ | ☐ |
| ModelOpt `MIXED_PRECISION`: STATIC per-tensor FP8 + W4A16_NVFP4 | ◐ LOADS, never RUN. `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121` carries 208 `input_scale`, `F32` scalar `weight_scale` and `"dynamic": false`, so the three `unsloth` blockers are absent (#1574, #821) | ✅ | ✅ | ☐ |
| fp8 weights, per-tensor scale | ✅ | ✅ | ✅ | ☐ |
| Block-wise (fine-grained 128x128) FP8, the `weight_scale_inv` layout | ◐ RUNS on CPU and on CUDA sm120 (#1189 M4/M6): 10 projections as 7 GEMMs, `gate_up`+QKV merged; N,K %128==0 only. **TOKEN GATE PASSED** on `Qwen/Qwen3.8-27B-FP8` vs the pinned oracle on GB10, 7 prompts x 16 tokens greedy, 6 strict + 1 in-band, weights resident at 1 byte/element; no speed number ([gate](../.agents/specs/gate-qwen38-27b-fp8-block.md), [kernel](../.agents/specs/vt-matmul-fp8-block-cuda.md)) | ✅ | ✅ | ☐ |
| Per-tensor FP8 W8A8 linear is a shared seam any model can bind | ✅ `models/dense_fp8_gemm.h` + `layers::Fp8W8A8LinearMethod` (#940), bound via `layers::MakeLinearMethod`. One definition, CUDA only ([spec](../.agents/specs/vt-fp8-shared-seam.md)) | ✅ `Fp8LinearMethod` | ✅ | ☐ |
| FP8 W8A8 works on a CUDA arch without `cutlass-fp8` | ✅ `vt::QuantFp8Static` registers from an unconditional TU (#960); sm_110 measured ([spec](../.agents/specs/vt-fp8-quant-arch-gate.md)) | ✅ | ✅ | ☐ |
| fp8-tower GDN `in_proj` emits bf16, unlocking packed GDN decode | ◐ `VT_GDN_FP8_IN_BF16` + `VT_GDN_PACKED_DECODE_FP8_TOWER` (inert alone), both default **OFF**, ungated (#339) ([spec](../.agents/specs/perf-fp8-alpha-fold.md)) | ✅ bf16 `out_dtype` | ☐ | ☐ |
| Merged fp8 projection folds per-column alpha in the GEMM epilogue | ◐ `VT_FP8_ALPHA_VEC_EPILOGUE`, CUDA only, default off, ungated; refuses split-K under a bf16-D equivalence claim (`claims_splitk1_premise`, default off) | n/a | n/a | n/a |
| Per-tensor FP8 CUTLASS GEMM picks a decode-sized tile at decode-sized M | ◐ `VT_FP8_CUTLASS_SMALL_M` (default **on**, `=0` rolls back), CUDA sm_12{0,1}a only, UNGATED on a device: upstream's four-way sm120 M ladder (`M<=16` 16x64x128 with a 16x32 `EpilogueTile`, `M<=32` 32x64x128 with 32x32, `M<=256` 64x64x128, else 128x128x128) instead of the two-way ladder this tree shipped, where every `M<=256` took the 64-row tile (#1866). Reached on `VT_DENSE_CUBLASLT_FP8=0` and as the cuBLASLt lane's no-heuristic fallback; the DEFAULT fp8 arm is still cuBLASLt, so a stock decode step does not take it. Ladder gated by value on the host tier; the CUDA compile is measured green by the `cuda-fat-build` job on head `d9bf525c0` (run `32802716762`, whose own conclusion is `cancelled` for an unrelated cancelled leg), and no compiled CUDA has changed since, while the current head's own `cuda-fat-build` is PENDING; the token gates and the arm A/B are owed ([spec](../.agents/specs/perf-fp8-small-m-dispatch.md)) | ✅ `cutlass_gemm_sm120_fp8_dispatch` | ☐ | ☐ |
| `vt::MulColVecF32` carries a bf16 store width | ✅ f32 arm byte-identical; bf16 arm rounds once; CPU + CUDA | n/a | ☐ | ☐ |
| bf16 / fp16 | ✅ | ✅ | ✅ | ✅ |
| Safetensors direct load, no conversion | ✅ at ANY tensor byte offset: the format aligns nothing, so no loader forms a typed pointer into the mapping. Last three fixed by #772; a checker is still owed on #627 | ✅ | ✅ | ☐ |
| Weights uploaded straight from the file mapping (no host copy first) | ◐ verbatim tensors only (37.8% of 27B BF16); arbitrary-offset reads are defined, including Laguna graph staging. Merged/transposed and merged FP4 weights still copy | ✅ | ✅ | ✅ mmap |

## Model coverage

The supported set is exactly what the C++ registry registers: every
architecture self-registers via `REGISTER_VLLM_MODEL`, and
`scripts/check-supported-models.py` gates this list against the source so it
cannot drift. Today that is **40 registered architectures**. Each row names the
checkpoint it was gated against and the verdict; caveats are in
[Project status](../README.md#project-status), agent detail in `.agents/model-matrix.md`. A mergeable
gate/up MLP routes through one shared merged-GEMM method, so a tuned arm added
once reaches every such arch; Command-R, GLM-4, MiniCPM, MiniCPM3 and Phi-3
joined on 2026-08-10 (#299), and
`scripts/merged-gemm-consistency-allowlist.txt` lists the rest with their
blocker.

Gate words: **strict** is token-for-token identical to the vLLM oracle;
**near-tie** is the ratified distributional gate used where vLLM's own greedy is
bf16-non-deterministic; **scaffold** means registered and config/loader-gated
but the forward is not yet a real-checkpoint run. Speed is a separate bar (match
or beat the reference on every axis); most rows are correctness-complete and
speed-pending, which [BENCHMARKS.md](BENCHMARKS.md) tracks.

### Registered architectures

<!-- supported-arch-table:begin -->
| Architecture | Tested checkpoint(s) | Correctness gate | Speed vs reference |
|---|---|---|---|
| `Qwen3_5ForConditionalGeneration` | Qwen3.6-27B NVFP4 (`unsloth` @`890bdef7`, `nvidia` @`0893e160`); Qwen3.5-4B BF16; **Qwen3.8-27B BF16** @`1d4bf0f2` | 27B strict 235/235 text + 32/32 image/video; 4B cached 3/3; Qwen3.8-27B 4/7 strict, 3 exact fp32 ties in band (#915) | `unsloth` 27B at/above vLLM, ModelOpt 0.85x; 4B 1.021x; 3.8-27B c4 **0.963x**, c1/c8 absolutes (#915). Loads BF16/per-tensor FP8/NVFP4 (CT+ModelOpt); `modelopt_mixed` FP8 tower NATIVE (#164), GDN qkvz merged. CUDA/CPU |
| `Qwen3_5MoeForConditionalGeneration` | Qwen3.6-35B-A3B (NVFP4 text; published BF16 text + vision tower) | NVFP4 strict 315/315 vs vLLM 0.25.0; published BF16 6/7 prompts strict 16/16 vs the pin, 7th an exact tie (#910). Image/video IMPLEMENTED, NOT GATED (#891): the tower loads and runs, mm gate OWED | gate model: 0.93x to 1.03x grid; NO BF16 or mm speed claim |
| `Qwen3_5ForCausalLM`, `Qwen3_5MoeForCausalLM` | none: no text-only Qwen3.5 checkpoint fits this hardware | **NO RUN GATE, OWED.** Gated on `test_qwen3_8_text_only.cpp`; NO token claim. Loader reads stacked BF16 experts (#740) plus BF16 towers, shared expert and `lm_head` (#864), so both published indices satisfy the load plan | not measured |
| `Qwen3ForCausalLM` | Qwen3 dense 0.6B/1.7B/4B/32B, NVFP4A16 | near-tie strict 16/16 vs vLLM 0.25.0 | c1 every-axis parity, c8 decode residual |
| `Qwen3MoeForCausalLM` | Qwen3-Coder-30B-A3B | strict 6/6 vs vLLM 0.25.0 | 11/16 grid cells at or above graphed vLLM |
| `Qwen3VLForConditionalGeneration` | Qwen3-VL-4B-Instruct (image + video) | image strict 32/32, video near-tie vs vLLM 0.25.0 | vision tower 0.57x vs vLLM encode; umbrella pending |
| `LlamaForCausalLM`, `InternLM3ForCausalLM` | Llama-3.2-1B, 01-ai/Yi-Coder-1.5B-Chat, internlm3-8b-instruct | strict 16/16 each vs vLLM 0.25.0 | pending |
| `InternLM2ForCausalLM` | internlm2-chat-1_8b | near-tie 16/16 vs vLLM 0.25.0 | pending |
| `MistralForCausalLM` | Mistral-7B-v0.3 | strict 16/16 vs vLLM 0.25.0 | pending |
| `OPTForCausalLM` | facebook/opt-125m | strict 6/6 vs vLLM 0.25.0 | pending |
| `PhiForCausalLM` | microsoft/phi-2 | near-tie 16/16 vs vLLM 0.25.0 | pending |
| `Phi3ForCausalLM` | microsoft/phi-4 (14B), Phi-3 | strict 16/16 vs vLLM 0.25.0 | pending |
| `GemmaForCausalLM` | google/gemma-1.1-2b-it, unsloth/gemma-2b | near-tie 48/48 vs vLLM 0.25.0 | pending |
| `Gemma2ForCausalLM` | google/gemma-2-2b-it | near-tie 48/48 vs vLLM 0.25.0 | pending |
| `Gemma3ForCausalLM` | google/gemma-3-1b-it | strict 48/48 vs vLLM 0.25.0 | pending |
| `Gemma4ForConditionalGeneration` | Gemma-4 multimodal (unsloth/gemma-4-E4B-it) | text strict, image mm near-tie; audio pending | pending |
| `Gemma4UnifiedForConditionalGeneration` | Gemma-4 "unified" HF export (google/gemma-4-12B-it), no-PLE dense layout | shares the Gemma-4 text+mm forward; loads on the same factory (contributor #140); no separate oracle gate for this arch name yet | pending |
| `GraniteForCausalLM` | ibm-granite/granite-3.3-2b-instruct | near-tie 16/16 vs vLLM 0.25.0 | pending |
| `StableLmForCausalLM` | stabilityai/stablelm-2-1_6b | near-tie 16/16 vs vLLM 0.25.0 | pending |
| `MiniCPMForCausalLM` | openbmb/MiniCPM-2B-sft-bf16 | strict 16/16 vs vLLM 0.25.0 | pending |
| `MiniCPM3ForCausalLM` | openbmb/MiniCPM3-4B (MLA) | near-tie 16/16 vs vLLM 0.25.0 | pending |
| `Olmo2ForCausalLM`, `Olmo3ForCausalLM` | allenai/OLMo-2-0425-1B; OLMo-3 (Olmo2 factory alias) | OLMo-2 strict 16/16; OLMo-3 oracle-blocked (vLLM 0.25.0 cannot build it) | pending |
| `DeepseekV2ForCausalLM` | DeepSeek-V2-Lite (MLA) | strict 8/8 vs vLLM 0.25.0 | speed short, attributed |
| `DeepseekV4ForCausalLM` | DeepSeek-V4-Flash GGUF (ds4 q2-imatrix, UD-IQ2) | coherent near-tie vs ds4 oracle (vLLM cannot fit one GB10) | decode beats ds4 1.144x, default on, via the `deepseek-v4-gen` CLI; the registered engine forward is a W3 stub (`ARCH-ONE-SURFACE` fold) |
| `Glm4ForCausalLM` | GLM-4-9B-0414 | near-tie 16/16 vs vLLM 0.25.0 | pending |
| `Glm4MoeLiteForCausalLM` | zai-org/GLM-4.7-Flash (31.2B, MLA MoE) | near-tie 8/8 vs vLLM 0.25.0 | pending |
| `LagunaForCausalLM` | poolside/Laguna-S-2.1-NVFP4, GGUF-Q4_K, Laguna-XS | byte-exact near-tie (distributional vs vLLM) | vLLM parity+ 1.03x, default on, via the `laguna-gen` CLI; the registered engine forward VT_CHECKs non-bf16 (`ARCH-ONE-SURFACE` fold) |
| `KimiLinearForCausalLM` | Kimi-Linear-48B-A3B (KDA + NoPE-MLA + MoE) | **Folded onto the shared paged runner (ROW 7 §21, #122): engine==CLI 128/128 byte-identical; vs golden 122/128 (the intrinsic near-tie profile); FA2 paged MLA default-ON; SACRED post-fold green** | Served via `vllm_engine_load` + `vllm_complete_tokens` (ABI v13); server 19.0 tok/s wall vs vLLM ~21 (~0.90×), speed residual open |
| `KimiK3ForConditionalGeneration` | Kimi-K3 (2.8T MoE) | scaffold: registry+config+enumeration gated, forward refuses | HW-infeasible (~1.56 TB); no run |
| `Dots3NoteForCausalLM` | `dots-studio/dots3-note-prev` @`1e1e7b0c` (280B-A16B multimodal MoE, ~576 GB bf16; the `-fp8` sibling is ~290 GB). Headers only — no tensor byte downloaded | W1+W2 scaffold: registry + config gated off the REAL released `config.json`, with one assertion per §4 config trap (ungrouped 1/1 router, GPT-J indexer RoPE, one nextn layer, the two LoRA rescales, the sliding theta); name map accounted **38006/38006** over the WHOLE released index — 35381 language, 2195 vision, 430 audio, with the two tower files carried as named W6/W7 deferrals rather than dropped; load, GGUF and forward all REFUSE BY NAME | **No oracle, on any host we own** (~290 GB fp8 against a 122 GiB ceiling), so NO number is claimable on any axis and the e2e gate is an open gap by construction ([spec](../.agents/specs/dots3-note.md) §6.4, #699) |
| `NemotronHForCausalLM` | Nemotron-3.5-Lightning-30B-A3B-NVFP4 (`nvidia` @`29f2d174`) | config+enumeration+KV-shape gated; hybrid forward COMPUTES; loader materializes 18487/18487 as SHIPPED; A3 e2e gate 96/96 `STRICT PASS` on GB10 at `0ea5d249f` (#1221); NO run against current `main` | **PAGED (#810 A2-P): K/V go to the runner's pages; conv+SSM rows carry at the metadata's state indices.** G-SAFE: `num_reqs <= 1`. Device `lm_head` (A2-Q2b), UNMEASURED. Owed: FP8 mamba (A2-Q1), MTP, GGUF |
| `MuseGlimmerForCausalLM` | real tensors, **bf16 depth 4/52 only**: 5 prefill argmax positions match a torch transcription of vllm#51655 and HF. GGUF full depth generates coherently (#347, #359) but is **NOT token-exact** | text forward + loader vs an fp32 reference, per-mechanism property tests, scaffold 11/11, GGUF gate 17/17. An ABSENT config key now takes the architecture's constant (#412): GGUF post-norms ran at 1e-5, not 1e-8 | no vLLM denominator (pin cannot load it); SECONDARY llama.cpp, same GGUF, GB10 CPU: prefill tie **0.997x**, decode 0.232x, RSS 1.92x (#333) |
| `MuseGlimmerForConditionalGeneration` | vision: **no reference run of any kind**; enumeration gated vs the released 30B index (1436/1436). Image/video need bf16 safetensors: `mmproj-kquant.gguf` is refused by name | perception encoder loaded and wired, so an image or video prompt runs; `perception_emb_norm` now armed by default (#405). Reachability plus placeholder scatter only, no image or video correctness | not measurable; anchored to open vllm#51655 |
| `LlamaModel` | landed tiny synthetic embedding fixture (engine path == direct pooler path, identical vectors; f64 LAST+normalize reference); real checkpoint (e5-mistral class) is a NAMED residual | pooling/embed only, text paths refuse by task; `vllm_embed` + `/v1/embeddings` | n/a (CPU correctness-grade embeddings) |
| `ParakeetForCTC`, `ParakeetForRNNT`, `ParakeetForTDT` | nvidia/parakeet-ctc-0.6b/-1.1b, -rnnt-0.6b, -tdt-0.6b-v3 (transcribed, ids exact vs HF `generate()`, P4/P6 2026-08-07; not retained) + committed synthetic fold fixture | ASR transcription-only (`SupportsTranscription` mirror; text paths refuse by task); fold gate byte-identical to the pre-refactor pipeline | n/a (CPU correctness-grade ASR via `vllm_transcribe` + `/v1/audio/transcriptions`) |
| `CohereForCausalLM` | Command-R / Cohere (and Cohere2) | scaffold: W0 tiny-random oracle run-verified; real-checkpoint gate blocked | no run |
<!-- supported-arch-table:end -->

### Standalone and non-registered lanes

These run through dedicated forwards, not the `REGISTER_VLLM_MODEL` registry, so
they sit outside the gated list above. One caveat the LTX-2.5 row is too narrow
to carry: its text tower's prompt tokenization mirrors upstream only while the
checkpoint's tokenizer `post_processor` adds nothing. The shipped one is MEASURED
empty, so this port's plain encode plus an explicit BOS prepend matches
upstream's `add_special_tokens=True` today; a checkpoint with a non-empty
`post_processor` would tokenize differently here, and `Ltx2TokenizeGemmaPrompt`
in `ltx2_text_encoder.cpp` is the call that would have to change.

| Lane | Tested checkpoint(s) | Correctness gate | Speed vs reference |
|---|---|---|---|
| Voxtral audio (`VoxtralForConditionalGeneration`) | Voxtral-Mini-3B-2507 | near-tie-robust 16/16 vs vLLM 0.25.0 | decode 0.97x (beats vLLM); encoder FORWARD 15.90x of vLLM's whole TTFT (pin 46.02 ms), or 2.89x with opt-in `VT_WHISPER_ENC_FA2=1` (costs 3 near-tie divergences vs 0). Not a TTFT ratio. Pending |
| Whisper audio encoder | openai/whisper-small; whisper-large-v3 (Voxtral cfg) | encoder tower 77/77; large-v3 tower 203/203 | pending |
| MiniMax-H3 DiT (`MiniMaxH3DiTModel`, vllm-omni lane) | MiniMax-H3 (33.1B video+audio) | portable 79/79; all three modalities COHERENT on Q4_K_M (§8.20); PRUNED ckpts run, Q8_0 seam 0.9941 (§8.21); ref2va grid was NVFP4 quant error, §8.9 REFUTED; GGUF/NVFP4/bf16 shards stream | FP4/Marlin landed; speed pending; no bf16 render yet. Render from the Q4_K_M GGUF, not the NVFP4 arm. Krea 2 text-to-image (roadmap C11) is scoped to reuse these DiT seams |
| LTX-2.5 DiT (`LTX2VideoTransformer3DModel`, Lightricks lane) | LTX-2.5 (21.00B video+audio) | `SPIKE`. DiT, VAEs+ENCs, cond, pipeline, quant loaders gated, reduced dims. Prompt AdaLN host+dev; Gemma-4->xattn FIXTURE-gated. Img chain PPM->resize->encode->place->noise. Temporal x2 ups DRIVEN. Render OWED | `ltx-2.5`/`ltx2-gen`. NVFP4 ~29 GB/GB10, FP8 ~44, bf16 42.0; +24 GB tower. FP8/torchao/NVFP4/**bf16**; kf abs-pos ported; ALL 3 load, NO `allow_unported`. IMG+LAST kf `crf=0`, A2V WAV+LoRA; DiffVAE/ref refused. PENDING |
| MiniMax-Music3 (`MiniMaxMusic3ForConditionalGeneration`, diffusers lane) | MiniMax-Music3 (8.6B Qwen3 LLM + 0.646B RVQ decoder + 2.4B fp32 DiT + DAC Flow-VAE); diffusers arm, ~28.5 GB | `ACTIVE`. Loader 1413/1413; AR, acoustic and the 8.6B LM forward all gated vs real weights; `SpeechRegistry` + `vllm_speech_*` v21 + `/v1/audio/speech`; GGUF Q4_K depth decoder value-gated. HTTP request OBSERVED (#852) | No reference number. Host kernels multi-core, same song bytes (§12). PARTIAL device arm (#672): 8.6B LM + 2.4B fp32 DiT staged once (§14); rest host. Denominator SGLang-Omni production |
| LTX-2.5 DFR base + generated keyframe slots | LTX-2.5 (21.00B video+audio) | gated vs EXECUTED upstream `dfr_layout` + 3 `dfr_pipeline` helpers @ `fd4ded7f` (`test_ltx2_dfr` 11/11, 652 assertions); canvas, tiles, stitch, carry-forward as EXACT index vectors, since each defect is plausible| `--pipeline-kind dfr` plus `--lora`, which is now REQUIRED (#1445, both stages); NO `keyframe_slot_sft` base is published either, so the arm is REFUSED in practice. Canvas PADS 9->25, trims back; x8-grid slots MARKED, read before trim. ROUNDS below; detail LoRA (#975) refused|
| LTX-2.5 DFR temporal rounds (`temporal_upsample_rounds`) | LTX-2.5 (21.00B video+audio) | 7 `test_ltx2_video` cases via `LoadVideoEngine`/`Generate`; deleting the upsampler call site reds the reachability counter while frame counts stay green; the four per-tile guarantees are red-first by mutation | `--temporal-upsample-rounds` + `--temporal-upsampler`. FIRST caller of the temporal x2 ups. `(n-1)*2**r+1` frames; cond fps capped 60, playback not. Tiles clamp to segments; unclamped arm ungated (#1493). Fixture-gated |
| LTX-2.5 checkpoint class | LTX-2.5 DiT | 8 load-path cases; deleting validation reds 4 engine cases | `--checkpoint-class`: `full`, `distilled`, or `keyframe_slot_sft`. Required except for `dmd2`; declarations are checked, but matching bf16 headers prevent automatic detection (#1137) |
| LTX-2.5 tiled + streaming Conv VAE decode | LTX-2.5 video VAE | gated vs executed upstream `ltx_core` @ `fd4ded7f` (`test_ltx2_tiling` 10/10, 915 assertions); one-tile and untiled-spatial controls BIT-EXACT vs untiled on both causality arms; an untiled frames axis is REFUSED | Streams temporal chunks through upstream's AUTO layout (768/64 px, 80/24 frames); above one tile the pixel volume is never materialized. NO-OP below 768px and 81 frames; 81-120 IS tiled, differing 6.70% of range |
| LTX-2.5 Conv VAE decode arithmetic width | LTX-2.5 video VAE | `test_ltx2_vae` "the decode's convolution accumulates in f32", entering through `Ltx2VideoDecodeStreaming`; widening the accumulator to `double`, or deleting the production call site, each turns it RED | **f32**, the width `F.conv3d` uses at f32 AND bf16 (MEASURED). Was f64 at 8 sites ([#1008](https://github.com/mudler/vllm.cpp/issues/1008)). Conv sums BLOCKED per input channel, as torch's. STORAGE stays f32; bf16 owed |
| LTX-2.5 Conv VAE decode threading | LTX-2.5 video VAE | `test_ltx2_vae` "the decode DISPATCHES its convolutions to the CPU threadpool" and "...BIT-IDENTICAL across thread counts", through `Ltx2VideoDecodeStreaming`; 34 golden margins UNCHANGED; TSan clean | **Parallel** over CONV output lines via `vt::cpu::ParallelForRows` ([#1009](https://github.com/mudler/vllm.cpp/issues/1009)). ~9x at 16-20 workers, contended box, 21-23% spread. Bit-identical at any count |
| LTX-2.5 Conv VAE decode DEVICE arm | LTX-2.5 video VAE | `test_ops_conv3d` 4/4 vs an independent scalar reference; `test_diffusion_device_seam` runs a non-CPU dispatch and requires byte-identical pixels | Routes through the new `vt::Conv3d` op on the queue the engine resolved at load ([#1007](https://github.com/mudler/vllm.cpp/issues/1007)). NO GPU has run the CUDA arm (#1452); other stages stay host-side (#1451) |
| LTX-2.5 retake (`RetakePipeline`, regenerate a time window) | LTX-2.5 DiT + video VAE encoder | `test_ltx2_retake` 4/4 (69 assertions) and 4 `test_ltx2_video` cases entering through `Generate`; mask, conform and the four-way plan pinned to upstream `fd4ded7f` | `--pipeline-kind retake` on `ltx2-gen`. Source is a `frame_%06d.ppm` DIRECTORY; a container is REFUSED (no demuxer). Geometry comes from the clip. A folder has no audio, so the soundtrack is generated |
| LTX-2.5 text-to-audio (`T2AOneStagePipeline`) | LTX-2.5 DiT + audio VAE, no video VAE | `test_ltx2_video`'s `ltx2 t2a:` cases, entering through `Generate`; 18 mutations, 17 DETECTED (four by review of a conditional-only #1039 gate) and the 18th proven an identity, not a blind spot | `--pipeline-kind t2a_one_stage`. NO picture: 0 frames, no mux argv. The only AUDIO-ONLY guided arm (CFG + STG, 3 forwards/step), so it needs a text tower. CPU only; the device forward is refused by name |
| LTX-2.5 HQ preset (`TI2VidTwoStagesHQPipeline`, `res_2s` sampler) | LTX-2.5 DiT | 6 `test_ltx2_pipeline` cases + 2 `test_ltx2_video` cases through `Generate`, vs UPSTREAM'S OWN loop run at `fd4ded7f`: video latents BIT-EXACT on 3 of 5 fixtures, 1 ulp on 2. 20 mutations, 18 DETECTED | `--pipeline-kind res2s_two_stage` plus `--lora`, which is now REQUIRED (#1445, both stages, at ONE strength rather than upstream's 0.25/0.5 — #1144). 2.5 only. TWO denoiser calls per step plus a terminal one, and stage 1 is GUIDED at cfg 3.0 / modality 3.0, so 15 + 3 steps is 38 calls and 100 forwards. The preset IS the sampler |
| LTX-2.5 T2A guidance space | LTX-2.5 DiT (T2A arm) | `test_ltx2_video` "the guider is handed x0 predictions" through `Generate`, on all 3 arms plus the guider output and the Euler input; a seam case puts the two spaces 1.5e-07 apart at rescale 0 and 0.352 at 0.7 | Combines **denoised (x0)**, mirroring `X0Model` (`model.py:590-604`). Was velocity space, which agrees only at rescale 0 ([#1039](https://github.com/mudler/vllm.cpp/issues/1039)) |
| LTX-2.5 VIDEO guidance | LTX-2.5 DiT, joint video+audio | `test_ltx2_video`'s `ltx2 one_stage:` cases through `Generate`; all FOUR arms carry the x0 invariant and the guider output replays EXACTLY | `--pipeline-kind one_stage` runs `_guided_denoise`: 4 forwards/step, combined per modality in **x0**. Was ONE unguided forward, every `video_guidance` field dead ([#1092](https://github.com/mudler/vllm.cpp/issues/1092)) |
| LTX-2.5 cross-attention perturbations | LTX-2.5 DiT | `test_ltx2_video` and `test_ltx2_device` each gate one direction ALONE, on a forward where the other stream is PRESENT but DISABLED. Swapping the two flags is RED on both arms | `SKIP_A2V_CROSS_ATTN` / `SKIP_V2A_CROSS_ATTN`, which `modality_scale = 3.0` selects on every video row. On the DEVICE forward too since 2026-08-19 ([#1092](https://github.com/mudler/vllm.cpp/issues/1092)) |
| LTX-2.5 audio-to-video (`A2VidPipelineTwoStage`) | LTX-2.5 DiT + audio VAE encoder + spatial upsampler | `test_ltx2_pipeline` and `test_ltx2_video`'s `ltx2 a2vid:` cases through `LoadVideoEngine`+`Generate`; the take's latent is bit-identical across SEEDS and moves with the WINDOW | `--pipeline-kind a2vid_two_stage`. Guided half-res stage 1, DERIVED schedule, plain Euler; distilled 3-sigma stage 2. `--audio-path` and `--lora` REQUIRED; the distilled adapter rides stage 2 ALONE (#1118) |
| LTX-2.5 keyframe interpolation (`KeyframeInterpolationPipeline`) | LTX-2.5 DiT + spatial upsampler | `ltx2 keyframe:` cases in `test_ltx2_pipeline` / `test_ltx2_video` via `LoadVideoEngine`+`Generate`: frame 0 APPENDS against a `ti2vid_two_stage` control, the x0 invariant on four arms, the 4096 anchor | `--pipeline-kind keyframe_interpolation`. No frame-0 special case, so `--first-frame` is guidance to interpolate FROM; stage 2's audio leaves. `--lora` REQUIRED. `--last-frame` new (#1191). CPU fixtures |
| LTX-2.5 two-stage text/image-to-video (`TI2VidTwoStagesPipeline`) | LTX-2.5 DiT + spatial upsampler | `test_ltx2_pipeline` and `test_ltx2_video` `ltx2 ti2vid:` cases through `LoadVideoEngine`+`Generate`; the x0 invariant on all FOUR arms, and the 4096 anchor read at two geometries against a res_2s control that moves | `--pipeline-kind ti2vid_two_stage`. Guided half-res stage 1 on the UNADAPTED model, plain Euler; distilled 3-sigma stage 2. `--lora` REQUIRED, no `--audio-path`; stage 1's audio leaves. CPU fixtures, Full-model run owed |
| LTX-2.5 guidance knobs | LTX-2.5 request surface | `test_ltx2_video` renders with an override and refuses one on a fixed recipe | Seven video/audio guider extras mirroring `default_1_stage_arg_parser`, plus a negative embeds pair for a tower-less engine. Refused whole on `distilled_two_stage` and `retake`, whose guidance is distilled in |
| MTP speculator | Qwen3.6-27B, Qwen3.6-35B-A3B | token-identical to vLLM `mtp` at c1 | ~4% faster c1; +16% output tput (MoE) |
| MTP speculation DEPTH (`num_speculative_tokens` > 1) | Qwen3.5/3.6 `mtp.*` heads | k=1..4 through the loader, greedy tokens unmoved, two witnesses per arm: the draft decode forwards the propose RAN, and whether the DELIVERED draft row varied with depth. `test_mtp_depth` 5/5, 63 assertions | Default stays k=1. NO speed claim at k>1. Drafts are proposed and verified, never ACCEPTED, and neither witness proves per-column provenance. Both await the owed DGX gate (#81) |
| DFlash block-diffusion | Qwen3 (DFlash draft) | near-tie e2e 27/27 vs vLLM | 2.9x over spec-off, 1.003x vs vLLM DFlash-on |
| DFlash2 block-diffusion (dynamic conv + candidate selector) | Qwen3 DFlash2 draft, safetensors or GGUF (bf16 / Q8_0 / Q4_K_M) | Gated against vLLM: 4/4 token-exact, 45/47 draft blocks identical, acceptance identical per prompt. All 7 DFlash2 suites green on `sm_121a`, zero CUDA skips | GREEDY only. Speed 0.8017x vLLM: RECORDED, no floor, NOT a pass (#1562). A GGUF draft is dequantized to bf16 at load ([#1314](https://github.com/mudler/vllm.cpp/issues/1314)) |
| Async scheduling × speculative decoding | any Eagle-type speculator (`mtp`, `dflash`/DFlash2, `dspark`) | `test_mtp_depth` W7 cases: a spec engine resolves async ON and emits the sync scheduler's exact tokens through both engine fronts (depth-1 and the depth-2 batch queue); `test_engine_core_proc` pins the -1-placeholder / worker-fill contract | Mirrors vLLM's polarity (async disabled only OUTSIDE the Eagle-type family): drafts ride as `-1` placeholders the worker fills from its own propose; host `ngram` and `draft_model` stay synchronous. `VT_ASYNC_SCHED=0` rolls back. Spec steps keep the host sampler (device-resident spec sampling owed); the GPU TPOT A/B owed ([#1824](https://github.com/mudler/vllm.cpp/issues/1824)) |
| DFlash/DFlash2 shared `lm_head` kept PACKED | a DFlash or DFlash2 draft off an NVFP4 safetensors target | `test_qwen3_dflash2_draft` 36/36 (353): block logits BITWISE equal to `Qwen3_5MTPModel::ComputeLogits` on the same packed head, and `FromModelDir` loads and drafts off one | Widening a head stays refused by name: GGUF `output.weight`, FP8, W4A4. `VT_LMHEAD_FP4=0` rolls back to the refusal. DSpark and the CUDA arm owed ([#1628](https://github.com/mudler/vllm.cpp/issues/1628)) |
| DeepSeek-V4 MTP | DeepSeek-V4-Flash (nextn head) | lossless 5/5; real-model weight-blocked | pending |

### Inventoried but blocked

Enumerated in `.agents/model-matrix.md`, not registered, no runnable GB10 gate:

| Architecture | Model | Why blocked |
|---|---|---|
| `DeepseekV3ForCausalLM`, `DeepseekV32ForCausalLM` | DeepSeek-V3 / V3.2 | 671B, ~642 GiB fp8 vs 119 GiB unified; V3.2 also DSA-indexer dep-blocked |
| `GlmMoeDsaForCausalLM` | GLM-5 (DSA) | ~1404 GiB bf16; dep-blocked (GLM-5.x is DeepSeek-V3.2 verbatim) |
| `MiniMaxM2ForCausalLM` | MiniMax-M2 | ~230B, ~428 GiB bf16, ~4x over the unified pool |
| `Dots3NoteMTPModel` | dots3-note nextn head (the target arch `Dots3NoteForCausalLM` IS registered — see the supported table above) | W10 owns it and it is deliberately NOT registered: a speculator that cannot propose makes the engine accept a speculative config it then dies on mid-run. The checkpoint ships exactly one nextn layer. Blocked behind the target row: no oracle runs here, ~290 GB fp8 against a 122 GiB ceiling, so NO number is claimable on any axis ([spec](../.agents/specs/dots3-note.md), #699) |

27 of the 37 registered text-generation architectures carry a passing
correctness gate today; the rest are honestly marked scaffold or blocked above.
(The 41 registered total also covers 3 Parakeet ASR entry points and the
`LlamaModel` embedding arch, which are not text generation.)
vLLM registers 130+ text architectures, so this is a curated, gated subset, not
a breadth claim. The first EMBEDDING architecture is registered and live
(`LlamaModel`, task=embed, LAST pooling, the as_embedding_model mirror, gated
on the committed fixture); reranking/classify models are not yet registered.

## Multimodal

| Input | vllm.cpp | vLLM | SGLang | llama.cpp |
|---|---|---|---|---|
| Image | ✅ correctness-gated | ✅ | ✅ | ◐ |
| Video | ✅ correctness-gated | ✅ | ✅ | ☐ |
| Audio | ✅ correctness-gated | ✅ | ◐ | ◐ |
| Video+audio GENERATION (MiniMax-H3 DiT, LTX-2.5 DiT) | ◐ H3: all three modalities COHERENT on Q4_K_M (t2va, fl2va, ref2va; §8.20); the NVFP4 arm carries the patch grid; GGUF/NVFP4/bf16 loaders, pruned too (§8.21). LTX-2.5: a second lane, `SPIKE`, gated at reduced dims | ✅ H3 (vllm-omni, BF16-only, no quantized arm); LTX-2.5 only through the generic diffusers adapter, no native recipe ([vllm-omni#6066](https://github.com/vllm-project/vllm-omni/issues/6066)) | ☐ | ☐ |
| Speech / audio GENERATION (TTS, vLLM-Omni lane) | ◐ IndexTTS-2.5: vllm_synthesize renders TEXT to AUDIO on real weights, and the reference clip CONDITIONS it -- CAMPPlus speaker vector into the talker's row 0 and the S2Mel style; two clips give different audio (rms 0.0064 vs rms 0.0956), same clip twice is bit-identical. STRUCTURE only: emotion conditioning is excluded and vLLM-Omni is unpinned, so nothing here is a correctness claim (#634, #633) | ✅ (vllm-omni: MOSS-TTS, Qwen3-TTS, Higgs Audio v3, Voxtral TTS, IndexTTS-2.5) | not assessed | not assessed |
| MUSIC generation (MiniMax-Music3) | ✓ every stage gated; an HTTP request observed e2e over a REAL SOCKET against a MUSIC-ONLY server (#852, #672, [spec](../.agents/specs/minimax-music3.md) §10); adjacent caption italics match upstream (#1083) | ☐ absent from the pin, from vLLM `main` and from `vllm-omni` | ◐ SGLang-Omni serves the NATIVE layout; its 32 kHz resample and batching are OWED | ☐ |
| Multimodal over the OpenAI server | ◐ image request path wired, forward pending | ✅ | ✅ | ◐ |
| Per-modality input LIMITS (`--limit-mm-per-prompt`, `--language-model-only`) | ✅ limits, refusals, and the TOWER SKIP: a tower whose every modality sits at 0 is constructed but never loaded. Byte saving measured on **Qwen3-VL-4B-Instruct only**: 1.542 GiB of host RSS at load, `--device cpu`, threshold MET on both pairs ([#607](https://github.com/mudler/vllm.cpp/issues/607)). Not a general or a VRAM claim, about half of it is our own bf16→f32 widening ([#1359](https://github.com/mudler/vllm.cpp/issues/1359)), and `muse-glimmer-30b` is still unmeasured ([benchmark](benchmarks/memory.md)) | ✅ | ☐ | ☐ |

Image, video and audio are correct through the CLI and library. Over the HTTP
API the image **request** path is wired end to end (`ROAD-V1-MM` W1-W3): the
production server attaches the seam at `server_main.cpp:826`. Two residuals keep
it from ✅: the model runner has no mm-forward consuming `Request.mm_features`,
and no image codec is vendored (raw RGB only). Video, audio and multi-image over
HTTP are not started. Audio **in** is gated. Audio **out** has a surface now
(`/v1/audio/speech`, `vllm_speech_*` v20), but no family renders from a prompt:
both refuse, naming what is missing.

## Speculative decoding

| Speculator | vllm.cpp | vLLM | SGLang |
|---|---|---|---|
| MTP (multi-token prediction) | ✅ token-identical, ~4% faster at c1, depth `k` configurable with default 1 | ✅ | ✅ |
| Draft model | ◐ CPU brick | ✅ | ✅ |
| Medusa | ☐ spike only | ✅ | ✅ |
| EAGLE / EAGLE3 | ☐ | ✅ | ✅ |
| DFlash block diffusion | ✅ 2.9x over spec-off, at/above vLLM DFlash-on | ✅ | ☐ |
| DFlash2 block diffusion (a SECOND DFlash architecture, not a change to DFlash) | ◐ safetensors AND GGUF drafts DRAFT (bf16, Q8_0, mixed Q4_K_M: 45 Q4_K + 4 Q6_K), greedy, 0.8017x vLLM RECORDED not a pass (#1562), no published artifact LOADED yet ([spec](../.agents/specs/dflash2-spec-decode.md)) | ✅ BEYOND-PIN, [vllm#52816](https://github.com/vllm-project/vllm/pull/52816) | ☐ |
| n-gram / prompt lookup | ✅ 27B 5/5 strict vs vLLM | ✅ | ✅ |
| DSpark (semi-autoregressive block drafter) | ◐ **both gate models** ([spec](../.agents/specs/dspark-spec-decode.md)): token-identical to spec-off; T=1+k verify CAPTURED. Cross-engine ratio UNSETTLED (**0.834x** matched-and-warm); Marlin MoE CLEARED as the residual | ✅ | ◐ |
| DSpark draft routing (which draft the loader takes) | ◐ `Qwen3DSparkModel`, `Gemma4DSparkModel` and (BEYOND-PIN, vllm#52197) `DSparkDraftModel` + `qwen3` take the Qwen3 lane; DeepSeek-V4 DSpark is REFUSED by name ([spec](../.agents/specs/dspark-qwen3-routing.md)) | ◐ at the pinned `555967922` that pair routes to DeepSeek-V4; ✅ only since vllm#52197, merged 2026-08-17 | not assessed |
| Other methods (ngram-gpu, suffix, custom-class, dynamic-k, mlp-speculator) | ☐ inventoried | ✅ | ◐ |

## Structured output and tool calling

| Feature | vllm.cpp | vLLM | SGLang | llama.cpp |
|---|---|---|---|---|
| JSON schema constrained decode | ✅ | ✅ | ✅ | ✅ |
| Regex constrained decode | ✅ | ✅ | ✅ | ✅ |
| GBNF grammars | ✅ | ☐ | ☐ | ✅ |
| xgrammar backend | ✅ | ✅ | ✅ | ☐ |
| Jump-forward decoding | ✅ opt-in | ☐ | ✅ | ☐ |
| Tool-call parsers | ✅ 38 families | ✅ | ✅ | ◐ |
| Reasoning-content parsers | ✅ 12 | ✅ | ✅ | ☐ |
| Muse Glimmer ATEM parsers (`muse_glimmer`) | ◐ UNIT-GATED ON STRINGS; **CHANNEL SCOPING FAILS AT SERVER DEFAULTS**: no `adjust_request` seam, so `skip_special_tokens: true` strips the framing. OPEN GAP, [spec](../.agents/specs/muse-glimmer.md) §6.7 | ✅ | ☐ | ☐ |
| Custom logits processors | ◐ CPU-verified | ✅ | ✅ | ☐ |

## Backends and hardware

| Backend | vllm.cpp | vLLM | SGLang | llama.cpp |
|---|---|---|---|---|
| CUDA | ✅ sm_80 to sm_121a | ✅ | ✅ | ✅ |
| CPU (x86, Arm i8mm; A76 assembly correct/default, llama speed gate open, and the closed 20-core floor ran a SUPERSEDED fork denominator rather than the stock `b10451` pin, re-take owed #1003) | ✅ `CPU_ATTN` registered (#1371/#1392, [spec](../.agents/specs/attn-validate-configuration.md)) | ◐ | ☐ | ✅ |
| Metal (Apple Silicon) | ✅ builds under Apple Clang with project warnings promoted to errors, the Qwen3.5 MoE loader included; its layout-refusal path uses the same messages and behavior on every platform (#1054) | ☐ | ☐ | ✅ |
| Vulkan | ◐ | ☐ | ☐ | ✅ |
| ROCm | W0: 5 gfx archs; dense/GDN all-native; 0.8B dispatch fixed. **M4:** Qwen3-0.6B/3.5-0.8B 16/16 (#41). **M3:** `ROCM_ATTN` registered (#1056/#1065, [spec](../.agents/specs/rocm-attn-backend.md)). CPU parity open (#269) | 49 registered ops: full GDN, MoE combine/gate, keep-quant GEMM; ctest-green gfx1151/1103/1100/1201/1200 ([#41](https://github.com/mudler/vllm.cpp/issues/41)). APU managed allocation is unverified. [ROCm guide](ROCM.md) | ✅ | ✅ |
| XPU / TPU | ☐ | ✅ | ◐ | ☐ |
| Tenstorrent Blackhole | ◐ `ACTIVE`, OPT-125m 6/6; Qwen3-0.6B wired; Mistral-7B-v0.3 16/16 on P150 ([spec](../.agents/specs/tenstorrent-mistral.md)). 16x16 rerun and residual-RMS owed ([spec](../.agents/specs/tenstorrent-backend.md)) | ✅ | ☐ | ☐ |
| Tenstorrent host-free decode | ◐ DEFAULT since #1604 (`0` opts out): no per-step host readback; 2.1x default-leg tok/s; both golden pairs re-adjudicated, both paged gates 16/16. Capture opt-in only (#1625 hang); async off (#1627) | ☐ | ☐ | ☐ |

CUDA runtime-verified on GB10 (sm_121a), Jetson Thor (sm_110) and Jetson AGX
Orin (sm_87). sm_110 has no CUTLASS FP4 tensor-core kernels and no `fp4-mma`,
so it stays a correctness venue for those; the one fast path it does get is the
vendored **Marlin NVFP4 W4A16** GEMM, enabled since 2026-08-11 and validated on
Thor silicon (8.0x-29.0x per GEMM at M=1, e2e 16.61 to 81.63 tok/s at c=1 on
Qwen3-1.7B-NVFP4A16). That is a kernel-level result, not a token-exact
model-level gate.

Vulkan **runs a model end to end**: `opt-125m` greedy is STRICT token-exact,
6/6 prompts vs the vLLM 0.25.0 oracle, every op dispatched natively with **zero
provider declines**. Qwen3.6-27B runs too, both GDN recurrences and the fused
attention preamble native: **decode 4.36 tok/s vs llama.cpp's 4.35, parity met
narrowly**, denominator SUPERSEDED (#1003), and prefill **21.5x**, a SELF-ratio
(GB10). A load keeps **one** copy of the weights, not two, and is 1.54x faster
warm: 27B peak RSS 100.8 -> **53.4 GiB**. Still partial at 25 natively
registered ops of 112 (8 GDN), the rest on the portable CPU tier; quant/MoE/MLA
have none.
Build with `-DVLLM_CPP_VULKAN=ON`; off by default.

## Serving, API and operations

| Feature | vllm.cpp | vLLM | SGLang | llama.cpp |
|---|---|---|---|---|
| OpenAI-compatible `/v1/chat/completions` | ✅ | ✅ | ✅ | ✅ |
| Streaming (SSE) | ✅ | ✅ | ✅ | ✅ |
| Offline batch API | ✅ | ✅ | ◐ | ☐ |
| Prometheus metrics | ✅ live per-step values on the serving path, not just the catalog; async detach and server teardown wait for the final fold | ✅ | ✅ | ◐ |
| Container images | ◐ `cuda`/`vulkan`/`cpu` lanes build and gate from one Dockerfile (amd64+arm64, `ENTRYPOINT vllm-server`, ffmpeg included); **nothing published to GHCR yet** | ✅ | ✅ | ✅ |
| Graceful shutdown on `SIGTERM` | ✅ clean exit in 0.25 s, including as container PID 1 (#312) | ✅ | ✅ | ✅ |
| Plugin / out-of-tree model registration | ✅ in-tree factory `DONE` + plugin seam | ✅ | ◐ | ☐ |
| Fetch a checkpoint from Hugging Face by name | ◐ `vllm-server --model org/repo[:QUANT]` fetches into the HF cache. HTTPS via system OpenSSL (`VLLM_CPP_HF_DOWNLOAD`, ON). The musl-static archive has no TLS. `vllm-cli` and the C ABI take a local path (#1280) | ✅ | ✅ | ✅ |
| A registered forward opens its OWN model type, not whatever it was handed | ✅ all 35 entry points establish the concrete type first and refuse a mismatch by name (#775, swept in [#847](https://github.com/mudler/vllm.cpp/issues/847)) | n/a | n/a | n/a |
| Multiple engines in one process (build, destroy, rebuild) | ✅ resident device state is owned by the weights, so a new engine never inherits a freed one's pointers | ✅ | ✅ | ✅ |
| LoRA adapters | ☐ CPU brick only | ✅ | ✅ | ✅ |
| Embedding / pooling endpoints | ◐ `/v1/embeddings` live (task=embed; score/rerank/classify pending) | ✅ | ✅ | ✅ |
| OpenAI video generation `/v1/videos` (Sora shape) | ✅ `model`/`size`/`seconds` aliases + `GET /{id}/content`; `input_reference` and `metadata` references condition the render; `--video-family` pins the family (default DETECT), `--video-extra K=V` carries family knobs | ◐ (vllm-omni, its own request shape) | ☐ | ☐ |
| OpenAI speech generation `/v1/audio/speech` (createSpeech shape) | ◐ route + ABI live, opt-in behind `--speech-model`; `lyrics` + `description` are extra named fields for a music family; 20 unsupported keys refused by name; every key read at the top level and under `extra_params` | ◐ (vllm-omni) | ☐ | ☐ |
| `logprobs` / `top_logprobs` / `prompt_logprobs` on both generate routes | ✅ generated-token logprobs on completions and chat; `prompt_logprobs` scores the prompt and reaches the client (per choice on completions, top-level on chat), with vLLM's three refusals. `-1` (whole vocabulary) is SERVED here — there is no separate `max_logprobs` cap. OpenAI `echo` still does not prepend the prompt to the payload ([#223](https://github.com/mudler/vllm.cpp/issues/223)) | ✅ | ✅ | ◐ |
| Flat C ABI for embedding in other languages | ✅ versioned | ☐ | ☐ | ✅ |
| Request-length bound before tokenization | ◐ `/tokenize` + both generate routes REFUSE a prompt above `max_model_len` x the longest vocabulary token, naming the limit, never truncating; other routes and the C ABI unbounded (#1541) | ◐ header block only; a prompt COUNT bound and an audio-upload byte bound, no prompt byte bound | ☐ | ☐ |

### C-ABI capability coverage <!-- abi-capability-table:begin -->
- Which capabilities an embedder drives through the flat C ABI (`include/vllm.h`, the only installed header), gated by `scripts/check-surface-coverage.py`: a `reachable` row names an entry point that exists; an `embedder-unreachable` row is tracked in `scripts/abi-capability-allowlist.txt` against its fold row (`ARCH-ONE-SURFACE`). The ABI is text-generation-complete; the one `embedder-unreachable` row (multimodal input) is the open capability gap.

| Capability | C-ABI surface | Embedder-reachable |
|---|---|---|
| Text completion (blocking + streaming) | `vllm_complete`, `vllm_complete_stream` | reachable |
| Pre-tokenized completion (token-id prompts, ABI v13) | `vllm_complete_tokens` | reachable |
| OpenAI chat (tools, streaming) | `vllm_chat`, `vllm_chat_stream` | reachable |
| Async request submission | `vllm_request_submit` | reachable |
| Structured output / grammars | `structured_json`, `structured_grammar` | reachable |
| Tool + reasoning parser selection | `tool_parser`, `reasoning_parser` | reachable |
| Speculative decoding config | `speculative_config` | reachable |
| Custom logits processor | `vllm_logits_processor` | reachable |
| Embeddings / pooling (task=embed) | `vllm_embed`, `vllm_embedding_result_free` (ABI v15; pooling checkpoints load via `vllm_engine_load`) | reachable |
| Audio transcription (Parakeet ASR) | `vllm_transcribe`, `vllm_transcription_params_default`, `vllm_transcription_free` | reachable |
| Video+audio generation (MiniMax-H3, LTX-2.5) | `vllm_video_engine_load`, `vllm_video_generate`, `vllm_video_result_free`, `vllm_video_mux_argv`, `vllm_video_engine_family` (ABI v18 family registry), `vllm_video_last_phase_log` (ABI v23 render phase table) | reachable |
| Explicit device selection (auto/cpu/cuda) | `device` field on `vllm_model_params` (ABI v14; 0=auto keeps the probe, explicit absent device fails loud) | reachable |
| Run the OpenAI server (server as a thin ABI client) | `vllm_server_main` (ABI v18) | reachable |
| Speech + music generation (MiniMax-Music3; the IndexTTS-2.5 seam) | `vllm_speech_engine_load`, `vllm_synthesize`, `vllm_speech_result_free`, `vllm_speech_engine_family`, `vllm_speech_engine_sample_rate`, `vllm_speech_engine_requires_reference_audio` (ABI v20) | reachable |
| Multimodal input (image/audio/video) | none | embedder-unreachable | <!-- abi-capability-table:end -->

## Parallelism and scale-out

Single-GPU today. Every mode below is scoped against one `vt::Communicator`
abstraction, and `world_size == 1` stays byte-identical.

| Mode | vllm.cpp | vLLM | SGLang |
|---|---|---|---|
| Tensor parallel (TP) | ◐ CPU-gated, no 2-GPU run; TP-W1 LANDED 2026-08-08 (rank-layout group table + per-rank handle); TP-W2..W4+W7 CPU-completable | ✅ | ✅ |
| Collective / process-group abstraction | ✅ CPU + NCCL transport | ✅ | ✅ |
| Pipeline parallel (PP) | ☐ spike written | ✅ | ✅ |
| Expert parallel (EP) + EPLB | ☐ spike written | ✅ | ✅ |
| Data parallel (DP) | ☐ spike written | ✅ | ✅ |
| Context parallel (PCP / DCP) | ☐ scoped | ✅ | ◐ |
| Multi-node | ☐ spike written | ✅ | ✅ |
| PD disaggregation | ☐ | ✅ | ✅ |

CPU elementwise GEMM (f32/f16/bf16) runs AVX2 and AVX-512 tiers on x86 where the CPU supports them (SSE2 before), selected by a runtime probe, and can take a transpose-free `[K,N]` weight path via an opt-in load-time repack (`VT_CPU_ELEM_KN_REPACK`, CPU only, default off). Byte-identical to the portable tier either way.

## Not supported yet

| Gap | State | Detail |
|---|---|---|
| Kimi-Linear-48B-A3B (KDA + NoPE-MLA + MoE hybrid) | **Runner fold LANDS (ROW 7 §21, #122): the ENGINE/SERVER surface serves Kimi at the 122/128 golden profile (engine==CLI 128/128); STRICT stays closed (intrinsic p7 near-tie)** | server 19.0 tok/s wall / CLI 18.9 vs vLLM ~21 (~0.90×), speed residual named (§21) |
| Muse Glimmer 30B (Meta) | Text gated at **reduced depth 4/52** only; vision wired but never reference-checked | [spec](../.agents/specs/muse-glimmer.md) / [#268](https://github.com/mudler/vllm.cpp/issues/268). Full depth, multi-step decode, image/video, server path and parser scoping open. vLLM speed OPEN GAP; llama.cpp bar #333 |
| LTX-2.5 AUTO duration (the duration head) | Brick ported, never constructed | `duration_head_path` is REFUSED by name rather than accepted-and-ignored ([#611](https://github.com/mudler/vllm.cpp/issues/611)); supplying a head cannot load one. Give `num_frames` or `duration` |
| LTX-2.5 arms a request CAN reach | Refused by name at the call site | The spatiotemporal latent upsampler (both flags set). Supplying that checkpoint names that arm, not the temporal one. The temporal-only x2 arm is ported, not refused |
| LTX-2.5 resolution | Off-grid sizes refused, naming the offending axis and a size you can actually pass; frames still round | `--width`/`--height` must divide 64 (two-stage) or 32 (one-stage), from the VAE factor times the phase downscale ([#919](https://github.com/mudler/vllm.cpp/issues/919)). `--frames` rounds to `8k + 1`. No size cap |
| LTX-2.5 arms nothing can request | Declared, not requestable | `int8-convrot` (ComfyUI-only), single-node multi-GPU, `BetaScheduler` (upstream selects no scheduler either). No flag or extra asks for these. `multishot` was RETIRED (absent upstream) and `kLoraFusion` too (now served) |
| Qwen3.8-27B quantized arms (Q4_K_M GGUF, its `clip` mmproj, the `unsloth` "NVFP4" = `mixed-precision`) | All three files ACCOUNTED against committed manifests (866 / 334 / 1968 names). **Q4_K_M RUNS on CPU and its token gate vs llama.cpp `b10451` FAILED** (2026-08-23): tokenizer exact 6/6, generation divergent 5/6, every divergence a rank-2 loss under 0.18 logits over 288 steps, so a precision difference and not a wiring defect ([evidence](bench-evidence/qwen38-27b-q4km-token-gate-20260823.md)). No speed or memory number is admissible. NVFP4 still un-run (#1632) | BF16 gated (#915). `--mmproj` (ABI v22) loads a `clip` GGUF; a load refuses a file with a tensor no loader reads. "NVFP4" @`7d6f8d4d`: W4A4 loads, FP8 refused ([spec](../.agents/specs/qwen38-27b-quant-arms.md)) |
| Qwen3.8-27B, the SECOND NVFP4 artifact (`r0b0tlab/...-MTP-sm121`, a ModelOpt checkpoint) | 2001 names ACCOUNTED per scheme against four committed manifests; LOADS, never RUN: no token gate (#1632) | @`36f717a2`: 208 per-tensor STATIC FP8 + 193 W4A16_NVFP4 modules, both halves load. Not the `unsloth` format ([spec](../.agents/specs/qwen38-27b-quant-arms.md)) |
| Multi-GPU execution | Hardware-blocked | TP proven equal to tp=1 on CPU; no 2-GPU box to run it |
| LoRA end to end | CPU brick landed | Unwired standalone; not usable through the server |
| Multimodal over HTTP | Image request path wired; forward + codec pending | `ROAD-V1-MM` W1-W3 landed. Open: no mm-forward on `Request.mm_features`; no image codec. Video/audio/multi-image now **refuse** with HTTP 400 rather than drop ([#686](https://github.com/mudler/vllm.cpp/issues/686)) |
| Reranking / classify models | Engine side only | Embeddings are LIVE (`LlamaModel`, `vllm_embed`, `/v1/embeddings`); the classify/score heads are landed ops with no registered arch |
| ROCm | W0 community-verified on 5 gfx archs; classic-dense and GDN-hybrid e2e run all-native; correctness gaps remain | 49 registered ops including the GDN state/conv/postconv/recurrence set, MoE combine/gate, and keep-quant expert GEMM; APU managed-allocation branch remains unverified. [ROCm guide](ROCM.md) |
| XPU, TPU | Not started | CUDA, CPU, Metal and Vulkan are the built backends |
| Custom logits processors on CUDA | Open, not root-caused | Segfaults in a CUDA build, 232/232 green on CPU |
| Memory budgeting (`ROAD-V1-MEM`, #83) | M1+M2 landed (absolute bytes) | `--kv-cache-memory` sizes the KV pool from an absolute byte budget (ABI v16, group-aware divisor); `--num-blocks` overrides; `--gpu-memory-utilization` needs the M3 profile run (dgx-gated). See `specs/kv-sizing.md` |
| Gemma4 MoE ROCm FP8 + SharedK-WMMA | Partial | Dual-GPU FP8 resident experts, SharedK-WMMA prefill (RDNA4); decode-graph and forward extract deferred. Env `VT_GEMMA4_*`/`VT_ATTN_*`, seam `test_gemma4_rocm_fp8_seams`. [spec](../.agents/specs/gemma4-rocm-fp8-moe.md) |

## How to read this page

A ✅ means the implementation has a named gate. For a model, that gate compares
the same workload with the pinned oracle. For an engine feature, the gate is a
named test in the tree. A ◐ means the path works only within the limits in its
table row.

The marks describe support, not speed or current ownership. See
[Status](../README.md#project-status) for lifecycle state and [Benchmarks](BENCHMARKS.md) for
performance. An inventoried row is not a supported feature.
