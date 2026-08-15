# roadmap_v1 completion punch-list — the plan of record for finishing the portfolio

*(records-only planning spec, 2026-07-27. Base `origin/main` `76d2b8ea`. READ-ONLY
analysis; no `src/`/`tests/` touched, SACRED byte-identical. This spec classifies
EVERY `ROAD-V1-*` portfolio row against the true matrix/ledger/state state, names
the HW/external-blocked bounding set, and gives the ranked reachable execution
order. It is a plan, not a support claim: nothing here advances any row's
lifecycle state.)*

## Method + hardware envelope

Every classification below is grounded in the six area matrices
([engine](../engine-matrix.md), [model](../model-matrix.md),
[kernel](../kernel-matrix.md), [backend](../backend-matrix.md),
[quantization](../quantization-matrix.md), [feature](../feature-matrix.md)),
the [roadmap portfolio table](../roadmap_v1.md), and the append-only
[git history](../../AGENTS.md)/[ledger](../parity-ledger.md). "DONE" requires merged code +
a passing gate; where a row ships a capability but the every-axis **speed** gate
or a **breadth** tail is open, the row is REACHABLE-INCOMPLETE, not DONE.

**Reachable HW** = one GB10/DGX-Spark GPU (sm_121a, 119 GiB unified; the vLLM
0.26.0.dev0 oracle box) + a CPU dev box + an Apple M4 Mac mini (Metal/MLX, 16 GiB,
small models only). **NOT available** (⇒ HW-BLOCKED, honesty-pass/build-only
ceiling): any AMD/ROCm, Intel/XPU, discrete-Vulkan, ANE, or multi-GPU board, and
any model that does not fit the 119 GiB unified pool.

## 1. Per-row classification (18 portfolio rows)

Legend: **DONE** (merged+gated) / **RI** (reachable-incomplete on GB10/CPU/M4) /
**HW** (hardware-blocked) / **EXT** (external-blocked: pin/dep/HF-token). Rows that
carry more than one class list the dominant one first.

| Row | Class | Delivered + gated (anchor) | Remaining to close the row |
|---|---|---|---|
| `ROAD-V1-A` perf/SGLang floor | **RI** | 27B **effective parity-or-better ratified** (two-grid totality 115/124: 110 pass-in-both + 5 coin-flip; residuals are the net-positive determinism tradeoff). `ENG-ASYNC-SCHED` DONE. | 35B every-axis closure (first binding 70/124, c4–c32 already win, c1/c2 residual — `BACKEND-GATE-CUDA-VLLM` PARTIAL); then the SGLang floor arms (`BACKEND-GATE-CUDA-SGLANG` BLOCKED on `SERVE-ASYNC-LLM` prod-ON; `-PREFIX` READY). |
| `ROAD-V1-MM` multimodal | **RI** (+EXT sub) | **Image + video STRICT token-exact 32/32** on Qwen3.6-27B and Qwen3-VL-4B; **audio e2e** on Voxtral-Mini-3B (near-tie-robust; decoder 48/48). Correctness is the user's #1 priority and it LANDED. SPEED progress: tower lever #1 CLOSED (2114→148 ms, faster than vLLM eager encode); decode lever #2 CLOSED 2026-07-27 (on-GPU greedy argmax + no embed round-trip; bit-exact; 27B decode NEUTRAL at parity, audio ~0.4% win — multimodal-speed.md §8); lever #3 FIRST BRICK 2026-07-27 (the 27B image+video decode now routes through the production `Qwen3_5DenseDecodeGraph` captured decode = GRAPH-CAPTURABLE, token-exact 32/32 held, NEUTRAL at the 27B bandwidth floor — multimodal-speed.md §9); lever #3 **W1 LANDED 2026-07-27** (new `VoxtralDecodeGraph` graph-captures the Voxtral audio decode; bit-exact 14/14 held; A/B steady TPOT graphed 60.94 vs eager 61.71 ms/tok, non-overlapping — a small real win that NARROWS the audio gap 1.52×→1.49× but does NOT close it — multimodal-speed.md §10); decode-kernel efficiency ATTRIBUTED + VALIDATED ceiling 2026-07-27 (`CLAIM-MM-SPEED-DECODE-KERN`, multimodal-speed.md §11) — the whole ~20 ms/tok audio residual is the naive scalar `PagedAttentionKernel` decode attention (723 µs × 30 = 21.7 ms/step); the 1:1 vLLM lever (FA2 `flash_attn_varlen` decode) is already in-binary, gated off only because the driver's single KV block (444) isn't ÷16; `block_size÷16` → TPOT 59.4→38.2 ms/tok (−21.2, ~36%) = 0.94× vLLM 40.8 ms (BEATS parity), FA2 sequence teacher-force-VALID (0 divergences, gap 0.0), but it flips the committed near-tie golden → blocked byte-exact; RECORDS-ONLY, 14/14 held, win one `block_size÷16` + golden regen away. **ADOPTED 2026-07-27 (USER-APPROVED, `CLAIM-MM-SPEED-DECODE-KERN-ADOPT`, multimodal-speed.md §12): FA2 decode SHIPS as the Voxtral default — audio DECODE BEATS vLLM (0.97×), the LAST mm decode-speed gap CLOSED.** One-line `block_size÷16` (nsys: `flash_fwd_splitkv` 1410 @ 18.5 µs, zero `PagedAttentionKernel`); `test_voxtral_e2e` → ratified near-tie DISTRIBUTIONAL gate (binding = teacher-force PASS, kernel-independent; strict prefix exact to first bf16 tie, FA2 pos 18; determinism anchor to the FA2 seq); `voxtral_neartie.json` regenerated (md5 `937b9ad3…`), STRICT greedy golden UNCHANGED; gate PASS 16/16; teacher-force vLLM 0.25.0 = 0 divergent, gap 0.0, PASS; capture-safe (graph 46 replays + compute-sanitizer 0 errors + 3-run byte-identical) ⇒ DEFAULT graph path; A/B scalar 60.50 → FA2 **39.50 ms/tok** (−21.0, ~35%, NON-OVERLAPPING) = 0.97× vLLM 40.8. Audio DECODE now correctness+speed DONE; umbrella MM row stays PARTIAL (audio TTFT/encoder + c2+ batched serving). **ENCODER TTFT MEASURED + warp-attention brick 2026-07-27 (`CLAIM-MM-SPEED-AUDIO-ENC`, multimodal-speed.md §13):** routed the Whisper encoder self-attention (hd-64, non-causal) from the naive O(t²) `kAttention` to the warp-scoped `vt::AttentionDenseFast` (§7 tower fix; text byte-identical) — encoder forward **8870→1890 ms (4.7×, NON-OVERLAPPING)**, `test_voxtral_e2e` **16/16** with ZERO token flips, goldens md5 unchanged; **NOT at parity** (~1.89 s vs vLLM 43 ms, ~44× — the warp kernel is STILL 31.8 ms/layer, O(t²) memory-bound), closing needs a flash-TILED non-causal hd-64 attention (LARGE) + resident one-time encoder weights (MEDIUM, byte-exact). | Every-axis **SPEED** gate on all mm rows — none is DONE (`MODEL-MM-*` all PARTIAL/ACTIVE, speed-pending). Dominant residual = **lever #3: batched/graphed mm serving (c2+)**; the audio decode-KERNEL residual is now FULLY ATTRIBUTED (the scalar `PagedAttentionKernel`) and the fix is a VALIDATED bf16-near-tie ceiling (FA2 decode beats vLLM but changes the golden's near-tie branch — §11). **W1 REFINED the audio attribution:** graphing the Voxtral decode removed the per-step launch overhead but it was only ~1.25% of TPOT, so the ~20 ms/tok gap vs vLLM's 40.8 ms is per-step COMPUTE (the scalar decode-attention kernel), NOT launch overhead as §9.5 hypothesized. The audio decode win is now ADOPTED (FA2 SHIPS, §12 — audio decode BEATS vLLM 0.97×). Remaining lever-#3 W-plan (multimodal-speed.md §9.5/§13): audio TTFT NOW MEASURED our-side (32-layer Whisper encoder forward 1890 ms after the §13 warp-attention brick, still ~44× vs vLLM 43 ms — a flash-TILED non-causal hd-64 encoder attention + resident one-time encoder weights close it); W2 batched multi-seq (c2+); W3 `image_url`/`audio_url` serving ingestion. Gemma-4 mm/audio = EXT (below). Qwen3.6-35B mm needs a vision-inclusive checkpoint download + M2/M3 tower attach. |
| `ROAD-V1-C1` extensibility | **DONE** (cornerstone) | Drop-in kernel ABI W0, Platform seam, model self-registration, and the **portable op-fusion framework ORDER-1 milestone** (W0–W4 merged+gated, `KERNEL-FUSION-FRAMEWORK`); consistency-audit CI check landed. `BACKEND-ABI-VT`/`BACKEND-CUDA-ARCH-ADDITIVITY` seams gated on sm_121a. | Row stays SPIKE-open only for **non-blocking** tail: Tier-1 fusion perf interpreter (composite-only → single-launch), `FUSION-DENSE-MIGRATE` (route 5 drift models — CLOSED 2026-08-10, [#299](https://github.com/mudler/vllm.cpp/issues/299)), a real Metal/Vulkan catalog realization (M4-reachable / HW-blocked), and migrating a production kernel family onto the common adapter. Correctness cornerstone is closed. |
| `ROAD-V1-C2` model families | **RI** (+HW/EXT sub) | First additive model (Qwen3 dense) + a broad **text sweep correctness-complete + SACRED-gated**: Qwen3/Qwen3Moe/Coder, Llama/Yi/InternLM3, Mistral, GLM-4-9B/GLM-4.7-Flash, Gemma-1/2/3, OPT, DeepSeek-V2-Lite (MLA), OLMo-2, Phi-3/4, Phi-1/2, Granite-3, StableLM, InternLM2, MiniCPM, MiniCPM3 (MLA). 20 ACTIVE model rows. | **SPEED close** on every one (all 20 are "correctness-complete, speed pending"). MoE/SSM breadth (Qwen3-Next, Falcon, Falcon-H1, GraniteMoe*, Cohere2Moe, PhiMoE, Mamba/Jamba/Zamba2/NemotronH) = RI (INVENTORIED/SPIKE). Frontier: Kimi-Linear-48B fits (RI, +KDA kernel); DeepSeek-V3/GLM-5/MiniMax-M2/M3/Kimi-K2 = HW (>119 GiB); Command-R = EXT (HF token). |
| `ROAD-V1-C2-LOCAL-BF16` | **RI** (S) | Local Qwen3.5-4B plain-BF16 diagnostic rebased onto current additive seams; CPU/CUDA + direct ON/OFF token-equivalence green; H32 AOT / plain-BF16 graphs / ratio-4 FA2 landed + trace-proven. | Port device-resident sampled-token mapping to discrete CUDA (remove the measured main-stream wait) and rerun the exact 4B series. Small. |
| `ROAD-V1-C3` spec-decode | **DONE** (core) | **MTP k=1 DONE + gated on BOTH gate models** (`SPEC-MTP`, c1 token-exact + above vLLM, c2–c8 on-par-or-above); **DFlash DONE + speed gate MET** (`SPEC-DFLASH` D14, our-ON ≥ vLLM-ON). | ~~Named tail only: DSpark (`SPEC-DSPARK`) + heterogeneous-vocabulary TLI (`SPEC-TLI`) unspiked — overlaps `ROAD-V1-D3`.~~ **SUPERSEDED 2026-08-12** ([#536](https://github.com/mudler/vllm.cpp/issues/536), see §3 item 17): `SPEC-DSPARK` is `ACTIVE` — W1–W8 implemented and GPU-gated, 35B-A3B MoE **0.975x** code / **1.012x** prose vs the pinned graphed oracle (#442), remaining work a perf tail plus owed gates. `SPEC-TLI` is genuinely untouched and belongs under `SPEC-DRAFT-MODEL`, whose W3 blocks it. `ROAD-V1-D3` excludes both by its own spec, so it overlaps nothing here. Core spec-decode is gate-closed. |
| `ROAD-V1-C4` quantization | **RI** | **3 schemes DONE**: NVFP4-MO-W4A16, NVFP4-CT-W4A4, FP8-MO-STATIC (all R/M/C/E/P). **GGUF CPU vs llama.cpp is CLOSED** (2026-07-22, aarch64 binding host): decode **at parity** (1.03× behind, inside llama.cpp's ±1.8% run spread — the elementwise f16/bf16 GEMM lever `KERNEL-GEMM-CPU-ELEM` E1-E4 `18094ee2` took it 3.38×→1.03×), prefill **1.18× ahead** (q8_0 repack-at-load G7), RSS **1.01×**, byte-identical greedy tokens. | NVFP4-CT-W4A16 perf gate. FP8-generic dispatch (static/dyn × tensor/channel/token/block). Breadth: AWQ/GPTQ/Marlin-wiring, i-quants, MXFP4/MX, bitsandbytes, KV-quant — all INVENTORIED. (GGUF-vs-llama.cpp speed is no longer an open C4 blocker.) |
| `ROAD-V1-C5` sliding/YaRN | **RI** | Joint spike accepted; all W1–W8 leaves implemented and CPU/oracle/sanitizer green. **CUDA GPU CLOSURE 2026-07-27 (`CLAIM-ROADMAP-C5`, dgx GB10 sm_121a, clean build of `489f7771`, oracle vLLM 0.26.0.dev0):** shared scaled-RoPE + local-mask CUDA path compiles `-Werror`-clean + RUNS on GB10; feature-positive correctness gates PASS — SWA Gemma-2/Gemma-3 48/48, LongRoPE Phi-4-mini 16/16 (RED-first), llama3 Llama-3.2-1B 16/16, dynamic-NTK InternLM2 16/16; both RoPE 0.26-oracle recaptures BIT-IDENTICAL to goldens. `ATTN-SLIDING-WINDOW`/`ATTN-ROPE-{LLAMA3,LONGROPE,DYNAMIC-NTK}`/`ATTN-YARN` → `ACTIVE`; `ATTN-CHUNKED-LOCAL` + `KV-*-SPEC` honest. | **Honest residual (vehicle-blocked, not skipped):** YaRN model e2e (no cached Nomic/gpt-oss consumer) + chunked-local model e2e (no Llama4 row) REACHABLE-BLOCKED; long-context positive-mask (prompt > W) SWA e2e + KV-memory G8; every-axis **SPEED** tail (all leaves correctness-complete, speed-pending). |
| `ROAD-V1-C6` async/priority serving | **RI** | `ENG-ASYNC-SCHED` **DONE** (`6ea7856`, default-ON, DGX token-neutral). W1/W2/W4 landed. | `SERVE-ASYNC-LLM` (GATING → prod-ON, blocks the SGLang floor + `ROAD-V1-A`), `ENG-PRIORITY-SCHED` + `ENG-CORE-BUSY-LOOP` GPU gates (GATING, held behind SERVE-GATE-ONLINE). |
| `ROAD-V1-C7` sampling/logprobs | **RI** | **W1-W4 LANDED + CPU-GATED 2026-07-27 (`CLAIM-ROADMAP-C7`, NOT pushed):** `SAMPLE-CORE` + `SAMPLE-LOGIT-FILTERS` -> `ACTIVE` — the full sampling-control surface (temperature/top_p/top_k/min_p/penalties/seed/stop/min_tokens/logit_bias/allowed_token_ids/bad_words/logprobs-count) WIRED end-to-end params->protocol->InputProcessor->InputBatch->SamplingMetadata->Sampler + gated exactly on the CPU reference backend (RED-first; default/greedy byte-identical). | **RESIDUAL:** W5 `SAMPLE-LOGPROBS` payload end-to-end (LogprobsProcessor + OpenAI `CompletionLogProbs` serialization, PARTIAL — count wired, sampler produces tensors) + `SAMPLE-PROMPT-LOGPROBS` need the engine-output plumbing + a running-engine gate; then `n>1`, philox-RNG, logprobs_mode, beam-search, custom logits processors (INVENTORIED). |
| `ROAD-V1-C8` tokenize/parse/metrics | **RI** | Tokenizer engines have code (`LOAD-HF-BPE` ANCHOR-BACKFILL, `LOAD-SENTENCEPIECE` ACTIVE 6/6). **`SERVE-METRICS` + `SERVE-UTILITY-ENDPOINTS` LANDED + CPU-GATED 2026-07-27 (`CLAIM-ROADMAP-C8`):** the **oldest open T0 debt** `/metrics` Prometheus is CLOSED — self-contained registry + text-0.0.4 exposition + always-on vLLM metric catalog (names/labels/buckets 1:1), gated by the vLLM scrape spec `EXPECTED_METRICS_V1` (RED-first, `test_prometheus_metrics` 4/4/81); plus `/tokenize`,`/detokenize`,`/ping`,`/server_info`,`/reset_prefix_cache` (`test_openai_api_server` 26/26/277). **`TOOLS-STREAMING-PARSER` engine core LANDED + CPU-GATED 2026-07-27 (`CLAIM-ROADMAP-C8-PARSER`):** vLLM 0.26 declarative `StreamingParserEngine` + qwen3/seed_oss/kimi_k2 configs + unified registry, EXACT event-for-event gate (`test_streaming_parser_engine` 586/586). **Parser ASSEMBLY layer LANDED + CPU-GATED 2026-07-27 (`CLAIM-ROADMAP-C8-ASSEMBLY`):** vLLM 0.26 `ParserEngine` (SemanticEvent -> streaming `DeltaMessage` + one-shot `ExtractedToolCallInformation`) + qwen3/seed_oss/kimi_k2 assembled parsers + `parser_manager` dispatch, EXACT field-for-field gate (`test_parser_engine_assembly` 9 scenarios, 1652/1652, RED-first 32 asserts). **Serving-SSE dispatch swap LANDED + CPU-GATED 2026-07-27 (`CLAIM-ROADMAP-C8-SERVING`):** the OpenAI chat streaming path routes engine-backed `--tool-call-parser` names through `parser_manager get_parser_engine` (drives `parse_delta`/`parse`), EXACT chunk-for-chunk gate vs vLLM 0.26 `chat_completion_stream_generator` (`test_openai_serving_chat_stream` 9 scenarios, 210/210, RED-first 6 CHECKs); OFF by default, legacy seam byte-identical. **5 more engine CONFIG families LANDED + CPU-GATED 2026-07-27 (`CLAIM-ROADMAP-C8-CONFIGS`):** minimax_m2/glm47_moe/deepseek_v4/deepseek_v32/nemotron_v3 as additive `ParserEngineConfig` builders + regex arg-converters + `Glm47MoeParser` name-`.strip()`, EXACT field-for-field (`test_parser_engine_assembly` 19 scenarios, 3510/3510, RED-first 2 asserts). **LAST 2 CONFIG families gemma4 + inkling LANDED + CPU-GATED 2026-07-27 (`CLAIM-ROADMAP-C8-CONFIGS-2`) — vLLM tool-parser family parity CLOSED:** 4 default-inert assembly-core virtual seams (`preprocess_feed`, virtual `events_to_delta`/`single_pass_parse`/`reset`/`extract_reasoning`, `args_wrapper_keys`) + `gemma4_config`/`inkling_config` + `Gemma4Parser` (channel-injection + `thought\n`-strip) / `InklingParser` (args-key unwrap + trailing-text flush), EXACT field-for-field (`test_parser_engine_assembly` 26 scenarios, 4526/4526, adds a non-streaming parse() gate, RED-first for all 4 seams; engine-core 586/586 + serving-SSE 210/210 byte-identical). **LIVE PER-STEP METRIC WIRING LANDED + CPU-GATED 2026-07-27 (`CLAIM-ROADMAP-C8-METRICS-WIRE`) — oldest T0 metrics debt RETIRED:** `/metrics` serves LIVE values — `Scheduler::make_stats()`→`EngineCoreOutputs.scheduler_stats` + `OutputProcessor`-built `IterationStats` fold into the logger at the `LLMEngine` step site (`llm_engine.py:308-329`); behavioural CPU gate `test_llm_engine.cpp` case 6 (44 asserts, RED-first 14 flip 0→correct) — gauges track the batch, token counters == exact counts, request_success + TTFT/ITL/e2e/TPOT/iteration histograms correct; catalog gate 4/4/81 + greedy stream unchanged. **PER-REQUEST TIMING via EngineCoreEvents LANDED + CPU-GATED 2026-07-27 (`CLAIM-ROADMAP-C8-RESPONSE-METRICS`) — `SERVE-RESPONSE-METRICS` INVENTORIED→ACTIVE:** the scheduler records QUEUED/SCHEDULED/PREEMPTED `EngineCoreEvent`s 1:1 vLLM (gated on `log_stats_`), drained onto `EngineCoreOutput.events`; `OutputProcessor.update_from_events` fills `FinishedRequestStats.{queued,prefill,inference}_time` + `IterationStats.num_preempted_reqs`, so `vllm:request_{queue,prefill,inference}_time_seconds` + `vllm:num_preemptions_total` carry real durations (the live-metrics wiring left them at 0); `test_scheduler` +1/15 (RED-first, real KV-exhaustion preemption) + `test_llm_engine` +1/26 (RED-first: 5 flip 0→positive), no-logger path byte-identical. **CHAT-FORM `/tokenize` LANDED + CPU-GATED 2026-07-28 (`CLAIM-C8-CHAT-TOKENIZE`):** `/tokenize` now accepts BOTH arms of the `TokenizeRequest` union — the chat form (`TokenizeChatRequest{messages, add_generation_prompt, continue_final_message, add_special_tokens, tools?}`) renders through the SAME chat template `create_chat_completion` tokenizes through (`chat_.prompt_fn()`, no reinvention), applies `check_generation_prompt` (→400), tokenizes with the chat-form `add_special_tokens` default False, returns `{count,max_model_len,tokens,token_strs?}` identically to vLLM 0.26 `serving_tokenization`; exact-gated tokens == render→Encode, RED-first (`test_openai_api_server` 27/27/337). **JSON-SCHEMA ARG-TYPE COERCION LANDED + CPU-GATED 2026-07-28 (`CLAIM-C8-ARG-COERCION`):** `_fix_arg_types`/`_streamable_string_keys`/`find_tool_properties` (parser_engine.py:227,269,365,348) ported over the already-ported `extract_types_from_schema`/`coerce_to_schema_type` helpers — `ParserTool` carries the function `parameters` schema (threaded from `serving_chat.cpp` `ToParserRequest`), so tools declaring typed params get their assembled `tool_calls[].function.arguments` coerced to int/number/bool/string/array/null in BOTH streaming and one-shot (priority null>int>number>bool>object>array>string, uncoercible left as-is); no-schema/absent-tools identity byte-identical. EXACT field-for-field (`test_parser_engine_assembly` 30 scenarios, 5038/5038, scenarios 27-30 qwen3 typed-schema whole+char + schema-mismatch + kimi JSON-native `"5"`->int, RED-first 38 asserts; engine-core 586/586 + serving-SSE 210/210 byte-identical). | **ASYNC SERVING-PATH METRIC WIRING LANDED + CPU-GATED 2026-08-10 (`CLAIM-SERVE-METRICS-ASYNC`, #277) — punch-list item 7 CLOSED:** `AsyncLLM::RunOutputHandler` folds each step's `SchedulerStats` + `IterationStats` into the attached logger, `EngineCore::step_with_batch_queue` stamps the `scheduler_stats`/`timestamp` upstream stamps in the shared `update_from_output` path, the logger takes a leaf mutex for scrape/record overlap, and `server_main.cpp` attaches one logger to both frontends; RED-first on three gates, CPU `ctest` 366/366. Residual: the chat/completion RESPONSE-BODY timing surface + CLI validation; config-gated families (spec-decode/kv-connector/mm/LoRA); `chat_template_kwargs`/`continue_final_message` full render passthrough. |
| `ROAD-V1-C9` recurring sync | **RI** (recurring) | Pin **advanced to `555967922` / vLLM 0.26.0.dev0 + transformers 5.14.1** (`CLAIM-PIN-ADVANCE-W5`); re-gate 296/299 GREEN, zero golden drift. This unblocked OLMo-3 W5, DFlash, Gemma-4 module. | Refresh exact performance denominators + target goldens/tests on 0.26; then the ongoing mechanical cycle. Never terminally "done" (recurring). |
| `ROAD-V1-D1` backend fan-out | **RI + HW** | CUDA-arch **seams landed + gated** on sm_121a (`BACKEND-CUDA-ARCH-ADDITIVITY` ACTIVE: feature table, capability probe, runtime SM-dispatch registry). **`ROAD-V1-D1-CUDA` derive-and-ship first brick LANDED (2026-07-27): FA2 Ampere (WA-1) DERIVED+BUILD-VERIFIED (testing-welcome)** — `fa2` cell widened to `8.0,8.6,8.7,8.9,12.0a,12.1a`; `87`+`80` dgx builds `-Werror` 0-warn + `cuobjdump` sm_87/sm_80 FA2 cubins; sm_121a OLMo-2 gate 16/16 unchanged; NO Ampere board ran it. **CPU REACHABLE** (macOS M4 `-Werror`-clean, 108k assertions). **Metal/MLX REACHABLE + running** (`BACKEND-METAL-MLX` ACTIVE: 18 MSL kernels, 2 models token-exact/near-tie on Apple GPU, M3c-1/M3d landed). | **REACHABLE:** Metal/MLX perf close (still ~2.96× behind MLX-LM b=1; batched-decode tile kernel; binding needs M4 quiet/sudo), CPU B4 speed/RSS gates. **HW-BLOCKED (build-only ceiling):** ROCm (INVENTORIED, no AMD board), Intel XPU (SPIKE, no board + upstream loyalty target incomplete), discrete-Vulkan (ACTIVE skeleton + hermetic SPIR-V, but only llvmpipe/GB10-unified here), ANE (encoder/pooling niche), and every cross-family CUDA kernel body / gencode beyond sm_120/121. |
| `ROAD-V1-D2` multi-GPU/TP | **HW** | `PAR-TP` READY (spike accepted). | Runtime is **HW-BLOCKED: needs a 2-GPU box** (GB10 is single-GPU). Reachable build-only: TP mock/ABI Phase-0. PP/EP/DP/sequence-MoE/multinode all INVENTORIED and also multi-GPU-gated. |
| `ROAD-V1-D3` spec-decode breadth | **DONE** (2026-07-27) | Reused the landed spec machinery (frozen ABI + rejection sampler + runner loop from MTP/DFlash). | `SPEC-NGRAM` **DONE/ACTIVE** — draft-FREE n-gram proposer (1:1 port of `ngram_proposer.py`), 27B gate 5/5 STRICT our-ngram-ON == vLLM-ngram-ON + 180/180 accepted, spec-OFF byte-identical (SACRED 235 + MTP 9 + DFlash 27), no new kernel. `SPEC-EAGLE3` **BLOCKED (honest)** — no ungated oracle-runnable EAGLE3 draft arch/checkpoint for a Qwen3.6 gate model at pin `555967922` (registry has no `Eagle3Qwen3_5*`; z-lab published DFlash not EAGLE3); port scoped. See [spec-decode-breadth-d3.md](spec-decode-breadth-d3.md). |
| `ROAD-V1-D4` KV-disk/LMCache | **RI** (+HW sub) | `KV-OFFLOAD` ACTIVE (CPU+disk tiers W1–W5, 32/48 restart-prefill saved); `KV-EXTERNAL-CACHE` ACTIVE (LMCache `lm://` client W1–W5, bit-identical vs cold prefill on OPT-125m); `KV-CONNECTORS` ACTIVE (ABI + `--kv-transfer-config` CLI). | `KV-EVENTS` **ACTIVE** (event generation + byte-exact `msgpack` payload DONE 2026-07-27; live ZMQ transport + engine batch wiring deferred), W6 LMCache go/no-go study, W7 named per-sequence save/restore (the beyond-parity item), and a **binding every-axis LMCache grid on a LARGER model** (125M is noise-dominated). **HW-BLOCKED sub:** NIXL/Mooncake/HF3FS/MoRI-IO + P/D disaggregation (need RDMA/multi-node). |
| `ROAD-V1-D4-APC` prefix caching | **DONE** (headline; +RI tails) | Deep port default-ON for dense: `KV-PREFIX-CACHE` **DONE** (chain hashing, block pool, all coordinators, hybrid intersection, stats, extra_keys, cache-ON e2e gated); `KV-BLOCK-POOL`/`KV-HYBRID-COORD` PARTIAL; `ENG-CASCADE-ATTN` verified-not-owed. **W2 `extra_keys` DONE 2026-07-27** (RED-first no-false-share `n1 48->0`). **W3 DONE 2026-07-27 (`CLAIM-ROADMAP-D4APC-W3`, dgx GB10) — the FIRST-EVER cache-ON model gate:** on `Qwen/Qwen3-4B` (dense, full-attention, APC-default-ON — the vehicle the prior "vehicle-blocked" note missed) `test_qwen3_apc_e2e` 2/2: APC-ON hits 2240/2777 (0.807) / OFF 0; ON==OFF token-exact 5/6 (1 diff = vLLM-confirmed 0.125-nat near-tie, RCA'd not-a-bug); == vLLM-APC-ON teacher-forced (OFF gap 0.0, ON gap ≤0.125 nats); TTFT 70.1→39.9 ms = 1.76×. NO engine code changed (gate-only); 4B SACRED 16/16 no-regression. | **Headline DONE for the default dense APC path.** Named NON-BLOCKING tails (own rows/future): W4 events (`KV-EVENTS` ACTIVE — generation + payload gated 2026-07-27, ZMQ transport deferred), W5 partial-block primitive, W6 Mamba `align` hybrid cache-on (`KV-MAMBA-ALIGN` SPIKE — feeds `BACKEND-GATE-CUDA-SGLANG-PREFIX`), W7 `reset_prefix_cache` endpoint + hash-algos, W8/W9 beyond-vLLM save-restore. The **every-axis cache-on grid vs vLLM/SGLang** is a separate perf follow-on under `ROAD-V1-A`. |
| `ROAD-V1-D5` LoRA/offload/experts | **RI** | `ENG-EXPERT-STREAM` READY (surpass-track spike accepted; absent in pinned vLLM). | `ENG-EXPERT-STREAM` W0–W2 (bank/reader/cache-policy), `ENG-WEIGHT-OFFLOAD` (INVENTORIED, UVA `cpu_offload_gb`), `LORA-RUNTIME` + `LORA-ENDPOINTS` (INVENTORIED, Punica-style batched apply + dynamic load/unload). LoRA is the user headline; large. |

**Counts (18 portfolio rows):** **DONE = 4** (`C1` cornerstone, `C3` core spec-decode, `D3` spec-decode breadth, `D4-APC` prefix-caching headline — dense cache-ON e2e gated 2026-07-27, named tails non-blocking) · **REACHABLE-INCOMPLETE = 13** (`A, MM, C2, C2a, C4, C5, C6, C7, C8, C9, D1, D4, D5` — note `MM/C2/D1/D4` also carry HW/EXT sub-items) · **HW-BLOCKED (primary) = 1** (`D2`). No row is *purely* external-blocked; EXT applies to sub-items (Gemma-4, Command-R, DeepSeek-V3.2/GLM-5 DSA) inside `MM`/`C2`.

## 2. The blocked set (what bounds "complete roadmap_v1")

These cannot be gated on the current HW; the additive/build-only portion is
landable (honesty-pass ceiling) but the runtime/perf gate is not.

**HW-BLOCKED (need a board we don't have):**
- **Multi-GPU / TP-PP-EP-DP-seqMoE runtime** (`ROAD-V1-D2`, `PAR-*`) — needs ≥2 GPUs.
- **ROCm** (AMD), **Intel XPU** (Intel GPU), **discrete Vulkan** (only llvmpipe/GB10-unified here), **ANE** (encoder/pooling niche) — `ROAD-V1-D1` backend runtimes.
- **CUDA arch fan-out beyond sm_120/121** — only GB10 sm_121 is testable; sm_80/90/100 kernel bodies + gencode are build-only, and every cross-family kernel body under `kernel-matrix` is HW-gated.
- **RDMA/multi-node KV connectors** — NIXL, Mooncake, HF3FS, MoRI-IO, and prefill/decode disaggregation (`ROAD-V1-D4` sub).
- **Frontier models over the 119 GiB pool** — DeepSeek-V3/V3.2 (~642 GiB), GLM-5 (~1404 GiB), MiniMax-M2 (~214 GiB), MiniMax-M3 (~795 GiB), Kimi-K2 (~958 GiB). (Kimi-Linear-48B at 91.5 GiB and GLM-4.5-Air-FP8 at 104.8 GiB DO fit → reachable.)

**EXTERNAL-BLOCKED (need a pin/dep advance or an HF token):**
- **Gemma-4 mm + audio** — the transformers-module block is now cleared by the 0.26 pin, but public checkpoints (≥12B incl. the encoder-free unified 12B) are **HF-gated (no dgx token)**, and the USM-Conformer audio tower is still owed. Reopens with a token.
- **Command-R / Cohere** — implemented (zero-new-kernel) but every real checkpoint is HF-gated (no dgx token) + dgx disk-full; only tiny-random vehicles exist → no SACRED gate.
- **DeepSeek-V3.2 / GLM-5 DSA** — DEP-blocked on the DSA sparse-attention indexer dependency (on top of the memory HW block).

**Now-UNBLOCKED by the 0.26 pin advance (moved OUT of blocked → reachable):**
OLMo-3 W5 sliding-window SACRED gate, and DFlash (already DONE) — both were
oracle-blocked on transformers < the pin.

## 3. Ranked REACHABLE punch-list (execution order)

Ranked by (value × unblocks-others ÷ size). Each row: remaining W-plan skeleton →
gate → size (S/M/L) → vehicle model. `[H]` = user-directed headline.

1. **`ROAD-V1-MM` speed close** `[H]` (top user priority; correctness already landed).
   W: batched/graphed mm serving path (c2+), device-resident mm-token routing, audio
   our-side timing lever. **Gate:** every-axis (tput/TTFT/TPOT/mem) ≥ vLLM at the mm
   operating point, correctness held (image/video 32/32, audio near-tie). **Size M.**
   **Vehicle:** Qwen3.6-27B image+video, Voxtral-Mini-3B audio.
2. **`ROAD-V1-D4-APC` W3 cache-ON gate + W2 extra_keys — DONE 2026-07-27** `[H]`
   ("same featureset and better"). **W2 DONE** (`generate_block_hash_extra_keys`
   mm/LoRA/`cache_salt` 1:1; RED-first no-false-share). **W3 DONE (`CLAIM-ROADMAP-D4APC-W3`,
   dgx GB10):** the first-ever cache-ON model gate — `Qwen/Qwen3-4B` (dense, full-attention,
   APC-default-ON) IS the reachable vehicle (the prior "vehicle-blocked" note only checked
   the hybrid harnesses + OPT-125m). `test_qwen3_apc_e2e` 2/2: APC-ON==APC-OFF token-exact
   5/6 (1 vLLM-confirmed 0.125-nat near-tie), == vLLM-APC-ON teacher-forced, hits 2240/2777
   (0.807), TTFT 70.1→39.9 ms = 1.76×; NO engine code (gate-only); 4B SACRED 16/16
   no-regression. **Remaining (non-blocking tails, own rows/future):** W4 events,
   W5 partial-block, W6 Mamba `align`, W7 reset endpoint + `skip_reading_prefix_cache`,
   W8/W9 beyond-vLLM; the **every-axis cache-on grid vs vLLM/SGLang** is a perf follow-on
   under `ROAD-V1-A`. **Size (remaining) S→M.**
3. **`ROAD-V1-C5` sliding/YaRN GPU closure** (CPU already green; unblocks long-context
   + sliding-window model families). W: compile/run the shared scaled-RoPE + local-mask
   CUDA path; pass feature-positive model/oracle/trace gates. **Gate:** model e2e +
   every-axis on a scaled-RoPE consumer + both Qwen3.6 regressions. **Size M.**
   **Vehicle:** a YaRN/LongRoPE model + Gemma-2 sliding-window.
4. **`ROAD-V1-C7` sampling/logprobs API wiring** `[H]` (unblocks C8 + API parity).
   W: finish `SAMPLE-LOGPROBS`/`SAMPLE-LOGIT-FILTERS` payloads; add `prompt_logprobs`,
   logprobs_mode. **Gate:** end-to-end parity vs vLLM payloads on the server. **Size M.**
   **Vehicle:** any gated dense model via the OpenAI server.
5. **`ROAD-V1-A` 35B closure + SGLang floor** `[H]` (27B already ratified). W: close the
   35B c1/c2 residual (merged-projection fp8 glue + aux-stream slices); flip
   `SERVE-ASYNC-LLM` prod-ON to unblock the SGLang arm. **Gate:** 35B every-axis ≥ vLLM;
   SGLang preflight equivalence then the binding SGLang + prefix arms. **Size M→L.**
   **Vehicle:** Qwen3.6-35B-A3B-NVFP4.
6. **`ROAD-V1-D3` ngram + EAGLE3 — DONE 2026-07-27.** `SPEC-NGRAM` (draft-free
   proposer, no model) LANDED + gated: 27B 5/5 STRICT our-ngram-ON == vLLM-ngram-ON
   + 180/180 accepted, spec-OFF byte-identical, no new kernel. `SPEC-EAGLE3` honest
   reachable-BLOCKED (no ungated oracle-runnable EAGLE3 draft arch/checkpoint for a
   Qwen3.6 gate model at pin `555967922`; port scoped). See
   [spec-decode-breadth-d3.md](spec-decode-breadth-d3.md).
7. **`ROAD-V1-C8` /metrics + streaming parser + utility endpoints** `[H]` (oldest T0
   debt). **PARTIAL 2026-07-27 (`CLAIM-ROADMAP-C8`):** `/metrics` Prometheus core CLOSED
   (`SERVE-METRICS` ACTIVE — names/labels/buckets 1:1, `EXPECTED_METRICS_V1` RED-first
   scrape gate) + `/tokenize`,`/detokenize`,`/ping`,`/server_info`,`/reset_prefix_cache`
   (`SERVE-UTILITY-ENDPOINTS` ACTIVE). **`TOOLS-STREAMING-PARSER` ENGINE CORE LANDED +
   CPU-GATED 2026-07-27 (`CLAIM-ROADMAP-C8-PARSER`):** vLLM 0.26 declarative
   `StreamingParserEngine` (token-ID scanner + prefix-buffering lexer + transition state
   machine + JSON-arg hold-back) with qwen3/seed_oss/kimi_k2 configs + unified registry,
   EXACT event-for-event gate vs vLLM 0.26 (`test_streaming_parser_engine` 586/586,
   RED-first). **Parser ASSEMBLY layer LANDED + CPU-GATED 2026-07-27
   (`CLAIM-ROADMAP-C8-ASSEMBLY`):** the `ParserEngine` (SemanticEvent -> streaming
   `DeltaMessage` + one-shot `ExtractedToolCallInformation`) + qwen3/seed_oss/kimi_k2
   assembled parsers + `parser_manager` dispatch, EXACT field-for-field gate
   (`test_parser_engine_assembly` 9 scenarios, 1652/1652, RED-first). **Serving-SSE
   dispatch swap LANDED + CPU-GATED 2026-07-27 (`CLAIM-ROADMAP-C8-SERVING`):** engine-backed
   `--tool-call-parser` names drive the live chat SSE path via `parser_manager`, EXACT
   chunk-for-chunk vs vLLM 0.26 (`test_openai_serving_chat_stream` 210/210, RED-first,
   OFF by default). **Live per-step metric wiring LANDED + CPU-GATED 2026-07-27
   (`CLAIM-ROADMAP-C8-METRICS-WIRE`):** `/metrics` serves LIVE values — `make_stats()` +
   `IterationStats` fold into the logger at the `LLMEngine` step site (behavioural CPU
   gate 44 asserts, RED-first). **`SERVE-RESPONSE-METRICS` per-request timing LANDED +
   CPU-GATED 2026-07-27 (`CLAIM-ROADMAP-C8-RESPONSE-METRICS`, INVENTORIED→ACTIVE):**
   QUEUED/SCHEDULED/PREEMPTED EngineCoreEvents 1:1 vLLM feed the queue/prefill/inference
   timing histograms + preemption counter the live-metrics wiring left at 0 (`test_scheduler`
   +1/15 + `test_llm_engine` +1/26, RED-first). JSON-schema arg-type coercion LANDED +
   CPU-GATED 2026-07-28 (`CLAIM-C8-ARG-COERCION`): typed tool-call arguments coerced in
   streaming + one-shot, no-schema identity (`test_parser_engine_assembly` 5038/5038 over
   30 scenarios, RED-first 38 asserts). ASYNC SERVING-PATH METRIC WIRING LANDED +
   CPU-GATED 2026-08-10 (`CLAIM-SERVE-METRICS-ASYNC`, #277): the shipped server
   serves `/metrics` off `AsyncLLM`, which folded nothing into the logger, so a real
   deployment scraped a catalog that never moved. Its output handler now folds each
   step's stats (`async_llm.py:648-652,664-665,676-678,697-702`),
   `step_with_batch_queue` stamps the `scheduler_stats`/`timestamp` upstream stamps
   in the path both step functions share, and the logger takes a leaf mutex for the
   scrape/record overlap (`test_llm_engine` +2, `test_async_llm` +2, RED-first;
   `ctest` 366/366). Remaining W:
   the chat/completion RESPONSE-BODY timing surface; config-gated families.
   **Gate:** metric-name
   (done) + streamed-delta parity vs vLLM. **Size M→L.** **Vehicle:** the OpenAI server.
8. **`ROAD-V1-C4` FP8-generic + quant breadth** `[H]`. **GGUF-vs-llama.cpp CPU speed is
   CLOSED** (decode at parity 1.03×, prefill 1.18× ahead, RSS 1.01×, byte-identical —
   `KERNEL-GEMM-CPU-ELEM` E1-E4 `18094ee2` + G7 repack; verified still-parity 2026-07-27,
   no open elementwise-GEMM decode lever, decode is DRAM-bound and should not receive
   further kernel effort). Remaining C4 W: NVFP4-CT-W4A16 perf gate; FP8-generic dispatch
   (static/dyn × tensor/channel/token/block); then AWQ/GPTQ/Marlin wiring; i-quants;
   MXFP4/MX; bitsandbytes; KV-quant. **Gate:** e2e correctness + every-axis perf per
   scheme. **Size M (FP8-generic) → L (breadth).** **Vehicle:** FP8 checkpoints on DGX;
   AWQ/GPTQ checkpoints for the breadth wiring.
9. **`ROAD-V1-C6` SERVE-ASYNC-LLM + priority prod gates** (shared with A#5). W: land the
   depth-2 throughput lever, flip `runner_supports_async` prod-ON, close priority/busy-
   loop GPU gates. **Gate:** every-axis no-regression + priority-vs-FCFS token-exact.
   **Size M.** **Vehicle:** 27B online serving.
10. **`ROAD-V1-C2` text-sweep speed close + MoE/SSM breadth** (20 families correctness-
    complete). W: per-family speed lever to vLLM parity; then Qwen3-Next / Falcon-H1 /
    GraniteMoe / Mamba-hybrid campaigns; Kimi-Linear-48B (+KDA kernel, disk reclaim).
    **Gate:** token-exact (near-tie-robust) AND ≥ vLLM every axis per model. **Size L
    (ongoing).** **Vehicle:** each family's smallest gateable checkpoint.
11. **`ROAD-V1-D4` KV-disk/LMCache finish** (mostly landed). W: `KV-EVENTS` payload +
    emission sites DONE 2026-07-27 (ACTIVE; live ZMQ transport + engine batch wiring deferred);
    W6 LMCache go/no-go; W7 named save/restore; larger-model LMCache
    grid. **Gate:** every-axis LMCache grid vs vLLM `--kv-transfer-config` on a large
    model. **Size S→M.** **Vehicle:** a larger dense model + a live LMCache server.
12. **`ROAD-V1-D5` expert-streaming → LoRA** `[H]` (LoRA is headline; large). W:
    `ENG-EXPERT-STREAM` W0–W2; then `LORA-RUNTIME` (Punica batched apply) + `LORA-
    ENDPOINTS` + `ENG-WEIGHT-OFFLOAD`. **Gate:** token-exact LoRA apply vs vLLM + memory
    gate. **Size L.** **Vehicle:** a base model + a public LoRA adapter.
13. **`ROAD-V1-D1` Metal/MLX + CPU perf close** (the only reachable accelerators). W:
    Metal batched-decode tile kernel + binding harness (M4 quiet/sudo); CPU B4 speed/RSS.
    **Gate:** ≥ MLX-LM (Metal) / ≥ llama.cpp (CPU) binding A/B. **Size M.** **Vehicle:**
    Qwen3-1.7B/4B on M4; Qwen3.5-2B Q8 on CPU.
    **CPU half, 2026-08-11 ([#433](https://github.com/mudler/vllm.cpp/issues/433),
    [x86 arm](cpu-llamacpp-floor-x86-2026-08-11.md)):** "CPU B4 speed/RSS" was a
    two-arm item that had quietly become three. The 20-core Arm/i8mm arm is
    CLOSED (parity or better on every axis) and the four-core A76 arm is OPEN on
    speed (#284), and both are AArch64. The **x86_64** arm had no post-lever
    number at all, and every lever that closed the first arm is Arm-scoped, so
    nothing transferred. It is now measured, and **no axis of it is met**: peak
    RSS **1.0022x = a hairline OPEN GAP**, 6.33 MB against us on a lower-is-better
    axis and 12-55x the leg spread; prefill/decode/E2E **`PENDING` a quiet host**
    (this box is `VOID` for binding timing, re-confirmed by a 5-rep series
    discarded at load 82); load-discipline gate `G5` **FAILING**, its raw per-leg
    output never committed and now gone. Correctness holds byte-identically at
    the measured length of 32 and **fails at the 64 originally declared** (one
    divergence, oracle self-stable, so no distributional gate applies). Next CPU
    lever is CIQ `G5` (x86 AVX2/AVX-512 quant tier + an AVX-512 consumer for the
    `G7` repack), and a per-pool RSS attribution for the 6.33 MB.
    **The Metal/MLX half is untouched: it needs an Apple M4.**
14. **`ROAD-V1-C2-LOCAL-BF16` device-resident sampled-token rerun** (small). **Size S.**
15. **`ROAD-V1-C1` fusion perf interpreter** (cornerstone done; perf tail,
    ≤3.5%/step ceiling). **Size M.** `FUSION-DENSE-MIGRATE` is CLOSED (2026-08-10,
    [#299](https://github.com/mudler/vllm.cpp/issues/299), spec
    [fusion-dense-migrate.md](fusion-dense-migrate.md)): the 5 drift models
    (commandr/glm4/minicpm/minicpm3/phi3) route gate/up through
    `layers::UnquantizedMlpGateUpMethod` and the merged-GEMM allowlist shrank 11 → 6,
    leaving only entries that need the shared layer extended. The dgx paged-engine
    confirmation for those five is OWED.
16. **`ROAD-V1-C9` 0.26 denominators/goldens refresh** (recurring). **Size S, ongoing.**
17. ~~**`ROAD-V1-C3` DSpark + TLI** (core spec-decode done; overlaps D3). **Size M.**~~
    **RECONCILED 2026-08-12** ([#536](https://github.com/mudler/vllm.cpp/issues/536)):
    the item was written when both halves were untouched. Neither half is what
    it says, and they are not one item.
    **DSpark is not unspiked.** Its spike spec landed 2026-08-09 (`2b342620e`,
    [dspark-spec-decode.md](dspark-spec-decode.md)), `SPEC-DSPARK` has been
    `ACTIVE` since, and W1–W8 are implemented and GPU-gated: the Markov head,
    the sequential sampler, native **and** Speculators-format loading, the `d2t`
    reduced draft vocab, the runner/one-surface wiring, the device sequential
    sample ([#436](https://github.com/mudler/vllm.cpp/issues/436)) and the T=1+k
    verify capture ([#442](https://github.com/mudler/vllm.cpp/issues/442),
    mirroring vLLM's `uniform_decode_query_len = 1 + num_speculative_tokens`).
    En route it fixed an engine-wide defect: `EngineCoreProc` never threaded
    `check_for_draft_tokens`, so EVERY speculator's drafts were dropped on the
    CLI and server paths. Measured against the pinned graphed oracle under
    pinned clocks the 35B-A3B MoE lane measured **0.975x** (code cell,
    non-overlapping distributions) and **1.012x** (prose cell) ON THE
    PRE-REIMAGE BOX. Both figures are SUPERSEDED: that machine no longer
    exists, and on the rebuilt stack the matched-and-warm paired ratio is
    **0.834** -- see the benchmark record entries of 2026-08-15, which also
    record that every earlier ratio used a single COLD oracle invocation. With the
    residual localised to one kernel and attributed to a **12.9%
    effective-DRAM-bandwidth** gap on byte-equivalent machine code (94 registers
    / 3664 SASS instructions on both sides, spec §§6t–6aa). **Remaining is a
    perf tail plus owed gates, not a port: Size S–M**, and its next lever is
    named (`cudaMemAdvise`/placement on the expert slab; upstream `ncu` counters
    are BLOCKED in both replay modes, so a standalone `moe_wna16_marlin_gemm`
    harness is the only remaining route). Owed for a binding W6: the SACRED-corpus
    token gate under the ratified near-tie protocol, the 27B dense re-measure
    (its earlier cells were never like-for-like), the Gemma4 `1 + N` layout on
    real weights, and padded/multi-request spec capture shapes.
    **TLI is untouched — and it is not a DSpark tail.** No commit, no code, no
    spec, no issue; `SPEC-TLI` is `INVENTORIED`. Upstream TLI is
    `use_heterogeneous_vocab` (`config/speculative.py:150`) + `VocabMapping`
    (`v1/spec_decode/vocab_mapping.py:68`), consumed ONLY by
    `v1/spec_decode/llm_base_proposer.py` (`SpecDecodeBaseProposer`) and
    `v1/spec_decode/draft_model.py:19`. The V2-runner speculators our
    DFlash/DSpark port mirrors (`v1/worker/gpu/spec_decode/{dflash,dspark}/`)
    have no heterogeneous-vocab path at all, so TLI's host is `SPEC-DRAFT-MODEL`
    — a CPU propose brick with no runner construction — and TLI is
    prerequisite-blocked behind that row's W3, not merely unspiked. DSpark's
    `d2t` does not cover it: `d2t` is an offset table inside ONE tokenizer's
    vocabulary (`draft_id + d2t[draft_id]`), while TLI builds a string-level
    intersection ACROSS tokenizer families (BPE `Ġ` vs SentencePiece `▁`,
    probed at init). **Size M, and it should be re-filed under `SPEC-DRAFT-MODEL`.**
    **The D3 overlap is backwards.**
    [spec-decode-breadth-d3.md](spec-decode-breadth-d3.md) §Scope puts DSpark and
    TLI explicitly *out of* `ROAD-V1-D3` and back under `ROAD-V1-C3`; D3's
    landing covers no part of this tail. What DSpark reused is C3's own
    MTP/DFlash verify/reject loop.

## 4. Bottom line

- **DONE (merged+gated): 4** portfolio rows — `C1` (extensibility + fusion ORDER-1
  cornerstone), `C3` (MTP + DFlash spec-decode, both speed-gated), `D3` (spec-decode
  breadth: ngram gated + EAGLE3 honestly blocked), and `D4-APC` (prefix-caching headline
  — dense cache-ON e2e gated on Qwen3-4B 2026-07-27; named W4-W9 tails non-blocking). At the
  *sub-milestone* level far more is delivered (MM image/video/audio **correctness**,
  first additive model + a 20-family text sweep **correctness**, D4 KV-disk+LMCache
  **landed**, D1 CUDA-arch **seams**), but those rows retain an open SPEED or breadth
  gate and so are not row-DONE.
- **REACHABLE-INCOMPLETE: 13** rows — executable now on GB10 + CPU + M4. The single
  dominant cross-cutting gate is **every-axis SPEED**: it is the entire remaining work
  for MM and the 20 correctness-complete model families, and a major part of A, C4, C5,
  C6, D1, D4. The reachable feature-parity gaps (C7 sampling, C8 metrics/parser,
  D5 LoRA) are ordinary ports with existing harnesses. (D3 spec-breadth CLOSED
  2026-07-27: ngram gated + EAGLE3 honestly blocked. D4-APC prefix-caching headline
  CLOSED 2026-07-27: dense cache-ON e2e gated on Qwen3-4B; W4-W9 tails non-blocking.)
- **HW/EXTERNAL-BLOCKED: bounds completeness.** "Complete roadmap_v1" is **NOT fully
  reachable on the current hardware.** It is bounded by, and only by, the blocked set in
  §2: `ROAD-V1-D2` multi-GPU; the `ROAD-V1-D1` non-CUDA accelerators (ROCm, Intel XPU,
  discrete Vulkan, ANE) and CUDA arch fan-out beyond sm_121; RDMA/multi-node KV
  connectors + PD-disaggregation; the >119 GiB frontier models; and the HF-token /
  DSA-dep models (Gemma-4 checkpoints, Command-R, DeepSeek-V3.2/GLM-5). Every one of
  those has a landable additive/build-only portion (honesty-pass ceiling) but no runtime
  gate without the missing board/token/dep. **Everything else — the whole C-track
  feature portfolio, all speed closure, the reachable model zoo, Metal/CPU — is
  reachable and is the punch-list in §3.**
