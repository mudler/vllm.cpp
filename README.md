# vllm.cpp

A 1:1 port of [vLLM](https://github.com/vllm-project/vllm) to pure C++ —
no Python, no PyTorch, no ggml at runtime. It mirrors the V1 architecture and
targets the same algorithms and serving surface; the tables below distinguish
the implemented subset from the remaining parity work. It is usable as a
library (llama.cpp-style) with a C API, an example CLI, and an
OpenAI-compatible server.

> ⚠️ **Pre-release, under heavy development.** Both NVFP4 gate models —
> **Qwen3.6-35B-A3B** and **Qwen3.6-27B** — run the full paged-KV engine on
> **NVIDIA GB10** with token-exact greedy correctness gates passing. Performance
> parity is **GATING** against
> [vLLM v0.25.0](https://github.com/vllm-project/vllm/releases/tag/v0.25.0)
> at `702f481`. The immutable `3f256ab` 27B cache-off gate is complete and
> binding: all **124/124** axes are valid, but only **55/124** pass. Therefore
> 27B parity is **FAILED/open**, and 35B performance remains prohibited.
>
> The previous exact 27B floor (`b5c6e4f`) used vLLM 0.24.0 with FlashInfer
> 0.6.12. It remains useful historical evidence—totals
> **0.993/0.952/0.966/0.976/1.021/1.022×** at c1/2/4/8/16/32—but it is
> superseded because even the source pin required FlashInfer 0.6.13. The
> replacement `3cc490c` attempt is **VOID**: it was stopped after **28/36**
> groups, **1,602/2,016** timed requests and four memory returns, before its
> paired trace, once the oracle drift was proven. No partial rate is published.
>
> The v0.25 audit found no obsolete local legacy PagedAttention source to
> remove: upstream deleted only `paged_attention_v1/v2`, while this project uses
> a live backend-neutral paged-KV contract with current FA2/fallback kernels.
> It also confirms the next FP4 candidates remain current: v0.25 still consumes
> directly swizzled scale factors with zeroed unread padding and a model-owned
> device `alpha` pointer. The exact v0.25 oracle is now validated and active. A
> fresh immutable `3f256ab` campaign completed the 27B model gate, all **36/36**
> timed groups (**2,016** requests), six memory returns and a paired execution
> trace under one uncontended whole-series lock. Median total-throughput ratios
> at c1/2/4/8/16/32 are **0.994/0.954/0.966/0.981/1.028/1.039×**; the per-point
> performance-axis pass counts are **5/4/5/4/17/18 of 20**. Host PSS/RSS fail
> at **0.585/0.592×** normalized, while GPU memory and available-memory drop
> pass at **1.829/1.228×**. We will modernize or delete paths only when the
> trace proves them obsolete. The first trace's CUDA-13 default whole-graph
> capture was attribution-incomplete, but immutable `def5f75` has now completed
> the replacement one-lock node-level trace: **2,315,412** graph-child kernels
> plus **272,354** eager kernels are visible, the model gate passes, and all
> trace/cache/lifecycle contracts pass. It selects one concrete first repair:
> vLLM executes **208** FP4 GEMMs per forward step through packed `qkv_proj`,
> while our graph executes about **240** because q/k/v remain three GEMMs across
> 16 full-attention layers. W3-D now implements that packed resident weight /
> scale, max-shard CT scalar, one-GEMM and row-strided split-view path, with
> `VT_FP4_MERGED_QKV=0` restoring the three-GEMM arm. Immutable `3f256ab`
> correctness passes **235/235 assertions and 16/16 tokens on both arms**. Its
> packed/split c16/96 AB/BA/AB means are **812.231/808.150 tok/s = 1.005049×**
> with **14/20 timing and 2/4 memory axes**; packed also reduces sampled peak
> GPU memory by **294 MiB**. The strict component gate therefore remains open,
> while the trace-grounded/upstream-mirrored packed default is retained. The
> clean post-pack node trace passes and closes the structural target: ours is
> **296,674 / 1,425 = 208.192 FP4 GEMMs/forward** (208 plus 274 capture/warmup
> calls), while vLLM remains exactly **330,304 / 1,588 = 208**. The fresh exact
> grid improves the binding disposition by one axis, from 54/124 to **55/124**,
> but strict parity still fails. A required generated-code dump corrects the
> trace-name interpretation: vLLM's **127,040** kernels named with
> `fused_add_rms_norm_scaled_fp4_quant` perform residual-add + RMSNorm only,
> write BF16, and are followed by separate `scaled_fp4_quant`/`cvt_fp16_to_fp4`
> launches; `fuse_norm_quant` is false. Our residual-add RMSNorm + separate FP4
> quant topology therefore already matches. `KERNEL-EW-NORM-QUANT` is not
> promoted. The continued body-level scan selects W3-E direct swizzled
> activation scales: ours launches **320,099** standalone scale swizzles
> (**1.239 s**, 224.631 launches / 0.869 ms per forward), while vLLM writes that
> layout in its executed quant producers and has no standalone equivalent. The
> [spike](.agents/specs/nvfp4-direct-swizzled-scales.md) is now implemented and
> `GATING`: normal and both fused producers accept an explicit linear/direct
> layout, directly zero/write the padded CUTLASS scale buffer, and default to
> that path only for CUDA true-W4A4. `VT_FP4_DIRECT_SF=0` restores the former
> linear producer + standalone swizzle exactly. CPU/CUDA byte tests, focused
> zero-error memchecks, both 27B model arms (**235/235 + 16/16**) and the 35B
> correctness-only gate (**315/315**) pass. Paired real-model tracing removes
> **624** activation swizzles (**832→208**; the remaining 208 are one-time
> weight-layout work) without changing producer/GEMM topology. Immutable
> `53ab149` completes the required c2/c16 same-binary A/B: direct gains
> **1.002108× / 1.006222×** mean total throughput, but only **32/40 timing +
> 6/8 memory** axes pass. The component therefore fails and no speed credit is
> claimed; no conditional exact grid or 35B performance run follows.
> The required post-failure scan then found that vLLM v0.25 actually loaded a
> persistent 64-entry FlashInfer FP4 tactic cache and hit it before serving;
> our prior “file cache disabled” conclusion applied only to v0.24. Our fresh
> W3-E processes independently retuned and matched just **18--33/64** paired
> tactic IDs, so sub-percent A/B results are confounded. W3-C is now
> [spiked and `ACTIVE`](.agents/specs/nvfp4-persistent-plan-cache.md) to mirror
> v0.25 JSON load/save, metadata/stale rejection, atomic publication, exact
> FlashInfer-cache import, a frozen read-only benchmark mode, and the current
> **5,000-us** miss-timing method. C1 supplies the CUDA-free document/import
> layer; C2 now installs compatible plans into the live ready map before the
> dummy request, fingerprints CUDA/CUTLASS/tactic sources, tunes only misses,
> rejects frozen misses before readiness, publishes native plans only after a
> successful warmup, and reports the complete selected map. The exact oracle
> fixture loads **64/64 with 0 tuned / 0 rejected / 0 saved** in read-only mode;
> cache-hit capture, cancelled-vs-completed save lifecycle, compute-sanitizer
> (**0 errors**) and both 27B model arms (**235/235 + 16/16 each**) pass. Focused
> Release/ASan+UBSan/TSan pass **7/7 + 189/189**. The current full CPU attempts
> are **102/103 FAILED** only on the unchanged C-API early-stop timing flake,
> reproduced by the baseline binary; the other **102/102** and an isolated C-API
> rerun pass. Immutable `d211b8f` then passes the six-fresh-process stability
> gate: three direct-scale and three composed-fallback processes are
> byte-identical, each loads the same **64/64** native plans with zero tuning /
> misses / writes and passes **235/235 + 16/16**. The first C3 driver stopped
> when independent c2 runs produced only **2/6** equal texts, leaving all
> partial timing/memory values **VOID**. W3-C3R now resolves that predicate:
> direct and composed arms are **6/6 equal for all 128 output tokens** when sent
> sequentially with identical batch shape, while each arm is **0/6** equal to
> its own c2 output. Production-default vLLM v0.25.0, with
> `VLLM_BATCH_INVARIANT` unset, is likewise **0/6** equal between sequential and
> c2 execution. Upstream enables cross-batch determinism only in its explicit
> opt-in mode, which changes NVFP4/attention/norm dispatch. The old cross-run
> exact-text condition is therefore void/reclassified, not a direct-scale
> correctness defect; operator byte parity, both fixed 16/16 oracle gates and
> the controlled 6/6 long-output proof remain mandatory. Corrected C3 is ready
> to rerun all timing/memory arms. No ratio changes, exact grid or 35B
> performance are claimed. This is mandatory reproduction control, not an
> accepted steady-state speedup; opt-in batch-invariant execution is separately
> inventoried and unsupported locally today.
> Every 27B speed, latency and memory axis must
> pass before 35B performance runs; broader roadmap work—including newly explicit
> **DSpark** support—waits behind parity.
>
> The exact post-pin audit classifies **145** commits as **94 inventory / 51
> ignore / 0 immediate runtime ports** for the currently implemented Qwen T0
> slice. Dedicated live rows now cover DSpark, heterogeneous-vocabulary TLI,
> opt-in batch-invariant execution, the unified Streaming Parser Engine,
> non-DP MoE sequence parallelism, per-request response timings, and the three
> new v0.25.0 registry targets.

## What's implemented (CPU, behaviorally tested)

The vLLM **V1 engine path needed by the two Qwen gate-model text workloads**
runs end-to-end on CPU:

- **Engine core** — the unified token-budget `Scheduler` (chunked prefill, FCFS
  preemption), hash-based prefix-caching `KVCacheManager` + hybrid KV coordinator
  (full-attn + GDN/mamba-state groups), the persistent `InputBatch` + step-input
  build, `EngineCore.step()` loop, and the `LLMEngine` (`add_request`/`step`/
  `generate`).
- **Local-attention KV + sliding-window operator** — `SlidingWindowSpec` and
  `ChunkedLocalAttentionSpec` now have registry-backed managers, bounded
  admission, local prefix reuse/recycling, and hybrid-disabled full-allocation
  fallbacks, all CPU/property/sanitizer-gated. The generic sliding window also
  propagates through config, the CPU reference, separately specialized
  portable-CUDA kernels, and vendored FA2 local dispatch; the sm_121a
  translation units compile cleanly. Chunked-local attention now has its cached
  ordinary-backend wrapper, exact virtual Q/K batches, reusable block-table
  gather/update plan, cudagraph rejection and generic spec seam; all six pinned
  shapes plus randomized causal-mask properties are CPU/sanitizer-gated.
  Neither mode is yet user-visible model support: GPU runtime, positive-model
  e2e, oracle, trace, and performance/memory gates remain open.
- **Long-context RoPE foundation (YaRN/MRoPE/Llama 3/Phi-3/dynamic-NTK)** —
  modern and legacy Hugging Face RoPE parameters now normalize into a typed,
  memoized factory with f32/bf16 caches.
  Plain YaRN and its `mrope_section` branch apply caller-supplied caches for
  NeoX/GPT-J layouts and 1-D or 3-axis contiguous/interleaved positions; Llama 3
  adds its exact unchanged/smoothed/scaled frequency bands, including equal
  low/high factors. Phi-3 LongRoPE builds its short and long factor-array caches
  and selects one globally from runtime max length, including explicit mscale
  overrides. Dynamic-NTK mirrors both its factor/trained-length and alpha base
  transforms with pinned alpha-first dispatch. Fifteen exact pinned-source
  oracle fixtures, CPU references, and ASan/UBSan pass. The CUDA implementation
  is source-complete but has not yet been compiled or run on an uncontended
  GPU, and no feature-positive model path is claimed yet.
- **Model forward** — Qwen3.6-35B-A3B hybrid (GDN×3 + gated full-attention, 256-
  expert MoE + shared expert), with **paged attention** (block-paged KV cache) +
  batched GDN. Loads from **safetensors** (NVFP4/FP8→bf16) and **GGUF**
  (k-quant→bf16 for the qwen35moe APEX files). Model selection now uses a
  central ordered, type-erased registry over the full Hugging Face
  `architectures` list for both implemented Qwen IDs; unsupported,
  previously-supported, and out-of-tree IDs are rejected explicitly instead of
  being guessed from `num_experts`. The registry's CPU suite is green; its
  unchanged two-model GPU path is awaiting the queued no-regression rerun.
- **CPU execution** — persistent ggml-derived native threadpool with chunked
  GEMM and row/batch op dispatch; `VLLM_CPP_CPU_THREADS` selects the worker
  count. Outputs are byte-identical at 1/3/20 threads and the concurrency suite
  is ThreadSanitizer-clean. The real-file ≥10x speed/RSS checkpoint still needs
  an exclusive idle-host rerun, and GGUF compute-in-quant remains open.
- **Sampler** — greedy (bit-exact vs `torch.argmax`) and parallel exponential-
  race random sampling, plus temperature, top-k/top-p, penalties, min-p and
  internal logit-filter/logprob primitives.
  `n`, full random/logprob payload behavior, and request/API wiring for the
  long-tail controls remain open; some paths synchronize results to the host.
- **OpenAI server** — basic `/v1/completions` and `/v1/chat/completions`
  transport (non-streaming plus **live incremental SSE** over an `AsyncLLM`
  engine thread), concurrent requests through one scheduler, and disconnect
  abort. Streaming completion and chat requests accept `stream_options` with
  native-ID final/continuous usage frames; `--enable-force-include-usage`
  mirrors vLLM's server-wide force mode. This path is CPU/sanitizer-gated and
  awaits its fresh two-model GB10 online checkpoint before the row can close.
  The HTTP transport now provisions a fixed `max_num_seqs + 4` worker floor so
  every scheduler-visible SSE stream plus bounded health/discovery traffic has
  delivery capacity; the legacy dynamic pool is a diagnostic A/B opt-out only.
  `--max-num-seqs` and `--max-num-batched-tokens` expose reproducible scheduler
  operating points. `/v1/models`, `/health`, and `/version` are
  present, while `/health`
  currently reports process liveness rather than probing engine health. The
  async/API path passes focused Release, ASan+UBSan and ThreadSanitizer; the
  current full serial suite is 104/105 only on the recorded unrelated C-API
  callback-count flake, which passes in isolation. Post-change GB10
  token/latency/throughput/memory gates are still pending. Chat templates
  use a bounded minja-subset Jinja engine. `--[no-]enable-prefix-caching`
  exposes the tri-state cache policy; hybrid/attention-free generation models
  default off like pinned vLLM, while ordinary decoder models default on.
  **Tool/function calling** provides Hermes-style `<tool_call>` parsing,
  streaming tool-call deltas, and `tool_choice` auto/required/named — `auto`
  is *relaxed*: the model may reply in plain text or call a tool, constrained only
  once it starts a call; ⚠️ the Qwen3.6 gate checkpoints' shipped chat template
  uses the **Qwen3-Coder XML** tool format and forced `<think>` blocks, which the
  current parser (including the local `qwen3` Hermes alias) and template engine
  do not yet handle — the
  `qwen3_coder` XML parser + reasoning parser + template-engine extensions are
  scoped in `.agents/specs/mm-tools-scoping-2026-07-10.md`); **grammars / structured output** (JSON-schema,
  `json_object`, regex, choice, GBNF — a native constrained-decoding engine with
  an explicitly bounded JSON-schema/structural-tag subset behind vLLM's
  structured-output seam).
- **Library packaging** — a stable 17-symbol C ABI (`include/vllm.h`),
  `libvllm` shared + static, an example CLI and OpenAI server, and a
  `dlopen`/FFI consumption path (for LocalAI-style embedding via cgo/purego).
  Alongside the blocking completion calls, additive request handles provide
  nonblocking submit/cancel/wait/done/error/free with per-delta callbacks.

Every unit is adversarially reviewed and behaviorally tested; the CPU test suite
is green (`ctest`). See the [parity ledger](.agents/parity-ledger.md) for the
per-change record vs upstream vLLM.

## Quick start

```sh
# Build (CPU; add -DVLLM_CPP_CUDA=ON on an NVIDIA box, -DVLLM_CPP_SERVER=ON for the server)
cmake -S . -B build -DVLLM_CPP_SERVER=ON
cmake --build build -j
ctest --test-dir build            # the behavioral suite

# Fast-GDN CUDA build (the MVP-parity numbers): adds the Triton-AOT GDN kernels.
# They are VENDORED as pre-compiled cubins (src/vt/cuda/triton_aot_vendored/) —
# building needs ONLY a C compiler, no Python/Triton (regenerating the vendored
# artifacts is a maintainer task: scripts/regen-triton-aot.sh).
cmake -S . -B build -DVLLM_CPP_CUDA=ON -DVLLM_CPP_TRITON=ON

# Serve a supported Qwen text checkpoint (safetensors dir or supported 35B .gguf)
./build/examples/server --model /path/to/Qwen3.6-35B-A3B --port 8000 \
  --max-num-seqs 32 --max-num-batched-tokens 8192
# then: curl localhost:8000/v1/chat/completions -d '{"model":"...","messages":[...]}'
# add --enable-force-include-usage to attach native cumulative usage to every
# streamed choice and an empty-choice final usage frame before [DONE].

# Or a one-shot completion via the CLI (drives the C ABI end-to-end)
./build/examples/vllm-cli --model /path/to/model --prompt "The capital of France is"
```

As a library, link `libvllm` (or `dlopen` it) and drive the C API in
`include/vllm.h`: use `vllm_complete` / `vllm_complete_stream` for blocking
calls, or `vllm_request_submit` plus the request lifecycle functions for
nonblocking concurrent streams.

## Supported model architectures

| Architecture | Families | Safetensors | GGUF | Status |
|---|---|---|---|---|
| Qwen3.5/3.6 hybrid (GDN + gated attention, MoE + dense) | Qwen3.5-4B, Qwen3.6-35B-A3B, Qwen3.6-27B | ✅ **text submodels** run end-to-end on GB10 and retain token-exact greedy correctness. The binding v0.25.0 `3f256ab` 27B cache-off gate completed all 36 groups but failed strict parity at **55/124 axes**; c16/c32 total throughput pass while low-concurrency decode latency and host PSS/RSS remain open. Packed QKV closes at **208.192 vs 208 FP4 GEMMs/forward**. W3-E removes 624 hot swizzles but strict-fails its component. W3-C passes six-process frozen-map stability (**6/6**, identical 64/64 plans, zero tuning); C3R proves direct/composed **6/6 x 128-token equality** under identical sequential batch shape and reclassifies the earlier independent-run 2/6 check against vLLM's default-off batch-invariance contract. Corrected component timing/memory remains pending. Plain-BF16 4B loads/runs on local sm_120 and matches 15/16 representative 32-token greedy sequences (adversarial repeated-token corpora: 9/16 and 10/16), so it is not parity-qualified. The upstream wrappers are multimodal; their vision path is not implemented. | ✅ 35B text path from real APEX k-quant `.gguf` on GB10 (greedy parity vs same-file llama.cpp oracle); 27B GGUF pending (no file exists) | 🟡 paged-KV text engine + basic server/tool/grammar subsets; correctness gated, v0.25.0 27B production performance `FAILED/GATING`; 35B performance held |
| Qwen3 / Qwen2 dense | Qwen3-32B, Qwen3-0.6B, … | — | — | 🗓 planned (post-MVP T1) |
| Llama-family dense | Llama 3.x, Mistral | — | — | 🗓 planned (post-MVP T1) |
| MoE decoders | Mixtral, Qwen3-MoE | — | — | 🗓 planned (post-MVP T1) |

## Acceleration

| Backend | Hardware | Status |
|---|---|---|
| CPU | x86-64 reference (correctness/CI grade) | 🟡 gate-model text engine + basic serving path end-to-end; multithreaded op dispatch (ggml-threadpool port, `VLLM_CPP_CPU_THREADS`) is 1/3/20-thread bit-identical and TSAN-clean. Its B4 real-file speed/RSS gate is pending an idle-host rerun; compute-in-quant GGUF speed remains open |
| CUDA | NVIDIA Blackwell (GB10 sm_121a; RTX 50-series sm_120) | 🟡 **gate-model paged-KV stack running on GB10** with both greedy correctness gates passing. Immutable `3f256ab` binds at **55/124**. Packed QKV closes at **208.192 vs vLLM 208 FP4 GEMMs/forward**; W3-E removes 624 swizzles but strict-fails its component. W3-C imports/publishes the v0.25 64-plan map before warmup, uses the 5,000-us miss ABI, freezes misses, and passes six fresh processes with identical 64/64 plans and zero tuning. C3R proves the direct/composed path equal at 6/6 x 128 tokens with identical sequential batch shape and reproduces default-mode batch-shape dependence in vLLM; the stopped component remains void and its corrected timing/memory rerun is pending. No performance ratio changes. No 35B performance run is authorized before every 27B axis passes. Qwen3.5-4B BF16 executes locally on RTX 5070 Ti with device-resident caches, but local correctness and performance gaps remain. |
| Other CUDA targets | vLLM's sm70/75/80/86/87/89/90/100/101/103/110/120 targets | 🗓 inventoried, **not yet built or validated here**; per-target kernel dispatch/AOT/build/correctness/trace/performance gates remain |
| Metal | Apple Silicon via MLX; custom MSL/MLX primitives for paged ops | 🗓 planned (M4 bring-up host available) |
| Vulkan | Portable GPU | 🗓 planned (post-MVP) |
| SYCL / XPU | Intel GPUs | 🗓 planned (post-MVP) |

### Kernels & target hardware

The ops exercised by the two GB10 text gates have portable references and CUDA
implementations grounded in vLLM or its dependencies. This is not a claim of
complete vLLM-kernel or cross-architecture coverage: the canonical kernel and
CUDA-target inventories track unimplemented and untraced families separately.

| Op / kernel | Implementation (upstream it mirrors) | CPU | CUDA · Blackwell (GB10, sm_121a) |
|---|---|:--:|---|
| Dense **W4A4** GEMM | cutlass fp4×fp4 (sm120a) + fp8 W8A8 — vLLM `cutlass_scaled_mm` | ✅ ref | ✅ |
| MoE **W4A16** GEMM | Marlin + fp4-resident — vLLM `marlin` / `fused_moe` | ✅ ref | ✅ |
| FP8 / bf16 projection GEMM | cuBLASLt col-major-TN → `nvjet_sm121` | ✅ ref | ✅ |
| Prefill attention | **vendored FlashAttention-2** `flash_fwd_splitkv` (vllm-project/flash-attention @ 2c839c33, the exact kernel vLLM runs; default-on for the bf16 head-256 path, 3.7× vs our WMMA) with the flash-style WMMA kernel as the portable fallback | ✅ ref | ✅ |
| Decode attention (paged) | FlashInfer-style paged, GQA-fused | ✅ ref | ✅ |
| GDN / linear-attn (chunk) | tensor-core WY solve — FLA `chunk_delta` | ✅ ref | ✅ |
| RMSNorm(+residual) → fp4 quant | the **traced gate workload** dispatches Inductor `fused_add_rms_norm` + external `scaled_fp4_quant`; FlashInfer also contains fused Add/RMSNorm+FP4 kernels that remain in the upstream kernel inventory · fp8: fused `RmsNormQuantFp8` (35B) | ✅ ref | ✅ gate path |
| Activation fp4 quant | HW `cvt.e2m1x2` PTX (vLLM `nvfp4_utils`) / software ladder | ✅ ref | ✅ ladder · HW-PTX 🚧 A/B |
| CUDA-graph decode | Qwen-specific captured decode step (vLLM cudagraph) | — | 🟡 explicit 35B capture gate; 27B evidence backfill open |
| Sampling (greedy/random/top-k/top-p/penalties) | vLLM V1 ordering subset; on-GPU multi-block random reduction, while token/logprob results still synchronize to host | ✅ | 🟡 bounded subset |
| RMSNorm / default RoPE / SwiGLU | fused elementwise | ✅ | ✅ |
| Scaled RoPE supplied-cache rotation | pinned vLLM typed cache/factory + YaRN/MRoPE/Llama 3/Phi-3 LongRoPE/dynamic-NTK construction and cache lookup/rotation | ✅ ref + fifteen oracle fixtures | 🚧 source present; compile/runtime gate open |

Only **GB10/sm_121a is built, traced and gated today**. Source-level fallbacks
suggest routes for other SMs, but none counts as support until its full build,
correctness, trace and performance block passes. Non-CUDA backends
(Metal/Vulkan/ROCm/XPU) are post-MVP and will port from the CPU reference.

## Quantization

| Format | Status |
|---|---|
| NVFP4 (W4A16 MoE / W4A4 dense, Blackwell) | ✅ **both running on GB10** with token-exact greedy gates passing. The W4A4 path includes all 32 SM12 tactics, merged/fused projections, pre-serve tuning and packed QKV. W3-E removes 624 activation swizzles but strict-fails its component. W3-C wires the pure-C++ native/FlashInfer cache into CUDA startup: exact 64-plan import is a zero-tune capture-safe hit, frozen misses fail, completed warmups publish atomically, and six fresh processes load one identical map. C3R proves direct/composed **6/6 x 128-token equality** when batch shape is held fixed; the earlier independent c2 texts exercised an opt-in batch-invariance property absent from the production vLLM denominator. The stopped run remains void and corrected performance evidence is pending. The `3f256ab` grid still fails **69/124** axes; no speed or support expansion is claimed. |
| GGUF materialization (F32, Q4_0, Q8_0, Q3_K/Q4_K/Q5_K/Q6_K) | 🟡 load-time bf16 materialization; synthetic layout tests plus real 35B APEX Q3/Q4/Q5/Q6/Q8 greedy parity vs same-file llama.cpp. CPU ops now use correctness-gated multithreaded dispatch, but its real-file speed/RSS gate and direct compute-in-quant path remain open; F16/BF16, Q2_K, IQ/TQ/Q1, MXFP4 and NVFP4 execution remain open. |
| FP8 | 🟡 the 35B ModelOpt static per-tensor W8A8 projection slice is native and gate-passing; generic FP8 modes/dispatch and FP8 KV remain planned |
| MXFP4 / MXFP8 | 🗓 planned, including MLX-native modes on Apple |

Legend: ✅ supported & tested · 🚧 in development · 🗓 planned.

## Status & caveats (honest)

- **Where we are (GB10 / DGX Spark, sm_121a):** both NVFP4 gate-model text paths
  run the full paged engine end-to-end, and their real-model **token-exact greedy
  correctness gates pass**. The CUDA kernel paths exercised by those gates are
  validated on real hardware.
- **Production-vLLM performance has a fresh v0.25.0 27B baseline and remains
  open.** The
  completed `b5c6e4f` grid remains reproducible history, but its vLLM 0.24.0
  environment carried FlashInfer 0.6.12 while the source pin already required
  0.6.13. The follow-up `3cc490c` run was intentionally stopped and is
  **VOID** at 28/36 groups, 1,602/2,016 timed requests, four returns and no
  paired trace. The canonical v0.25.0 oracle now passes package/import/CLI,
  lock-held real-model production-graph and clean server health/completion
  smoke checks. Immutable `3f256ab` then completed its model gate, all 36 timed
  groups, 2,016 requests, six memory returns and the paired trace. Every one of
  the 124 axes is binding-eligible; **55 pass and 69 fail**. Median total ratios
  are **0.994/0.954/0.966/0.981/1.028/1.039×** from c1 through c32. The run is
  stable (maximum total-throughput CV **0.189%**), so the low-concurrency and host
  memory gaps are reproduced rather than dismissed as noise. The original
  whole-graph trace was insufficient, but replacement `def5f75` passes the
  node-level contract with **2,315,412** graph-child and **272,354** eager rows.
  Matched launch counts promote packed full-attention QKV: vLLM's 64-layer
  forward has exactly **208** FP4 GEMMs per step, while our 16 separate q/k/v
  layers produced about **240**. W3-D now implements the one-GEMM packed path.
  Immutable `3f256ab` default/fallback correctness passes 235/235 + 16/16; its
  c16/96 component improves mean total throughput **0.5049%** and sampled GPU
  memory, but only **14/20 timing + 2/4 memory** axes pass. The post-pack trace
  then confirms **296,674/1,425 = 208.192** FP4 GEMMs/forward versus vLLM's
  exact 208. The completed exact rerun gains one axis versus `9cc7191` but remains
  failed/open. Its paired trace names **127,040** kernels with an apparent
  Add+RMSNorm+FP4-quant fusion, but the generated body performs only residual
  add + RMSNorm into BF16; a separate custom FP4 quant launch follows. The
  name reflects the topologically sorted source group, not one fused kernel.
- **Modernization and removal are trace-gated.** The v0.25.0 audit found no
  copied legacy `paged_attention_v1/v2` source and no MRV1 execution path to
  delete. The live `vt::PagedAttention` name denotes a backend-neutral paged-KV
  operation and remains required. v0.25.0 also preserves direct swizzled FP4
  scales, zeroed unread padding and a device-resident `alpha` pointer, so the
  packed FP4 topology repair was not obsolete. The required generated-code
  inspection refutes the apparent norm+FP4 lever. Continued body/source
  inspection selects only the independently executed direct-scale gap: local
  `SwizzleBlockscaleKernel` is 320,099 launches / 1.239 s while vLLM writes the
  swizzled address inside quant. W3-E now implements that direct address for
  normal/two-input/one-input producers and preserves the composed fallback.
  Byte parity, sanitizer, both 27B arms and the 35B inertness gate pass; a paired
  trace removes 624 activation swizzles. Immutable `53ab149` then completes all
  12 c2/c16 AB/BA/AB legs, 612 requests and 12 memory returns. Direct improves
  mean total throughput by **0.211% at c2** and **0.622% at c16**, but strict
  acceptance **fails at 32/40 timing + 6/8 memory axes**. Five of six paired
  128-token text hashes differ amid highly unstable per-process tactic choices;
  both model gates retain 235/235 assertions and the fixed 16/16 oracle stream.
  W3-E stays `GATING`, receives no accepted speed credit, and does not trigger
  an exact-grid or 35B performance run.
  The follow-up scan proves v0.25 loaded a persistent 64-plan cache whereas
  ours retuned each process. W3-C remains `ACTIVE` as mandatory same-plan
  reproduction control: C2 runtime publication and frozen-map correctness pass,
  and immutable `d211b8f` proves six-process **64/64** stability. Its first
  paired c2 arm then uses the same tactic map but stops at **2/6** cross-run
  text equality, so all partial rates remain void. C3R proves 6/6 x 128-token
  equality under an identical sequential batch shape and reproduces 0/6
  sequential-vs-c2 equality in both arms and production-default vLLM. The
  cross-run predicate is reclassified; corrected C3 is ready to rerun.
  No path is modernized or removed from a trace name alone. Every 27B throughput, latency and
  memory axis must close before 35B; DSpark and the rest of roadmap_v1 stay
  queued behind speed parity.
- **The competitor floor is workload-specific.** Source-auditing the referenced
  DGX Spark recipe confirmed its withdrawn 10--40x claim compared SGLang with
  radix caching on against vLLM with prefix caching explicitly off. Its
  remaining reported 25--45% cache-on lead is not yet project evidence: it is
  35B-only, uses vLLM 0.23.1 for the cache-on cells, mismatches BF16/FP8 KV and
  memory policies, enables MTP, has one or two repetitions, and omits several
  required axes, hit/eviction proof, memory, correctness and paired traces.
  Cache-neutral serving and deterministic 64k/256k shared-prefix serving now
  have distinct canonical gates. Ours, explicit vLLM v0.25.0 Qwen hybrid
  `mamba_cache_mode=align`, and SGLang v0.5.15 will receive identical
  checkpoints, BF16 cache dtype, capacity, no-spec policy, token corpus,
  reset/seed order and native hit proof; the faster equivalent reference binds
  each throughput, latency and memory axis. **No SGLang prefix-cache number or
  local Mamba-align support is claimed yet.**
- The optimized paths themselves remain implemented: vendored Triton-AOT GDN,
  cuBLASLt TN projection layouts, fused attention preamble, tiled causal-conv,
  vendored FlashAttention-2 prefill, register-tiled GDN `delta_h`, tensor-core WY,
  fused fp8 RMSNorm→quant, 27B-default BF16 GDN core/z output (35B unchanged),
  and default-on fp4 autotuning. The
  current FP4 implementation now gives M=1/2/4/8/16 distinct hybrid buckets, a
  complete plan key, single-flight tuning and capture-miss rejection, while
  retaining the aliased cache only under `VT_FP4_EXACT_BUCKETS=0`. W2 adds the
  complete eight-tile × two-orientation × two-scheduler SM12 family, merged
  gate/up resident weights and CT maximum-divisor/one-alpha semantics, plus the
  traced one-input activation/quant producer. The accepted
  [implementation spike](.agents/specs/nvfp4-small-m-dispatch.md) inventories
  the exact hybrid mapping, eight tiles × two operand orientations × two
  schedulers, upstream tests, workspace/capture rules and separate W1/W2 A/B
  gates. W1 passes focused Release stress, ThreadSanitizer and pushed-`c8807b0`
  exact/legacy CUDA capture, memcheck and 27B model gates. Its `bce2627`
  classification and clean `b5c6e4f` remain W1/W2 historical diagnostics from
  the superseded vLLM 0.24.0 oracle. W2 is focused-CUDA,
  compute-sanitizer and native-27B correctness green in its shipping and fused
  fallback arms. The one-input fusion's c16 component is +1.521%, but its strict
  17/20 timing and 0/4 sampled-memory result is not closure; the paired trace's
  +2.084%/20-of-20 result and launch reduction remain component diagnostics.
  The W2 campaign was below the all-axis floor and its trace promoted
  W3 planner/warmup parity rather than another tactic implementation. W3-A
  historically implements the v0.24 pre-event stream sync plus one-thread 1-ms nanosleep and
  reports the timing arm with each selected stable ID. Clean immutable
  `71f1e89` links on sm_121a; focused delayed/off each pass 14/14 and
  18,619/18,619, both 27B arms pass 235/235 + 16/16, and delayed memcheck is
  16,389/16,389 with zero errors. Its timed c16 component is +0.816% in mean
  total throughput but fails strict acceptance at 13/20 timing and 2/4 memory;
  only 5/35 common delayed keys retain one ID across three processes. W3-B now
  implements the pre-serve scope, exact 2,048-bucket enumeration and
  diagnostic-only later misses in the shared library loader. Clean `d7cdf66`
  passes sm_121a build, CPU/registry/loader, exact/legacy capture, native-27B,
  memcheck and server-readiness gates. Its repeated c16 component is neutral
  at **1.000293×** and strict-fails **15/20 timing, 2/4 memory**; only **20/80**
  prewarmed keys are tactic-stable, although first-use latency improves.
  Corrected paired tracing now closes the original FP4 structural gap at
  **110.623/109.932 s** prewarm/vLLM and leaves diagnostic normalized TPOT at
  **0.9673x**. The old-oracle `3cc490c` grid is VOID. Its first v0.25
  replacement `9cc7191` remains historical at 54/124; the current binding
  c1-c32 campaign is complete from immutable `3f256ab`. Plan/source/
  vLLM-corpus SHA-256 are `0e309d8b…9999`, `41bd634a…fd7a`, and
  `b048d789…e5dc`. One metadata-only
  invocation first failed before writing output because it omitted the repo
  module path; the corrected module invocation passed. The model gate, all
  **36/36** groups, **2,016/2,016** requests and six lifecycle/cache returns
  pass their evidence contracts. The scoped summary accepts all
  **124/124** axes but only **55/124** pass; therefore no parity claim is made.
  Generated-text equality is diagnostic for FP4 near-ties: the commit-bound
  16/16 token gate and exact native 128-token counts remain the correctness
  precondition. The recorded trace status passed the original graph-level
  contract, but the command lacked `--cuda-graph-trace=node`; SQLite proves
  **1,226** graph launches and **0** child-node kernel rows. The replacement
  `def5f75` `--trace-only` checkpoint passes its model gate and paired-profiler
  contract under one lock, retains **2,315,412** graph-child rows, and selects
  packed QKV from the observed **~240 versus 208 FP4 GEMMs/step** topology. The
  completed `3f256ab` trace validates the repaired 208 topology. Its apparent
  norm+FP4 fusion is refuted by the generated body and separate custom-op call.
  The remaining trace then selects W3-E direct scale emission from the local-only
  standalone swizzle family. Implementation/correctness/sanitizer/trace now
  pass, with 624 activation swizzles removed. The completed c2/c16 component is
  mean-positive but strict-fails at 32/40 timing + 6/8 memory; its conditional
  exact grid did not run. vLLM logged an unavailable optional
  `triton_kernels.matmul_ogs` import for GPT-OSS/MXFP4; the executed dense-27B
  dispatch used FlashInfer NVFP4, FLA/Triton GDN and FA2, so the warning is
  recorded as non-path evidence rather than an environment mutation.
  Historical same-binary component A/Bs remain evidence for those individual
  levers; they do not substitute for the reopened end-to-end oracle gate. The
  full record is in the [parity ledger](.agents/parity-ledger.md), and
  `scripts/dgx-bringup.sh` runs the CUDA suite and gates on dgx.casa.
- **Device-resident cache work is active.** W0 gives CUDA full-attention KV and
  GDN convolution/recurrent state stable `vt::Alloc` ownership and
  stream-ordered zeroing. W1 adds persistent device metadata plus one indexed
  BF16↔F32 gather/scatter launch per mixed GDN layer, including fresh-state
  zeroing and mixed decode/prefill turnover semantics; CPU retains host storage.
  `VT_DEVICE_KV_CACHE=0` restores the former CUDA host-vector path for an exact
  W0 A/B, while `VT_GDN_INDEXED_STATE_IO=0` restores W1's row-copy baseline.
  The current local serial CPU suite passes 105/105, as do focused ASan/UBSan
  access suites and a clean CUDA 13.0.88/sm_121a build. A strict local model
  LeakSanitizer run still
  reports 58,624 bytes in 153 process-lifetime scratch-pool allocations; the
  indexed op itself is leak-clean.
  Both 35B/27B tests pass in default indexed mode and in fallback mode; the W0
  default tests assert every cache pointer is CUDA device memory. On
  the exact cache-off c16/48 workload, three interleaved repetitions average
  **781.80 indexed vs 776.95 fallback tok/s (+0.6246%)** for W1, with all 20
  throughput+latency axes better, 0.846%/0.530% CV, and all six server
  lifecycles returning memory. Its traces collapse `cudaMemcpyAsync` calls from
  **163,540 to 7,508**, D2D calls from **142,717 to 1,231**, and D2D traffic from
  **49,088 to 1,856 MB**. The profiler perturbs the two arms unequally and
  inverts their throughput, so only the unprofiled AB/BA/AB series supplies the
  speed ratio; the trace supplies structural evidence. Separate W0 27B/35B
  compute-sanitizer runs pass 234/234 and 315/315 assertions with zero access
  errors, and the W1 indexed-op memcheck passes 7/7. Full leak checking remains
  red on pre-existing bounded
  caches (47.29 MB dense; 36.82 GB MoE/Marlin); W0's multi-GiB cache allocations
  are absent from both leak inventories and all server lifecycle returns pass.
  This component win does not close the full-grid vLLM throughput/host-memory
  gate.
- **Local RTX 5070 Ti / sm_120 status:** `Qwen/Qwen3.5-4B` now loads ordinary
  BF16 weights and runs end-to-end in the Nix CUDA shell. With all Triton AOT
  paths enabled it matches vLLM on **15/16 complete 32-token sequences** from a
  representative natural-prompt corpus. The one mismatch begins at a BF16
  top-logit tie that vLLM reports as exactly equal; adversarial repeated-token
  corpora remain 9/16 and 10/16, so this is **not full parity qualification**.
  Ordinary BF16 linears now retain torch's raw `[N,K]` layout and BF16 outputs,
  and a default-on sm_120 Triton AOT decode recurrence cuts that kernel from
  2.42 s to 0.64 s per trace window (about 192 to 51 us/call). The fused
  attention preamble now also defaults on for plain BF16 (FP8 remains off),
  replacing split + q/k norm + RoPE with one kernel and one shared cos/sin cache;
  `VT_FUSE_ATTN_PREAMBLE=0` restores the prior path.

  On the exact 128-request, concurrency-32, 1024-in/128-out workload, ordinary
  BF16 MLP gate/up and GDN B/A weights are now stacked at load and each pair is
  executed as one GEMM, mirroring vLLM's merged parameters. Two reproduced
  post-rebase temperature-1 runs average **6,246.86 tok/s** versus fresh vLLM
  **6,681.13 tok/s (0.935×)**; two greedy runs average **6,430.56 tok/s**
  versus fresh vLLM **6,715.54 tok/s (0.958×)**. These are open gaps, not
  performance parity.
  The original memory campaign found equal VRAM but a severe host gap: 12.59 vs
  4.15 GiB loaded PSS and 19.77 vs 7.39 GiB launch peak. The implemented
  `ENG-HOST-WEIGHT-RESIDENCY` W1-W3 now mirror vLLM ownership: tied logits share
  embeddings, gate/up and GDN B/A retain only canonical stacked storage, and an
  engine-owned plain-BF16 model is uploaded/released during model preparation.
  `VT_RELEASE_HOST_WEIGHTS=0` retains the W2-only control; CPU, UMA, borrowed and
  quantized release behavior is unchanged.

  At commit `6c30657`, three interleaved repetitions give steady GPU-resident
  process PSS **0.752 GiB versus 4.097 GiB vLLM (0.184×; 81.6% lower)** and peak
  process VRAM **11,689 versus 12,924 MiB (0.904×)**. Launch peak PSS improves
  to **15.55 GiB (-21.4%)** but remains above vLLM's **6.69 GiB (2.32×)**, so
  memory parity as a whole remains open. The W4 spike identifies cumulative
  resident pages from all-shard mappings as the first target. The first
  whole-mapping advice placement is **REJECTED**: its initial monitored leg was
  interrupted still pre-GPU at 173.75 s versus about 25 s for the prior complete
  run. Its 11.41-GiB partial peak is `VOID`, not an accepted improvement.
  Tensor-range-only advice is now pending; `VT_SAFETENSORS_DISCARD_PAGES=0`
  remains the same-binary control. Direct-device loading remains
  required if the owned-model floor still exceeds vLLM. Prepare-time release
  is **+1.74%** versus OFF and reaches **0.9754×**
  fresh-vLLM total/output throughput; performance parity also remains open.
  Full CPU CTest passes 105/105 and focused native-sm_120 CTest passes 4/4.
  W4's rejected implementation passed focused native-sm_120 default 5/5 and
  opt-out 2/2 tests, but load-time regression overrides that correctness gate.
  No improved memory or throughput number is claimed.
  The recurring NVIDIA
  `refcntRequestReference_IMPL ... status 0x00000056` kernel notice is now
  source-identified as an unsupported profiler request to change Blackwell's
  GR timestamp tick frequency. It is not an invalid-pointer or GPU-reset
  status; no adjacent Xid/UVM/AER fault or benchmark failure was observed.
  Random sampling uses the same exponential-race distribution as vLLM and has
  an empirical distribution test, but deterministic token streams differ because
  the project still uses SplitMix while vLLM uses torch Philox. The fresh Nsight
  trace puts random sampling plus softmax at 2.4% of project GPU time. Packed
  gate/up removes one GEMM per MLP invocation; a matched trace removed 1,760
  launches and 1.92% of GPU kernel time. GDN B/A removes one logical GEMM per
  GDN invocation; its wider cuBLASLt tactic adds split-K reductions, so the
  matched trace removes 624 physical launches and 0.62% of GPU kernel time.
  Attention QKV and GDN QKV/Z merge trials were reverted because they worsened
  the current 15/16 greedy result. Ordinary BF16 GEMMs and eager launch
  structure still dominate. A matched post-packing short Nsight run attributes
  the 1.03 s GPU-time gap versus vLLM roughly half to GEMMs (0.47 s) and half
  to non-GEMM kernels (0.56 s); host/GPU-uncovered time is already equal. The
  largest concrete non-GEMM gaps are local attention (the sm_120 build does not
  compile the vendored FA2 path) and GDN conv/post-conv kernels. A trial of the
  existing dense CUDA graph on BF16 preserved the 15/16 short-corpus result but
  segfaulted in `libcuda.so` under sustained load, so that routing was reverted.
  Persistent KV/GDN caches are now true device allocations on discrete GPUs,
  eliminating the prior HMM/UVM migration path; sustained runs and both engine
   traces completed without Xid/UVM faults.
- **CPU status:** the native threadpool and chunked op dispatch pass the complete
  suite at 1/3/20 workers and a focused ThreadSanitizer run. This is not yet a
  speed claim: the required same-binary 1-vs-20 Qwen3.5-2B GGUF throughput/RSS
  series was deferred because the recovered 20-core host was contended by
  unowned inference services. It remains `GATING` until an exclusive idle-host
  run reproduces the ≥10x prefill/decode floor and refreshes llama.cpp.
- No PyTorch / no ggml at build or runtime (ggml is a *format* reference for GGUF
  only). Original runtime/packaging components (the `vt` op runtime, the minja
  chat-template engine, the native grammar backend, the C API) are documented as
  deviations in the [porting inventory](.agents/porting-inventory.md) §9.
- The CUDA low-concurrency comparison against SGLang remains pinned to v0.5.13
  for its already implemented P1 corpus/client/summary/process-memory/dry-run
  evidence. A distinct shared-prefix gate pins SGLang v0.5.15 (`f63458b`) and
  the cited DGX recipe at `03253ef`; it is fully spiked and `READY` for its PX1
  harness/counter work plus `KV-MAMBA-ALIGN`, not implemented support. **No
  image, model compatibility, accepted SGLang performance result, or local
  hybrid cache-hit result is claimed yet.** The unchanged 27B checkpoint has
  a plausible native path; exact 35B mixed-quant loading must be proven, and
  pinned-client raw E2E/TPOT detail remains a loud preflight gap. True
  incremental async HTTP streaming exists, but binding TTFT/ITL numbers wait
  for the post-W2 GB10 online gate (`SERVE-ASYNC-LLM` is `GATING`, not
  `DONE`). Its fixed capacity-derived HTTP pool is CPU/sanitizer-green; the
  same-binary c32 GPU series is throughput-neutral and the fresh exact ladder
  has no unread socket, while the broader every-axis row remains open.
  Historical `31d053f` and offline ratios
  are non-binding because the
  audit found cache, sampling, token-budget, admission, and model-length
  mismatches. Pushed `b5c6e4f` is the last complete 27B cache-off, closed-loop
  historical checkpoint: 2,016/2,016 timed requests, three repetitions, six memory returns,
  the real-model gate, and paired traces pass their contracts. Median c1→c32
  total ratios are **0.993/0.952/0.966/0.976/1.021/1.022×**; only
  4/4/5/4/17/14 of 20 performance axes and 2/4 memory axes pass. All fixed c32
  legs are healthy. Exact-bucket/single-flight W1 is fully classified: its
  component series improves c8/c32 but fails strict c16/memory no-regression.
  W2 ports and correctness-gates the 32-tactic family together with merged CT
  gate/up scaling and fused SiLU→NVFP4 quantization. Its historical trace
  identifies selection parity as W3: vLLM resolves the narrow 128x32x256
  Stream-K/static pair while our planner predominantly selects wide tactics.
  W3-A delayed-event timing is immutable correctness/safety-green but its
  +0.816% component fails strict 13/20 timing and 2/4 memory acceptance, with
  only 5/35 delayed keys stable. W3-B is immutable build/correctness/safety-
  green at `d7cdf66`, with 80/80 profiles and zero later misses. Its repeated
  component is **1.000293×**, strict-fails **15/20 timing and 2/4 memory**, and
  leaves tactic selection unstable at **20/80** keys; it receives no speed
  credit. Its corrected trace nevertheless closes the original structural
  selection mismatch: FP4 kernel time is **110.623 s** versus vLLM's
  **109.932 s**, with the narrow pair at **70.333/70.986 s**. The same trace's
  normalized mean TPOT is still **0.9673×**, so a clean full-grid rerun is
  mandatory against v0.25.0. The binding v0.25 log corrects that historical
  cache classification: its 64-entry file cache is active, and W3-C now owns
  the persistent/frozen-plan parity control before another sensitive A/B. C2
  runtime/frozen-map correctness and six-process stability pass. The first
  same-plan c2 pair stopped at the old 2/6 cross-run equality predicate. C3R
  proves 6/6 x 128 equality with matched sequential batch shape and reproduces
  default-mode batch-shape dependence in vLLM, so corrected C3 is ready; the
  stopped rates remain void and the exact grid is still blocked on its result.
  W0 is
  merged at `7d29e0c`; its clean GB10 build,
  default/fallback
  35B+27B pointer/token gates, lifecycle, trace, and repeated 27B A/B pass. The
  measured W0 same-binary gain is +2.12% (20/20 timing axes). W1 then wins a
  separate same-binary A/B by +0.6246% (20/20) and collapses the traced copy
  loop from 163,540 to 7,508 async calls. Neither component result is a fresh
  vLLM denominator or may be multiplied into the old grid. Both model paths are
  W0 memory-access clean and the W1 indexed op is memcheck-clean, but the
  required zero-leak result remains open on inherited process-lifetime pools.
  Exact v0.25.0 oracle activation, the immutable `3f256ab` 27B grid and the
  `def5f75` node-level paired trace and W3-D immutable correctness/A-B are
  complete; the A/B is mean-positive but strict-fails at 14/20 timing + 2/4
  memory. The post-pack trace closes 240→208 structurally; exact-grid and trace-driven
  repair of all 69 failed axes, fixed HTTP capacity and fresh every-axis 27B
  closure remain mandatory before 35B and later roadmap work.
  W2 direct indexed
  convolution-state update stays scoped until those confirmed causes are
  removed and the residual trace is re-ranked.
- **External KV-cache providers are roadmap-only, not supported yet.** Roadmap
  v1 now has a dedicated `KV-EXTERNAL-CACHE` outcome: mirror vLLM's
  producer/consumer/both connector ABI, scheduler/worker load-store lifecycle,
  dynamic external-provider seam and failure policy, then make the official
  [LMCache MP and in-process workflows](https://docs.lmcache.ai/getting_started/quickstart.html)
  work end-to-end. The first binding gates will cover shared-prefix
  store/retrieve, two-engine reuse, Qwen3.6 hybrid-cache behavior, correctness,
  TTFT, transfer throughput, memory, failure recovery and observability. No
  local connector implementation or LMCache benchmark is claimed today.
- **Speculative decoding is not user-visible yet.** The first MTP leaf now has
  safetensors loaders and a standalone dense/MoE Qwen3.5 head with CPU tests,
  but its exact 27B+35B oracle gate is still queued and the scheduler,
  rejection sampler, GDN state snapshots, and API/config wiring remain open.
  GGUF files do not currently carry `mtp.*`; normal non-speculative execution
  is unchanged.
- **Local attention is not user-visible yet.** Sliding-window KV bookkeeping
  and backend-neutral CPU/portable-CUDA/FA2 compute-window leaves are
  implemented; chunked-local KV sizing, registry/grouping, fixed-chunk prefix
  reuse/recycling, admission and fallback allocation are also implemented, as
  is the virtual-batch wrapper that delegates each fixed chunk to an ordinary
  causal backend. Their CPU/property/sanitizer gates pass, and the
  sliding-window sm_121a translation units compile. GPU runtime, supported
  StarCoder2/Gemma3/Llama4-class model paths, oracle, trace, throughput, latency
  and memory evidence remain open; all four execution rows remain `GATING`, not
  supported.
- **Scaled long-context RoPE is not user-visible yet.** Typed config, memoized
  f32/bf16 cache construction, plain/MRoPE YaRN, Llama 3 frequency-band, Phi-3
  LongRoPE short/long, and both dynamic-NTK formulas, plus the CPU supplied-
  cache operator, pass fifteen exact pinned-source oracle fixtures and
  sanitizer checks. CUDA compile/runtime, Nomic/Qwen-VL YaRN, Llama-3.1,
  Phi-3, and Hunyuan consumers, both-engine traces, and correctness/performance/
  latency/memory closure remain open; all four scaled-RoPE execution rows are
  `GATING`, not supported.

## Project record

Development is fully documented under [`.agents/`](.agents/) (canonical index in
[AGENTS.md](AGENTS.md)): the [gates](.agents/gates.md), the [porting inventory](.agents/porting-inventory.md)
vs upstream vLLM, the [parity ledger](.agents/parity-ledger.md), the [roadmap](.agents/roadmap_v1.md),
the [engine](.agents/engine-matrix.md), [model](.agents/model-matrix.md),
[quantization](.agents/quantization-matrix.md), [kernel](.agents/kernel-matrix.md),
and [backend](.agents/backend-matrix.md) matrices,
and the [upstream sync protocol](.agents/upstream-sync.md). The concise public
accepted/pending/failed/void benchmark checkpoint is in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## License

Licensed under the [Apache License, Version 2.0](LICENSE). See [`NOTICE`](NOTICE)
for third-party attributions (this project is a C++ port of the Apache-2.0-licensed
[vLLM](https://github.com/vllm-project/vllm) and vendors additional components under
their respective licenses).
