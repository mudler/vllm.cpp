# vllm.cpp

A 1:1 port of [vLLM](https://github.com/vllm-project/vllm) to pure C++ —
no Python, no PyTorch, no ggml at runtime. It mirrors the V1 architecture and
targets the same algorithms and serving surface; the tables below distinguish
the implemented subset from the remaining parity work. It is usable as a
library (llama.cpp-style) with a C API, an example CLI, and an
OpenAI-compatible server.

> ⚠️ **Pre-release, under heavy development.** Both NVFP4 gate models —
> **Qwen3.6-35B-A3B** (MoE hybrid) and **Qwen3.6-27B** (dense W4A4) — now run the
> full paged engine end-to-end on **NVIDIA GB10** (DGX Spark, sm_121a) with
> **token-exact greedy correctness gates passing**. Performance parity is
> currently **GATING**. The exact pushed-`bce2627` cache-off, closed-loop 27B
> online checkpoint validates all 12 performance groups but remains below the
> every-axis floor: median total-throughput ratios at c1/2/4/8/16/32 are
> **0.968/0.932/0.940/0.952/0.994/1.007×**, with only
> **4/4/5/4/4/12 of 20** throughput+latency axes and **2/4** memory axes
> passing. The fixed `max_num_seqs + 4` HTTP pool completed all three c32 legs
> without the old unread-socket tail. Its separate same-binary fixed/legacy
> c32 AB/BA/AB is a steady-state-neutral **0.999764×** (8/20 axes; all 1,152
> requests and six memory returns pass); the legacy arm did not reproduce the
> rare stall in that bounded series, so no speed gain is attributed to the
> transport repair. The next confirmed c1-c32 gap is kernel-side: the
> original FP4 key aliased M=1/2/4/8/16 and its four wide tactics still omit
> the FlashInfer narrow static-persistent/Stream-K family observed in vLLM. The
> accepted [FP4 repair spike](.agents/specs/nvfp4-small-m-dispatch.md) splits
> exact hybrid buckets/single-flight tuning from the 32-tactic SM12 family;
> W1 is implemented behind `VT_FP4_EXACT_BUCKETS=0` and passes focused Release
> 100/100 plus ThreadSanitizer (9 cases / 615 assertions). Exact pushed
> `c8807b0` clean-builds for sm_121a; exact and legacy CUDA capture suites each
> pass 10/10 cases and 18,333/18,333 assertions, focused memcheck passes 1/1
> with 0 errors, and both 27B model arms pass 234/234. Its same-binary
> exact/legacy c1-c32 ratios are **1.0002/1.0012/1.0006/1.0093/0.9996/
> 1.0072×**: c8/c32 improve, but the strict all-axis and memory component gates
> fail. The trace proves independent exact plans and reduces aggregate FP4
> kernel time by 1.05%; the fresh oracle trace still attributes 25.1% of
> profiled kernel time to vLLM's absent 128x32x256 Stream-K/static pair. W2's
> full 32-tactic family is therefore active. The 35B
> campaign is held until 27B passes every axis. Historical
> temperature/token-budget-mismatched ratios
> remain diagnostics only. The tables track real, tested support and are kept
> current as work lands (see
> `.agents/`). CPU multithreaded dispatch is now
> correctness-gated, but its real-file throughput/RSS checkpoint is still open.
> The current serving checkpoint implements native-ID final/continuous stream
> usage for completion and chat and is CPU/ASan/UBSan/TSan-green; the fresh
> two-model online latency rerun remains `GATING`. The prior `8289cbd`
> partial 27B arm is **void** and contributes no public ratio.

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
- **Sampler** — greedy (bit-exact vs `torch.argmax`) plus temperature,
  top-k/top-p, penalties, min-p and internal logit-filter/logprob primitives.
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
| Qwen3.5/3.6 hybrid (GDN + gated attention, MoE + dense) | Qwen3.6-35B-A3B, Qwen3.6-27B | ✅ **text submodels** run end-to-end on GB10; token-exact greedy correctness gates pass. The exact `bce2627` 27B online checkpoint is valid but below the all-axis floor; FP4 W1 is fully measured, closes c32 total throughput, and narrows c16, but only 4/4/5/4/4/12 of 20 axes pass. W2 full-tactic work is active and 35B performance remains held behind 27B closure. The upstream wrappers are multimodal; their vision path is not implemented. | ✅ 35B text path from real APEX k-quant `.gguf` on GB10 (greedy parity vs same-file llama.cpp oracle); 27B GGUF pending (no file exists) | 🟡 paged text engine + basic server/tool/grammar subsets; correctness gated, production-vLLM performance `GATING` |
| Qwen3 / Qwen2 dense | Qwen3-32B, Qwen3-0.6B, … | — | — | 🗓 planned (post-MVP T1) |
| Llama-family dense | Llama 3.x, Mistral | — | — | 🗓 planned (post-MVP T1) |
| MoE decoders | Mixtral, Qwen3-MoE | — | — | 🗓 planned (post-MVP T1) |

## Acceleration

| Backend | Hardware | Status |
|---|---|---|
| CPU | x86-64 reference (correctness/CI grade) | 🟡 gate-model text engine + basic serving path end-to-end; multithreaded op dispatch (ggml-threadpool port, `VLLM_CPP_CPU_THREADS`) is 1/3/20-thread bit-identical and TSAN-clean. Its B4 real-file speed/RSS gate is pending an idle-host rerun; compute-in-quant GGUF speed remains open |
| CUDA | NVIDIA (first target: GB10 / DGX Spark, sm_121a) | 🟡 **gate-model paged text stack running on GB10**: vendored torch-free kernels (cutlass NVFP4/FP8, cuBLASLt, FA-2, Triton-AOT GDN, Qwen-specific CUDA-graph decode); both 27B + 35B greedy correctness gates pass. Exact `bce2627` 27B online throughput is 0.932-1.007× vLLM over c2-c32 medians (c1 0.968×), but the all-axis gate remains red at 4/4/5/4/4/12 of 20 axes and 2/4 memory axes. The fixed HTTP pool remains healthy/neutral. Device-resident cache W0 (+2.1239%) and indexed GDN state-I/O W1 (+0.6246%) retain their separate component evidence. FP4 W1 retains its Release/TSan/capture/memcheck/model proof; its exact/legacy component series improves c8/c32 by 0.93%/0.72% but fails strict no-regression at c16 and in memory. The fresh trace confirms W2's absent narrow Stream-K/static tactics, which precede any 35B performance run. Both models are W0 compute-sanitizer access-clean and the indexed op is memcheck-clean; inherited process-lifetime pools still fail the zero-leak gate |
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
| Sampling (greedy/top-k/top-p/penalties) | vLLM V1 ordering subset; some token/logprob paths synchronize to host | ✅ | 🟡 bounded subset |
| RMSNorm / default RoPE / SwiGLU | fused elementwise | ✅ | ✅ |
| Scaled RoPE supplied-cache rotation | pinned vLLM typed cache/factory + YaRN/MRoPE/Llama 3/Phi-3 LongRoPE/dynamic-NTK construction and cache lookup/rotation | ✅ ref + fifteen oracle fixtures | 🚧 source present; compile/runtime gate open |

Only **GB10/sm_121a is built, traced and gated today**. Source-level fallbacks
suggest routes for other SMs, but none counts as support until its full build,
correctness, trace and performance block passes. Non-CUDA backends
(Metal/Vulkan/ROCm/XPU) are post-MVP and will port from the CPU reference.

## Quantization

| Format | Status |
|---|---|
| NVFP4 (W4A16 MoE / W4A4 dense, Blackwell) | ✅ **both running on GB10**: W4A16 MoE (35B, Marlin + fp4-resident) and W4A4 dense (27B, cutlass sm120a fp4×fp4 + fp8-W8A8 attn/GDN); token-exact greedy gates pass |
| GGUF materialization (F32, Q4_0, Q8_0, Q3_K/Q4_K/Q5_K/Q6_K) | 🟡 load-time bf16 materialization; synthetic layout tests plus real 35B APEX Q3/Q4/Q5/Q6/Q8 greedy parity vs same-file llama.cpp. CPU ops now use correctness-gated multithreaded dispatch, but its real-file speed/RSS gate and direct compute-in-quant path remain open; F16/BF16, Q2_K, IQ/TQ/Q1, MXFP4 and NVFP4 execution remain open. |
| FP8 | 🟡 the 35B ModelOpt static per-tensor W8A8 projection slice is native and gate-passing; generic FP8 modes/dispatch and FP8 KV remain planned |
| MXFP4 / MXFP8 | 🗓 planned, including MLX-native modes on Apple |

Legend: ✅ supported & tested · 🚧 in development · 🗓 planned.

## Status & caveats (honest)

- **Where we are (GB10 / DGX Spark, sm_121a):** both NVFP4 gate-model text paths
  run the full paged engine end-to-end, and their real-model **token-exact greedy
  correctness gates pass**. The CUDA kernel paths exercised by those gates are
  validated on real hardware.
- **Production-vLLM performance is below the exact gate.** The historical 35B
  1.0195× and
  27B 1.007× offline ratios were produced with vLLM's
  `bench throughput` default `temperature=1.0`, while vllm.cpp used greedy
  `temperature=0`; the 27B vLLM arm also resolved an 8192-token scheduler budget
  while vllm.cpp used 2048. Those measurements and their memory observations
  remain useful diagnostics, but they are not identical-workload acceptance
  evidence. The replacement pushed-`bce2627` 27B online series is exact and
  binding: all 12 groups validate, but no concurrency passes all 20 performance
  axes. Median total throughput is 0.968/0.932/0.940/0.952/0.994/1.007×
  vLLM from c1 through c32; performance-axis counts are 4/4/5/4/4/12 of 20,
  and host PSS/RSS remain above vLLM. All three fixed
  c32 repetitions complete without the former unread-socket tail. The separate
  same-binary c32 fixed/legacy series is neutral within noise (0.999764× total
  throughput, 0.541%/0.311% CV) and its bounded legacy arms stay healthy, so it
  establishes capacity/lifecycle behavior rather than a steady-state speed win.
  W1's exact/legacy component series proves independent M=1/2/4/8/16 plans and
  improves c8/c32 total throughput by 0.93%/0.72%, but c16 is 0.9996× and only
  1/4 memory axes pass, so its strict component gate fails. The fresh vLLM
  execution trace attributes 25.1% of profiled kernel time to the absent
  128x32x256 Stream-K/static-persistent pair. Full-tactic W2 is now the active
  repair. This is a concrete runtime target, not a hardware ceiling. The exact
  35B run waits behind complete 27B closure.
- The optimized paths themselves remain implemented: vendored Triton-AOT GDN,
  cuBLASLt TN projection layouts, fused attention preamble, tiled causal-conv,
  vendored FlashAttention-2 prefill, register-tiled GDN `delta_h`, tensor-core WY,
  fused fp8 RMSNorm→quant, bf16 GDN I/O, and default-on fp4 autotuning. The
  current FP4 implementation is not yet fully FlashInfer-equivalent: W1 now
  gives M=1/2/4/8/16 distinct hybrid buckets, a complete plan key, single-flight
  tuning and capture-miss rejection, while retaining the aliased cache only
  under `VT_FP4_EXACT_BUCKETS=0`. It still exposes only four wide persistent
  tactics, while the traced vLLM dependency selects from a wider
  static-persistent/Stream-K family. The accepted
  [implementation spike](.agents/specs/nvfp4-small-m-dispatch.md) inventories
  the exact hybrid mapping, eight tiles × two operand orientations × two
  schedulers, upstream tests, workspace/capture rules and separate W1/W2 A/B
  gates. W1 passes focused Release stress, ThreadSanitizer and pushed-`c8807b0`
  exact/legacy CUDA capture, memcheck and 27B model gates. Its complete
  `bce2627` component/trace/oracle classification is positive but below the
  no-regression and every-axis floors; W2 owns the remaining 32-tactic port.
  Until gated, isolated per-shape wins do not establish end-to-end tactic parity.
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
- The CUDA low-concurrency comparison against SGLang is specified and pinned to
  v0.5.13. Its CPU-only P1 corpus/client/summary/process-memory/dry-run harness
  is implemented and contract-tested, but **no image, model compatibility, or
  SGLang performance result is claimed yet**. The unchanged 27B checkpoint has
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
  mismatches. Pushed `bce2627` is the current exact 27B cache-off, closed-loop
  checkpoint: 2,016/2,016 timed requests, three repetitions, six memory returns,
  the real-model gate, and paired traces pass their contracts. Median c1→c32
  total ratios are **0.968/0.932/0.940/0.952/0.994/1.007×**; only
  4/4/5/4/4/12 of 20 performance axes and 2/4 memory axes pass. All fixed c32
  legs are healthy. Exact-bucket/single-flight W1 is fully classified: its
  component series improves c8/c32 but fails strict c16/memory no-regression,
  and the fresh oracle trace confirms the absent 32-tactic W2 family. W0 is
  merged at `7d29e0c`; its clean GB10 build,
  default/fallback
  35B+27B pointer/token gates, lifecycle, trace, and repeated 27B A/B pass. The
  measured W0 same-binary gain is +2.12% (20/20 timing axes). W1 then wins a
  separate same-binary A/B by +0.6246% (20/20) and collapses the traced copy
  loop from 163,540 to 7,508 async calls. Neither component result is a fresh
  vLLM denominator or may be multiplied into the old grid. Both model paths are
  W0 memory-access clean and the W1 indexed op is memcheck-clean, but the
  required zero-leak result remains open on inherited process-lifetime pools.
  Fixed HTTP capacity, exact FP4 bucket/tactic parity, and fresh every-axis 27B
  closure remain mandatory before 35B and later roadmap work. W2 direct indexed
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
