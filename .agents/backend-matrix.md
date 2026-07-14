# Backend and architecture matrix

This is the canonical platform, CUDA-target, and native-competitor inventory for
roadmap D1. The accepted spike is
[specs/cuda-architecture-inventory.md](specs/cuda-architecture-inventory.md).
Upstream references are pinned to vLLM `e24d1b24`; local references describe
`vllm.cpp` at the inventory baseline `f7ccaa7`.

Build availability is not runtime support. A target remains `INVENTORIED` until
it has a native build, ported tests, hardware execution, and a trace showing the
intended dispatch. Performance claims additionally require same-workload runs
under [benchmark-protocol.md](benchmark-protocol.md).

## CUDA compiler branches

The global target list is selected by the CUDA compiler version at
`/home/mudler/_git/vllm/CMakeLists.txt:105-118`.

| Compiler branch | Global numeric targets | Count |
|---|---|---:|
| NVCC `< 12.8` | `7.0, 7.5, 8.0, 8.6, 8.7, 8.9, 9.0` | 7 |
| NVCC `>= 12.8, < 13.0` | `7.5, 8.0, 8.6, 8.7, 8.9, 9.0, 10.0, 10.1, 10.3, 12.0, 12.1` | 11 |
| NVCC `>= 13.0` | `7.5, 8.0, 8.6, 8.7, 8.9, 9.0, 10.0, 11.0, 12.0` | 9 |

The union is exactly 13 numeric targets. vLLM extracts PyTorch gencodes and
removes inherited `+PTX` (`CMakeLists.txt:201-220`), so a component must request
PTX explicitly. `cmake/utils.cmake:294-345` emits both SASS and PTX for
`+PTX`; `:348-485` implements loose target intersection. An `f` target matches
the same major family, an `a` target preserves architecture-specific features,
SASS fallback stays inside the major family, and PTX may cross major families.
The SM90+ helper is `cmake/utils.cmake:488-495`.

## CUDA target rows

The compiler column names global-list membership. `family via f` means a
component can still select that minor through vLLM's same-major `f` matching;
it does not mean the global list contains that numeric target.

| ID | Item | Compiler branches | Upstream | Our code | Tests/evidence | Spike/spec | State | Owner |
|---|---|---|---|---|---|---|---|---|
| `BACKEND-CUDA-SM070` | CUDA compute 7.0 | `<12.8` | `CMakeLists.txt:105-118` | configurable target only: [CMakeLists.txt:37](../CMakeLists.txt#L37), [CMakeLists.txt:64](../CMakeLists.txt#L64) | - | [CUDA inventory](specs/cuda-architecture-inventory.md) | `INVENTORIED` | - |
| `BACKEND-CUDA-SM075` | CUDA compute 7.5 | all three | `CMakeLists.txt:105-118`; Marlin Turing at `:548-679` | configurable target [CMakeLists.txt:37](../CMakeLists.txt#L37), applied globally at [CMakeLists.txt:64](../CMakeLists.txt#L64) | - | [CUDA inventory](specs/cuda-architecture-inventory.md) | `INVENTORIED` | - |
| `BACKEND-CUDA-SM080` | CUDA compute 8.0 | all three | `CMakeLists.txt:105-118`; Marlin `80+PTX` at `:548-679` | configurable target [CMakeLists.txt:37](../CMakeLists.txt#L37), applied globally at [CMakeLists.txt:64](../CMakeLists.txt#L64) | - | [CUDA inventory](specs/cuda-architecture-inventory.md) | `INVENTORIED` | - |
| `BACKEND-CUDA-SM086` | CUDA compute 8.6 | all three | `CMakeLists.txt:105-118`; AllSpark at `:721-731` | configurable target [CMakeLists.txt:37](../CMakeLists.txt#L37), applied globally at [CMakeLists.txt:64](../CMakeLists.txt#L64) | - | [CUDA inventory](specs/cuda-architecture-inventory.md) | `INVENTORIED` | - |
| `BACKEND-CUDA-SM087` | CUDA compute 8.7 | all three | `CMakeLists.txt:105-118`; scaled-mm C2x at `:839-863` | configurable target [CMakeLists.txt:37](../CMakeLists.txt#L37), applied globally at [CMakeLists.txt:64](../CMakeLists.txt#L64) | - | [CUDA inventory](specs/cuda-architecture-inventory.md) | `INVENTORIED` | - |
| `BACKEND-CUDA-SM089` | CUDA compute 8.9 | all three | `CMakeLists.txt:105-118`; FP8 Marlin at `:548-679` | configurable target [CMakeLists.txt:37](../CMakeLists.txt#L37), applied globally at [CMakeLists.txt:64](../CMakeLists.txt#L64) | - | [CUDA inventory](specs/cuda-architecture-inventory.md) | `INVENTORIED` | - |
| `BACKEND-CUDA-SM090` | CUDA compute 9.0 | all three | `CMakeLists.txt:105-118`; Hopper `90a` families at `:468-529,737-938` | configurable target [CMakeLists.txt:37](../CMakeLists.txt#L37), applied globally at [CMakeLists.txt:64](../CMakeLists.txt#L64) | - | [CUDA inventory](specs/cuda-architecture-inventory.md) | `INVENTORIED` | - |
| `BACKEND-CUDA-SM100` | CUDA compute 10.0 | `>=12.8` | `CMakeLists.txt:105-118`; SM10 C3x/MLA at `:806-837,1037-1061` | configurable target [CMakeLists.txt:37](../CMakeLists.txt#L37), applied globally at [CMakeLists.txt:64](../CMakeLists.txt#L64) | - | [CUDA inventory](specs/cuda-architecture-inventory.md) | `INVENTORIED` | - |
| `BACKEND-CUDA-SM101` | CUDA compute 10.1 | `>=12.8,<13` globally; family via `f` on 13 | `CMakeLists.txt:105-118`; loose intersection `cmake/utils.cmake:393-481` | configurable target [CMakeLists.txt:37](../CMakeLists.txt#L37), applied globally at [CMakeLists.txt:64](../CMakeLists.txt#L64) | - | [CUDA inventory](specs/cuda-architecture-inventory.md) | `INVENTORIED` | - |
| `BACKEND-CUDA-SM103` | CUDA compute 10.3 | `>=12.8,<13` globally; family via `f` on 13 | `CMakeLists.txt:105-118`; loose intersection `cmake/utils.cmake:393-481` | configurable target [CMakeLists.txt:37](../CMakeLists.txt#L37), applied globally at [CMakeLists.txt:64](../CMakeLists.txt#L64) | - | [CUDA inventory](specs/cuda-architecture-inventory.md) | `INVENTORIED` | - |
| `BACKEND-CUDA-SM110` | CUDA compute 11.0 | `>=13` | `CMakeLists.txt:105-118`; SM10/11 family rules at `:806-837,892-914` | configurable target [CMakeLists.txt:37](../CMakeLists.txt#L37), applied globally at [CMakeLists.txt:64](../CMakeLists.txt#L64) | - | [CUDA inventory](specs/cuda-architecture-inventory.md) | `INVENTORIED` | - |
| `BACKEND-CUDA-SM120` | CUDA compute 12.0 | `>=12.8` | `CMakeLists.txt:105-118`; SM12 FP4 at `:940-970` | SM12 fast paths admitted by [CMakeLists.txt:68](../CMakeLists.txt#L68), but unexecuted on SM120 | - | [CUDA inventory](specs/cuda-architecture-inventory.md) | `INVENTORIED` | - |
| `BACKEND-CUDA-SM121` | CUDA compute 12.1 / current `121a` gate | `>=12.8,<13` globally; family via `f` on 13 | `CMakeLists.txt:105-118`; SM12 FP4 at `:940-970` | default target [CMakeLists.txt:37](../CMakeLists.txt#L37); runtime [cuda_backend.cu:20](../src/vt/cuda/cuda_backend.cu#L20); SM12 fast paths [CMakeLists.txt:68](../CMakeLists.txt#L68) | [CUDA backend tests](../tests/vt/test_cuda_backend.cpp#L31); [27B gate](../tests/parity/test_qwen27_paged_engine.cpp#L110); [35B gate](../tests/parity/test_qwen36_paged_engine.cpp#L78); [ledger trace](parity-ledger.md#L284) | [CUDA inventory](specs/cuda-architecture-inventory.md) | `PARTIAL` - gate workload only, not full family coverage | - |

## CUDA component target rules

These are build and dispatch inventory rows, not claims that the component ran
on every listed target.

| ID | Item | Upstream target/capability rule | Our code | Tests/evidence | Spike/spec | State | Owner |
|---|---|---|---|---|---|---|---|
| `BACKEND-CUDA-COMP-CORE` | Stable core ops: activation, quant, RoPE, norm, sampler, SSM, cache, collectives | all filtered targets; `CMakeLists.txt:375-406,536-538` | CUDA source set [CMakeLists.txt:220](../CMakeLists.txt#L220) | [CUDA op tests](../tests/vt/test_cuda_ops.cpp#L206) on SM121 only | [CUDA inventory](specs/cuda-architecture-inventory.md) | `PARTIAL` | - |
| `BACKEND-CUDA-COMP-COOP-TOPK` | Cooperative top-k | CUDA >=12; `90a`; CUDA 13 uses SM10/11/12 `f` families; `:408-424,540-546` | - | - | [CUDA inventory](specs/cuda-architecture-inventory.md) | `INVENTORIED` | - |
| `BACKEND-CUDA-COMP-MARLIN` | Marlin dense and MoE | FP16 `80+PTX`, Turing `75`; BF16 adds `90+PTX`; FP8 input `89`; SM12 `a/f`; `:548-679,1168-1274` | NVFP4 SM12 subset [CMakeLists.txt:101](../CMakeLists.txt#L101) | [grouped-MoE tests](../tests/vt/test_ops_moe_grouped.cpp#L160) | [CUDA inventory](specs/cuda-architecture-inventory.md) | `PARTIAL` | - |
| `BACKEND-CUDA-COMP-MACHETE` | Machete low-bit GEMM | CUDA >=12, `90a`; `:468-529` | - | - | [CUDA inventory](specs/cuda-architecture-inventory.md) | `INVENTORIED` | - |
| `BACKEND-CUDA-COMP-DSV3` | DeepSeek V3 fused A GEMM and router | SM90+ family, CUDA >=12; `:687-719,1282-1299` | - | - | [CUDA inventory](specs/cuda-architecture-inventory.md) | `INVENTORIED` | - |
| `BACKEND-CUDA-COMP-ALLSPARK` | AllSpark GEMM | `80,86,87,89`; `:721-731` | - | - | [CUDA inventory](specs/cuda-architecture-inventory.md) | `INVENTORIED` | - |
| `BACKEND-CUDA-COMP-SCALEDMM-C3X` | CUTLASS scaled-mm C3x | `90a`; SM10/11 and SM12 `a/f`; CUDA >=12/12.8; `:737-837` | FP8 and NVFP4 SM121 subsets [CMakeLists.txt:236](../CMakeLists.txt#L236) | [FP8](../tests/vt/test_ops_fp8_cutlass.cpp#L188), [NVFP4](../tests/vt/test_ops_nvfp4_fp4.cpp#L48) | [CUDA inventory](specs/cuda-architecture-inventory.md) | `PARTIAL` | - |
| `BACKEND-CUDA-COMP-SCALEDMM-C2X` | CUTLASS scaled-mm C2x | `75,80,87,89+PTX`, excluding C3x; `:839-863` | - | - | [CUDA inventory](specs/cuda-architecture-inventory.md) | `INVENTORIED` | - |
| `BACKEND-CUDA-COMP-MOE-CUTLASS` | CUTLASS MoE | Hopper `90a` >=12.3; SM10/11 >=12.8; common data on SM90+; `:865-938` | - | - | [CUDA inventory](specs/cuda-architecture-inventory.md) | `INVENTORIED` | - |
| `BACKEND-CUDA-COMP-FP4` | NVFP4/MXFP4 dense, MoE, activation, KV | SM10/11/12, CUDA >=12.8; MXFP4 stubs before 12.9; `:940-1002` | NVFP4 SM121 subset [cuda_matmul_nvfp4.cu:934](../src/vt/cuda/cuda_matmul_nvfp4.cu#L934) | [NVFP4 units](../tests/vt/test_ops_nvfp4_fp4.cpp#L48), [27B gate](../tests/parity/test_qwen27_paged_engine.cpp#L110) | [CUDA inventory](specs/cuda-architecture-inventory.md) | `PARTIAL` | - |
| `BACKEND-CUDA-COMP-W4A8` | CUTLASS W4A8 | CUDA >=12, `90a`; `:1004-1035` | - | - | [CUDA inventory](specs/cuda-architecture-inventory.md) | `INVENTORIED` | - |
| `BACKEND-CUDA-COMP-MLA` | CUTLASS MLA | SM10/11, CUDA >=12.8; `:1037-1061` | - | - | [CUDA inventory](specs/cuda-architecture-inventory.md) | `INVENTORIED` | - |
| `BACKEND-CUDA-COMP-HADACORE` | Hadacore | `80+PTX,90+PTX`; `:1063-1072` | - | - | [CUDA inventory](specs/cuda-architecture-inventory.md) | `INVENTORIED` | - |
| `BACKEND-CUDA-COMP-FLASHMLA` | FlashMLA | `90a` >=12.3; SM100 `a/f` >=12.8/12.9; `cmake/external_projects/flashmla.cmake:51-142` | - | - | [CUDA inventory](specs/cuda-architecture-inventory.md) | `INVENTORIED` | - |
| `BACKEND-CUDA-COMP-DEEPGEMM` | DeepGEMM | `90a`, SM100 `a/f`, SM12 `a/f`; runtime JIT; `deepgemm.cmake:61-105` | - | - | [CUDA inventory](specs/cuda-architecture-inventory.md) | `INVENTORIED` | - |
| `BACKEND-CUDA-COMP-QUTLASS` | QuTLASS | SM100 or SM120 family, CUDA >=12.8; one family selector/build; `qutlass.cmake:56-147` | - | - | [CUDA inventory](specs/cuda-architecture-inventory.md) | `INVENTORIED` | - |
| `BACKEND-CUDA-COMP-FA` | FlashAttention | FA2 all filtered targets; FA3 >=12.3; FA4 CuTe package; `vllm_flash_attn.cmake:1-46`, `fa_utils.py:132-250` | FA2 BF16 causal head-256 prefill plus bounded ratio-6 non-causal split-KV decode [CMakeLists.txt:428](../CMakeLists.txt#L428), [adapter](../src/vt/cuda/cuda_flash_attn_fa2.cu#L429) | [paged-attention tests](../tests/vt/test_ops_paged_attn.cpp#L585), [27B gate](../tests/parity/test_qwen27_paged_engine.cpp#L110); immutable `ae9e8ff` clean sm_121a compile/operator/memcheck/model/paired-trace pass. Completed c2/c16 component is **1.017668×/1.006548×** mean total throughput but strict-fails **35/40 timing + 5/8 memory**; no speed credit | [CUDA inventory](specs/cuda-architecture-inventory.md); [W3-G spike](specs/fa2-gqa-split-kv-decode.md) | `PARTIAL` | - |
| `BACKEND-CUDA-COMP-JIT` | FlashInfer, CuTe DSL, Triton kernels, fmha_sm100 | package/JIT dispatch; `requirements/cuda.txt:12-25`, `cmake/external_projects/*.cmake` | vendored SM121 GDN AOT [TritonAOT.cmake:57](../cmake/TritonAOT.cmake#L57) | [GDN AOT tests](../tests/vt/test_ops_gdn.cpp#L1266) and GB10 trace only | [CUDA inventory](specs/cuda-architecture-inventory.md) | `PARTIAL` | `CLAIM-PR3` for its listed GDN leaves only |

## Other platform rows

| ID | Item | Upstream | Our code | Tests/evidence | Spike/spec | State | Owner |
|---|---|---|---|---|---|---|---|
| `BACKEND-ABI-VT` | Backend registry, device/queue, capture, drop-in adapter ABI | platform contract `vllm/platforms/interface.py:67-229`; stable stream `csrc/libtorch_stable/torch_utils.h:76-82`; raw Marlin boundary `csrc/libtorch_stable/moe/marlin_moe_wna16/ops.cu:344-356` | queue ID [device.h:9](../include/vt/device.h#L9), explicit resources [backend.h:53](../include/vt/backend.h#L53), scalar/layout ABI [ops.h:13](../include/vt/ops.h#L13), CUDA workspace/raw probe [cuda_dropin.cu:104](../src/vt/cuda/cuda_dropin.cu#L104) | [ported ABI tests](../tests/vt/test_dropin_abi.cpp#L86); CPU 94/94 + repair-focused 1/1; GCC13 repair at [test_dropin_abi.cpp:245](../tests/vt/test_dropin_abi.cpp#L245); exact sm_121a all-target build 100%, focused CUDA/ABI CTest 2/2, sanitizer 9/9 + 196/196 with 0 errors/leaks, 35B/27B gates 2/2 (`1141b79`, evidence manifest `4adbe952…601`) | [drop-in ABI](specs/dropin-kernel-abi.md) | `ACTIVE` - sm_80/sm_90a cross-build and unchanged-trace/model A/B-memory proof remain; common scalar forwarder and legacy backend-shim migration remain; no production family migrated | CLAIM-BACKEND-ABI-W0-GPU-1 |
| `BACKEND-CPU` | CPU correctness and production path | `platforms/cpu.py:42-125`, CPU ops rooted at `csrc/cpu/torch_bindings.cpp:123-139`; llama.cpp `ggml-cpu.c:471-610,3024-3390` owns the added native pool reference | [cpu_backend.cpp:11](../src/vt/cpu/cpu_backend.cpp#L11), [threadpool.cpp:78](../src/vt/cpu/cpu_threadpool.cpp#L78), [chunked ops](../src/vt/cpu/cpu_ops.cpp#L88) | [backend tests](../tests/vt/test_backend.cpp#L10), [op parity](../tests/parity/test_op_parity.cpp#L34), [threadpool/upstream-test port](../tests/vt/test_cpu_threadpool.cpp#L63); 1/3/20 full CPU suites + TSAN pass, but B4 speed/RSS and full model-context thread-safety gates remain open | [threadpool leaf](specs/gguf-cpu-threadpool.md) | `PARTIAL` | - |
| `BACKEND-CPU-ZEN` | AMD Zen CPU with ZenDNN/zentorch dispatch and weight prepack | `platforms/zen_cpu.py:12-32`; detection `platforms/__init__.py:153-192`; `tests/test_zen_cpu_platform_detection.py:8-37` | generic CPU backend only; no Zen-specific dispatch | - | [CUDA inventory](specs/cuda-architecture-inventory.md); leaf spike required | `INVENTORIED` | - |
| `BACKEND-ROCM` | AMD ROCm/HIP | `platforms/__init__.py:110-128`, `platforms/rocm.py:43-125`, ROCm ops rooted at `csrc/rocm/moe_q_gemm_rdna3.cu:1` | - | - | [CUDA inventory](specs/cuda-architecture-inventory.md) | `INVENTORIED` | - |
| `BACKEND-XPU` | Intel XPU loyal port; gating exploration E4 ([backends.md](backends.md) kernel sourcing: SYCL-vs-Level-Zero call + whether upstream XPU attention contracts translate 1:1; explore when T2 scheduling begins) | `platforms/__init__.py:131-150`, `platforms/xpu.py:103-125` | enum slot only [device.h:11](../include/vt/device.h#L11) | [unavailable-backend test](../tests/vt/test_backend.cpp#L37) | [CUDA inventory](specs/cuda-architecture-inventory.md) | `INVENTORIED` | - |
| `BACKEND-TPU` | vLLM TPU parity surface | `platforms/__init__.py:35-56,202-208`, `platforms/tpu.py:9-20` | - | - | [CUDA inventory](specs/cuda-architecture-inventory.md) | `INVENTORIED` | - |
| `BACKEND-METAL-MLX` | Apple Metal through MLX; native MSL fallback | vllm.cpp extension through upstream seam `platforms/interface.py:134-229` | enum slot only [device.h:11](../include/vt/device.h#L11) | - | [CUDA inventory](specs/cuda-architecture-inventory.md) | `INVENTORIED` | - |
| `BACKEND-VULKAN` | Vulkan compute backend | vllm.cpp extension through upstream seam `platforms/interface.py:134-229`; llama.cpp is maturity reference | enum slot only [device.h:11](../include/vt/device.h#L11) | [unavailable-backend test](../tests/vt/test_backend.cpp#L37) | [CUDA inventory](specs/cuda-architecture-inventory.md) | `INVENTORIED` | - |
| `BACKEND-ANE` | Apple Neural Engine for encoder/pooling/fixed-shape draft classes | vllm.cpp extension through upstream seam `platforms/interface.py:134-229`; not a paged decode backend | - | - | [CUDA inventory](specs/cuda-architecture-inventory.md) | `INVENTORIED` | - |

## Native competitor and performance gates

Every run records the competitor commit/release, model artifact hash, build
flags, exact command, warmup, repetitions, concurrency, output tokens, and peak
memory. Floating competitor versions do not count.

| ID | Item | Upstream/reference | Our code | Tests/evidence | Spike/spec | State | Owner |
|---|---|---|---|---|---|---|---|
| `BACKEND-GATE-CUDA-VLLM` | CUDA correctness and every-axis parity vs vLLM v0.25.0 | current pin `e24d1b24`; oracle `702f481`; async default `vllm/config/vllm.py:952-1043`; depth-2 engine path `vllm/v1/engine/core.py:519-607`; async output `vllm/v1/worker/gpu/async_utils.py:12-70`; [benchmark protocol](benchmark-protocol.md) | Existing schema-v5 validator/summary/driver remain; diagnostic oracle mode is [profile_vllm_online_gate.py](../tools/bench/profile_vllm_online_gate.py#L21) and derived evidence is fail-closed by [finalize_async_credit.py](../tools/bench/finalize_async_credit.py#L1), with no local engine-default change | [mode/finalizer contracts](../tests/tools/test_online_gate_trace.py#L30) and [async summary cases](../tests/tools/test_async_credit_summary.py#L1) are green. Binding `3f256ab` stays **55/124**. Accepted async `3812d8` is neutral for speed: ON/OFF total **1.002153×**, TPOT **1.010291×**, TTFT **0.862159×**, traced GPU time **1.002004×**; W3 stays `READY`/uncredited and speed work returns to low-batch kernels. Host PSS/RSS remains **22.920 GiB** CPU weights plus mmap pages; W3-I off and 35B blocked. See [async W3](specs/async-serving.md) and [scoreboard](../docs/BENCHMARKS.md) | [CUDA inventory](specs/cuda-architecture-inventory.md); [async W3](specs/async-serving.md); [W3-I](specs/nvfp4-fused-silu-producer.md) | `PARTIAL` | - |
| `BACKEND-BENCH-CUDA-SGLANG-PREFLIGHT` | SGLang corpus, harness, image and exact-checkpoint/token-ID preflights; P1 CPU harness is implemented/gated while P2 image/model/GPU classification remains | [SGLang v0.5.13](https://github.com/sgl-project/sglang/releases/tag/v0.5.13), commit `28b095c`; digest-pinned CUDA 13 image | [corpus](../tools/bench/make_serve_low_corpus.py#L143); [client/preflights](../tools/bench/run_serve_low.py#L124); [summary](../tools/bench/summarize_serve_low.py#L238); [memory](../tools/bench/sample_process_memory.py#L198); [driver](../scripts/dgx-sglang-low-concurrency.sh#L1) | [16 CPU contract cases](../tests/tools/test_serve_low_client.py#L103); [CTest registration](../tests/CMakeLists.txt#L12); dry-run manifest green; no image/model/GPU evidence or performance result | [SGLang low-concurrency spike](specs/cuda-sglang-low-concurrency.md) | `GATING` | - |
| `BACKEND-GATE-CUDA-SGLANG` | Binding CUDA low-concurrency serving vs SGLang | same pinned SGLang/checkpoints as `BACKEND-BENCH-CUDA-SGLANG-PREFLIGHT` | [bench main:1](../examples/bench/main.cpp#L1), [server main:1](../examples/server/main.cpp#L1) | no binding run; HTTP TTFT/ITL cannot be measured honestly yet | [SGLang low-concurrency spike](specs/cuda-sglang-low-concurrency.md) | `BLOCKED` on `SERVE-ASYNC-LLM` and successful exact-equivalence preflight | - |
| `BACKEND-GATE-CUDA-SGLANG-PREFIX` | Binding deterministic shared-prefix cache-on serving vs the faster equivalent vLLM/SGLang floor, separate from cache-neutral serving | SGLang v0.5.15 `f63458b`; digest `d0a667e`; [DGX recipe correction/results](https://github.com/Weschera/qwen-sglang-dgx-spark/tree/03253ef98c01de59a21c85b9a5cc6a27a871c383); vLLM v0.25.0 explicit Qwen hybrid `mamba_cache_mode=align` | current prefix hashes/managers [kv_cache_utils.cpp:259](../src/vllm/v1/core/kv_cache_utils.cpp#L259), [kv_cache_manager.cpp:124](../src/vllm/v1/core/kv_cache_manager.cpp#L124); no local GDN/Mamba aligned-state retention or binding long-prefix harness yet | Source/config audit rejects the original cache-asymmetric 10--40x claim and treats the reported residual 25--45% SGLang lead as unproven: cache-on vLLM is 0.23.1, KV dtypes/memory differ, MTP is enabled, only 35B is measured, and full axes/hits/memory/traces/repetitions are absent. PX1/PX2 implementation, exact 27B equivalence/hit proof and all performance evidence remain pending | [SGLang shared-prefix extension](specs/cuda-sglang-low-concurrency.md#shared-prefix-extension-2026-07-12) | `READY` for PX1 harness/counter work; PX2 begins with the `KV-MAMBA-ALIGN` leaf spike, and binding execution also needs exact v0.5.15 equivalence plus `SERVE-ASYNC-LLM` | - |
| `BACKEND-GATE-ROCM-VLLM` | ROCm parity vs vLLM | pinned vLLM ROCm backend | - | - | [CUDA inventory](specs/cuda-architecture-inventory.md) | `INVENTORIED` | - |
| `BACKEND-GATE-ROCM-SGLANG` | ROCm low-concurrency serving vs SGLang | pinned SGLang ROCm build | - | - | [competitive benchmark spike](specs/competitive-benchmarks.md) | `INVENTORIED` | - |
| `BACKEND-GATE-XPU-VLLM` | Intel XPU parity vs vLLM | pinned vLLM XPU backend | - | - | [CUDA inventory](specs/cuda-architecture-inventory.md) | `INVENTORIED` | - |
| `BACKEND-GATE-CPU-LLAMACPP` | CPU throughput/latency/memory vs llama.cpp | [llama.cpp](https://github.com/ggml-org/llama.cpp), commit pinned per run | [CPU backend:11](../src/vt/cpu/cpu_backend.cpp#L11), [bench main:1](../examples/bench/main.cpp#L1) | [CPU correctness](../tests/parity/test_op_parity.cpp#L34); first speed comparison measured (Qwen3.5-2B Q8 GGUF, same file/box): llama.cpp ahead 54–75× decode / ≈1,480× prefill / 2.7× peak RSS ([ledger B4 row](parity-ledger.md#L290)) — gate far open until `QUANT-GGUF-COMPUTE` lands | [competitive benchmark spike](specs/competitive-benchmarks.md) | `INVENTORIED` | - |
| `BACKEND-GATE-METAL-OMLX` | MLX/Metal serving vs oMLX | [oMLX v0.5.0rc1](https://github.com/jundot/omlx/releases/tag/v0.5.0rc1), then pinned successor | - | - | [competitive benchmark spike](specs/competitive-benchmarks.md) | `INVENTORIED` | - |
| `BACKEND-GATE-METAL-MLXLM` | MLX/Metal correctness and generation baseline vs MLX-LM | [MLX-LM](https://github.com/ml-explore/mlx-lm), commit pinned per run | - | - | [competitive benchmark spike](specs/competitive-benchmarks.md) | `INVENTORIED` | - |
| `BACKEND-GATE-METAL-LLAMACPP` | Metal kernel/model parity vs llama.cpp Metal | pinned llama.cpp Metal build | - | - | [competitive benchmark spike](specs/competitive-benchmarks.md) | `INVENTORIED` | - |
| `BACKEND-GATE-VULKAN-LLAMACPP` | Vulkan kernel/model parity vs llama.cpp Vulkan | pinned llama.cpp Vulkan build | - | - | [competitive benchmark spike](specs/competitive-benchmarks.md) | `INVENTORIED` | - |

## Count invariants

- Compiler branch target counts are exactly `7`, `11`, and `9`; the union is
  exactly 13 numeric CUDA targets.
- The CUDA target table has exactly 13 stable rows.
- The component target table has exactly 18 stable rows.
- The non-CUDA/platform ABI table has exactly 9 stable rows.
- The native competitor table has exactly 12 stable rows: one independently
  claimable SGLang preflight and eleven binding platform/workload competitor
  gates, including distinct cache-neutral and shared-prefix CUDA rows.
- Adding or removing an upstream target, component family, platform, or
  required competitor changes the corresponding count in the same commit.
