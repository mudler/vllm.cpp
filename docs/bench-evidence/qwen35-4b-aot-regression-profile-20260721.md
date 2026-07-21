# Qwen3.5-4B Triton-AOT regression profile (2026-07-21)

Disposition: diagnostic attribution complete; no engine behavior changed and no
speed credit claimed. The immutable profile root is
`/tmp/qwen35-transplant-4b-aot-profile-832ff89d`.

## Workload and capture

All five project arms used the exact benchmark workload already bound by
`/tmp/qwen35-transplant-4b-aot-557ab41d`: Qwen3.5-4B plain BF16, ShareGPT corpus
`/tmp/qwen35-4b-sharegpt-1024.json` (SHA-256
`9ea13603767c62c267e3f381fbccf42d0c9ca0c393655c37533eadca7aefca0c`),
128 requests, 131,784 input tokens, 16,384 output tokens, concurrency 32,
greedy ignore-EOS, `max-num-batched-tokens=2048`, and 1,280 blocks. One
`flock /tmp/gpu` covered the full series on the idle RTX 5070 Ti.

Each arm was captured with:

```sh
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --cuda-graph-trace=node --stats=false -o "$REPORT" \
  env LD_LIBRARY_PATH=/run/opengl-driver/lib \
  VT_RELEASE_HOST_WEIGHTS=1 VT_DIRECT_DEVICE_LOAD=1 "$BENCH" \
  --model "$MODEL" --dataset-path /tmp/qwen35-4b-sharegpt-1024.json \
  --num-prompts 128 --output-len 128 --concurrency 32 --temperature 0 \
  --max-num-batched-tokens 2048 --num-blocks 1280 \
  --output-token-ids "$TOKENS"
```

The previous binary is
`build-nix-cuda-triton-sm120-debug/examples/vllm-bench` (SHA-256
`2ecb5f97b3add9043780382a0950acad63e2f8bebd797656cfecbd66eab660c0`,
source `829883d`, CMake CUDA arch 120, Triton AOT ON). The current binary is
`build-nix-cuda-transplant-triton/examples/vllm-bench` (SHA-256
`fc3d0e157b9d578e8479fbbc61d2f90656a307d8a52da1bd315aba0749c7d6f1`,
source `557ab41d`, CMake CUDA arch 120a, generated Triton target sm_120, Triton
AOT ON). The old-capped arm additionally sets `VT_POOL_MAX_CACHED_MB=1024` and
`VT_SAFETENSORS_DISCARD_PAGES=1`; old-uncapped omits both. Current-sync sets
`VT_ASYNC_RUNNER=0`; current-sync-packed-off also sets
`VT_GDN_PACKED_DECODE=0`. Current-uncapped uses shipping defaults.

No project arm recorded a CUDA graph launch, so the node-trace requirement did
not hide child kernels. The existing vLLM comparison trace does contain graph
node rows and was captured with the same node option.

## Profile results

These are one exact-workload run per arm under profiler overhead; the prior
three-repetition 6155.10-versus-6607.04 tok/s comparison remains the binding
regression magnitude.

| Arm | Total tok/s | Output tok/s | Mean / median TTFT ms | Mean TPOT ms | Mean E2EL ms | GPU kernel time s |
|---|---:|---:|---:|---:|---:|---:|
| old capped | 6560.73 | 725.47 | 667.84 / 220.15 | 38.96 | 5615.86 | 21.944263 |
| old uncapped | 6586.39 | 728.30 | 661.32 / 220.42 | 38.86 | 5596.05 | 21.960202 |
| current default async | 6079.71 | 672.28 | 730.48 / 426.03 | 41.97 | 6060.39 | 23.637218 |
| current sync | 6213.75 | 687.10 | 720.98 / 423.46 | 41.02 | 5930.20 | 23.306561 |
| current sync, packed off | 6266.92 | 692.98 | 721.37 / 423.57 | 40.62 | 5879.92 | 23.091041 |

The old residency controls are not the speed cause: removing the 1,024-MiB
pool cap changes old throughput by only +0.39%. It does explain the memory
denominator change: old capped reports 296,315 pool hits / 896 misses / 374
evictions and about 1,014 MiB cached, while old uncapped reports 296,679 hits /
532 misses / zero evictions and about 1,123 MiB cached.

The shipping async runner is a secondary cost on this workload: current sync
recovers +2.20% over current default under the same profiler, with 128/128
identical request outputs. It does not explain the whole regression: current
sync remains 5.66% below old uncapped.

The matched synchronous traces prove a GPU-side regression. Total GPU kernel
time increases from 21.960202 s to 23.306561 s (+1.346360 s). The dominant
executed-path change is the recurrence:

| Executed kernel | Total time | Calls | Mean call |
|---|---:|---:|---:|
| old sm120 Triton `gdn_decode_kernel` | 0.601555 s | 12,624 | 47.652 us |
| current hand `GdnPackedDecodeKernel<bf16,float,8>` | 2.579749 s | 11,016 | 234.182 us |

The raw recurrence difference is +1.978 s; newer current-main attention and
fusion wins offset part of it, leaving the observed +1.346 s net GPU increase.
In particular, the old custom prefill attention contributes 0.897571 s while
the current FA2 split-KV kernel contributes 0.377271 s. The current trace also
removes an old 0.740847-s BF16 projection and the old post-conv/gated-norm
kernels, while some common small GEMMs and the current prefill convolution are
slower. Those are secondary follow-ups, not yet isolated causes.

Disabling packed decode is neither valid nor sufficient. It swaps the current
2.579749-s packed kernel for 2.652254 s of `GdnDecodeFusedKernel` plus glue,
reduces whole-run GPU time by only 0.215521 s, improves throughput only 0.86%
over current sync, and matches the shipping output on only 117/128 requests.
The shipping async and sync arms match 128/128.

## Source and oracle attribution

Current main builds only the Hk=16/Hv=48 packed-decode AOT specialization for
the 27B shape (`cmake/TritonAOTKernels.cmake:76-86`, `CMakeLists.txt:485-515`).
`TryTritonPackedDecode` explicitly rejects any geometry other than
Hk=16/Hv=48/Dk=Dv=128 (`src/vt/cuda/cuda_gdn.cu:4316-4380`). Qwen3.5-4B has
the H=32 geometry, so its eligible pure-decode call
(`src/vllm/model_executor/models/qwen3_5.cpp:51-58,3213-3230`) falls through to
the shared-memory hand kernel (`src/vt/cuda/cuda_gdn.cu:1768-1920`).

The previous branch had a shape-specific sm120 H=32 Triton decode realization.
The current packed Triton shim already mirrors vLLM's generic packed recurrence
body (`triton_kernels/fused_recurrent_packed_decode.py:25-69,77-163`) and vLLM
launches that body with one warp and three stages
(`.venv-vllm/lib/python3.12/site-packages/vllm/model_executor/layers/fla/ops/fused_recurrent.py:255-336,432-477`).
The existing exact-workload vLLM trace proves the runtime selection:
`fused_recurrent_gated_delta_rule_packed_decode_kernel` takes 1.451278 s across
13,872 calls (104.619 us/call), versus 234.182 us/call for the project's hand
packed kernel.

Therefore the repair target is an H=32 sm120 AOT specialization of the current
packed ABI with f32 state, plus an exact-shape launcher guard and same-binary
token/performance gates. Blindly restoring the old decomposed cubin would be
incorrect because its ABI consumed precomputed q/k/v/g/beta and BF16 state;
current main consumes raw packed QKV/BA and f32 state, matching vLLM.

## Artifact integrity

| Arm | nsys report SHA-256 | kernel-summary SHA-256 | token SHA-256 |
|---|---|---|---|
| old capped | `7b03bec859f41063578de6847ffde01e56476993369fd9385d95cb285f536675` | `2159b7a1cc2e2b67296d5589a55aa72ec0166b0a91d7553a93a0efd85b641664` | `6b847bda84993c70789d0e1bdf61e2ea2ee4cbbf01339b9f2fe19722dfc0cbab` |
| old uncapped | `70b4583b8f2eb114cce65fc53d2d927434e73fa88e1a3ddb3b3fe0b6e0a1bc66` | `3c8608d7379ab87c598646014b6726f20c92c89a68732ac14e4eab0aaa19680c` | `7831504efc778ac4e41c2411fe31d81e4a1eab478af68af3e238e25506e56198` |
| current default async | `3a1b73e7137c1c88d071c842062a67a0faccef2af00ebf6c3f340c39bef8205d` | `724ba89f7da6f29ce781b5b7efcb34904276ffcdca03fb3235989365188104de` | `7a4390638682690f49e58381d6bd88e7b7a8346a6664c67b01582aa2df603f04` |
| current sync | `092021eeca0b2f27c2cd9a79a3f2bb4a1f2ff3e8daa96d54e7d202235b0255f3` | `8181f319c1b6a4657a5d70cc20203a809ab3f71f1b74f468d7c25ac0a62f9e40` | `7a4390638682690f49e58381d6bd88e7b7a8346a6664c67b01582aa2df603f04` |
| current sync, packed off | `1dd59e3ed12e7027f1d0efcaaa52598b6b2559596831792f06e7aaff73e0f681` | `dff6357f45e26ad4bc4980df6dc34244a716e59020bc46e9ac1d2eb0feb61507` | `d20ae17a9cd892f32eebb77e5f43b359c73617165d82ed46f6683b0aa925dfc5` |

The vLLM packed-kernel CSV is
`/tmp/qwen35-transplant-4b-aot-557ab41d/trace-vllm-kernels_cuda_gpu_kern_sum.csv`
(SHA-256 `21af1c6a59491486279c2cfe898b80ac27c0151274b1a7fb2e1c7006485fb971`).
