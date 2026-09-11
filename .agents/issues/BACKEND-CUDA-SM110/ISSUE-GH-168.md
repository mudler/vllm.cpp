ID: ISSUE-GH-168
Title: Jetson AGX Thor (sm_110), CUDA 13.2: 32B NVFP4 serves, first throughput datapoints, and a Tekken tokenizer blocker
Row: BACKEND-CUDA-SM110
State: CLOSED
Kind: feature
GitHub: 168
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-08
Updated: 2026-08-11
Closed: 2026-08-11

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Hi — first, thanks for the unusually honest capability labelling in this project. `BACKEND-CUDA-SM110` being scoped to exactly "portable bf16 Llama-1B greedy" is what made it obvious where a useful contribution was, so this report tries to fill that gap rather than restate it.
>
> I have a **Jetson AGX Thor** and ran a window on it. Summary up front:
>
> | # | Result |
> |---|---|
> | 1 | **Qwen3-32B-NVFP4A16 loads and serves on sm_110** via the portable dequant-GEMM |
> | 2 | **First serving-throughput datapoints on sm_110** — and they are *flat* with concurrency |
> | 3 | **CUDA 13.2 builds and passes the CUDA gates** on sm_110 (you verified with 13.0) |
> | 4 | **Blocker: Mistral-Nemo (Tekken) is rejected at the tokenizer** |
> | 5 | The default async runner does **not** fault at op level here (unlike sm_87) |
>
> Everything below is from one session; I have kept the caveats attached to each claim rather than in a footnote.
>
> ## Environment
>
> - NVIDIA Jetson AGX Thor, `aarch64`, `nvidia-smi compute_cap = 11.0`, 122 GB unified memory
> - Built from `0e3bf3c`
> - **Built and run inside `nvcr.io/nvidia/vllm:26.04-py3`** — the Jetson host carries only the driver (no `/usr/local/cuda*`, no `nvcc`), so a native host build was not possible without installing a toolkit. The container supplies `nvcc cuda_13.2.r13.2`, cmake 3.31.6, ninja 1.13, g++ 13.3.0.
> - Container run standalone: `--runtime=nvidia --network=host`, **no** MPS pipe mount, **no** `--ipc=host`.
>
> ```
> cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
>   -DVLLM_CPP_CUDA=ON -DVLLM_CPP_TRITON=OFF -DVLLM_CPP_CUDA_ARCHITECTURES=110
> cmake --build build -j 8
> ```
>
> `1124/1124` objects, **0 errors, 0 warnings**. Codegen verified as
> `--generate-code=arch=compute_110,code=[compute_110,sm_110]` and `strings` on
> `cuda_backend.cu.o` contains `sm_110`.
>
> As expected, every accelerated path resolved empty for `[110]`: `cutlass-nvfp4`,
> `cutlass-nvfp4-sm100`, `cutlass-fp8`, `scaledmm-c3x-sm90`, `scaledmm-c3x-sm100`,
> `marlin-nvfp4`, `fa2`.
>
> ## 1. Qwen3-32B-NVFP4A16 serves on sm_110
>
> `RedHatAI/Qwen3-32B-NVFP4A16` (`Qwen3ForCausalLM`, 64 layers, `compressed-tensors` /
> `nvfp4-pack-quantized`) loads and produces coherent output:
>
> ```
> --model <local> --port 30803 --gpu-memory-utilization 0.35 \
> --max-model-len 4096 --max-num-seqs 16      # VT_ASYNC_RUNNER=0
> ```
>
> Two things this establishes on sm_110, if I am reading `BACKEND-CUDA-SM110`,
> `docs/BUILD.md` and `docs/BENCHMARKS.md` correctly — please correct me if not:
>
> - **NVFP4 computes on sm_110** through the portable `kMatmulNvfp4` dequant-GEMM. I had
>   initially assumed NVFP4 was unavailable here because the CUTLASS/Marlin/fp4-MMA fast
>   paths are all EMPTY for `110`; that turned out to be wrong, and the README's
>   `Qwen3-32B-NVFP4A16` line is what pointed the right way.
> - **A >1B model runs on sm_110** — 32B rather than the 1B the matrix row is scoped to.
>
> ## 2. First throughput datapoints on sm_110 — and they do not batch
>
> 64 max tokens, `temperature=0`, N concurrent chat-completions:
>
> | concurrency | wall | tokens | throughput |
> |---|---|---|---|
> | 1 | 45.9 s | 64 | **1.39 tok/s** |
> | 2 | 88.3 s | 128 | **1.45 tok/s** |
> | 4 | 142.3 s | 256 | **1.80 tok/s** |
>
> The *shape* is the interesting part: **c=4 buys only 1.3x over c=1**, where your published
> GB10 curve scales 86 -> 292 tok/s (~3.4x) over the same range. On this arch the portable
> path appears bound by the unaccelerated GEMM rather than by parallelism — i.e. the fast
> paths look like they buy *batching*, not just peak rate. That may be the more useful half
> of the observation.
>
> **Caveats, and they are not small:**
> - Crude harness (concurrent `curl`), **one run per point**, no warm-up excluded, wall time
>   includes HTTP and tokenization.
> - Qwen3 emits thinking tokens, so this is total generated tokens, not useful-answer tokens.
> - **Other GPU tenants were live on the box** (two unrelated vLLM containers under MPS,
>   ~19 GB). This was not an isolated device, so treat these as a **lower bound**.
> - `0.35` memory fence, `max-model-len 4096`, `max-num-seqs 16` — untuned.
> - Not comparable to the GB10 numbers: different arch, different quant path, different model.
>
> Happy to re-run properly isolated, with a warm-up and repeats, if that would be more useful
> than an indicative first datapoint.
>
> ## 3. CUDA 13.2
>
> `test_cuda_backend` and `test_cuda_ops` both **pass** on sm_110 built with **CUDA 13.2**,
> under both `VT_ASYNC_RUNNER=0` and the default. Since the SM110 runtime proof was on
> 13.0.48, this is at least weak evidence the sm_110 support is toolchain-robust across the
> 13.0/13.2 boundary. I saw #82 touching the same axis, which is why I mention it.
>
> I did **not** get the `llama_paged_engine` parity gate run — the checkpoint download was
> still in flight when my maintenance window closed. That is the arm that would actually test
> strict token-exactness under 13.2, and I would like to finish it.
>
> ## 4. Blocker: Mistral-Nemo is rejected at the tokenizer
>
> `mistralai/Mistral-Nemo-Instruct-2407` (BF16, `MistralForCausalLM`, 40 layers, GQA 32/8,
> head_dim 128, untied lm_head, `sliding_window: null`) fails immediately after the model
> loads:
>
> ```
> server: loading model from <path> (config .../config.json, tokenizer .../tokenizer.json)
> server: fatal: tokenizer: unrecognized pre-tokenizer split regex:
> [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|...
> ```
>
> This is the Tekken pre-tokenizer regex. It looks architecture-independent — the shape class
> matches your own Mistral parity gate, so this is a tokenizer-support gap rather than an
> sm_110 issue. I can open it separately if you would prefer it not be bundled with the Thor
> report.
>
> ## 5. Async runner
>
> `VT_ASYNC_RUNNER` default-ON does **not** fault at op level on sm_110 — `test_cuda_backend`
> and `test_cuda_ops` pass identically with and without it. Given `docs/STATUS.md` scopes the
> sm_87 verification to `VT_ASYNC_RUNNER=0` because of an illegal memory access, this at least
> suggests that fault does not reproduce on Thor at this level. **I would not call the async
> runner verified on sm_110 from this** — the paged-engine forward is the test that matters and
> I have not run it yet.
>
> ## Two small documentation notes
>
> - `docs/STATUS.md:1482` still describes sm_110 as build-only, which contradicts the
>   `BACKEND-CUDA-SM110` row and `docs/BUILD.md:80`. Happy to PR the one-line fix.
> - For anyone else building in a CUDA container: `CMakeCache.txt` reports
>   `CMAKE_CUDA_ARCHITECTURES=75` even on a correct `-DVLLM_CPP_CUDA_ARCHITECTURES=110` build,
>   because `CMakeLists.txt:363` uses a non-cache `set()` that shadows the cache entry. Reading
>   the cache to confirm the target arch is misleading; the `build.ninja` gencode line is
>   accurate. Not a bug, but it cost me a while.
>
> ## Offer
>
> I have ongoing access to this board. If useful I can: finish the `llama_paged_engine` parity
> run under 13.2, redo the throughput sweep isolated and with repeats, test the decode CUDA
> graph path, or try other checkpoints. Tell me what would be most valuable and I will run it.
>

## Resolution

The current tree contains the Tekken implementation in commit `ce0fa29bb5217bb11e0fe642727eb25f7072c8b2` dated 2026-08-11, which names issue #168. The binding issue comments split the sm_110 decode finding to #325/#326 and the KV sizing finding to #357.
