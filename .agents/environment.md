# Environment & assets

- **Dev box (no GPU)**: CPU reference backend + engine logic + CI development.
- **GPU box**: `ssh dgx.casa` — DGX Spark, GB10 (Blackwell, **sm_121**),
  ~119 GB unified memory, 20 cores, CUDA toolkit 13.0.88 (nvcc); compute
  capability 12.1 → sm_121. Unified memory: both gate models fit
  in bf16; the machine is memory-bandwidth-bound (~273 GB/s class) — decode
  parity is a bandwidth/launch-overhead game, hence CUDA graphs + fused
  kernels in T0. Give each active claim its own `~/work/<claim>/` directory;
  never share a build tree between agents.
  - Non-interactive SSH does not put nvcc on PATH — prepend
    `export PATH=/usr/local/cuda/bin:$PATH` in remote build commands.
  - Oracle venv: `~/venvs/vllm-oracle` is now a canonical symlink to the
    validated `~/venvs/vllm-oracle-v0.25.0-stage`; the immediately restorable
    v0.24.0 directory is preserved at
    `~/venvs/vllm-oracle-v0.24.0-retired`. The active stack is pip vLLM 0.25.0,
    FlashInfer Python/cubin 0.6.13, Torch 2.11.0+cu130, CUTLASS DSL 4.5.2,
    Humming kernels 0.1.10, Transformers 5.13.1 and Ninja 1.13.0. Its serving
    dependencies are pandas 2.2.3, python-dateutil 2.9.0.post0, pytz 2024.2 and
    tzdata 2024.2. Install/serving report SHA-256 values are
    `ab786eee…c297` / `536385d8…f506`; executable vLLM/Ninja hashes are
    `ec6d76ff…96c` / `abf71487…10b`, and the sorted freeze hash is
    `cf1636cc…fa5f`.
  - The only dependency-check exception is NVIDIA's
    `nvidia-cusparselt-cu13==0.8.0` wheel: PyPI served the aarch64 wheel
    (`sha256:400c6ed1…77c`), its library is an AArch64 ELF and direct
    `ctypes.CDLL`/Torch imports pass, but its internal WHEEL tag is
    `manylinux2014_sbsa`, so `pip check` reports it unsupported. This is a
    recorded vendor-tag defect, not silently treated as a green `pip check`.
  - Lock-held production-graph validation on the exact 27B snapshot passed both
    offline generation (16 input IDs, one output ID) and the actual text-only
    server: `/health` 200 plus `/v1/completions` 200 with exact 1+1 usage and
    `finish_reason=length`. Server log/response SHA-256 are
    `f56be69a…3787` / `82307db4…8e1` under
    `~/work/vllm-oracle-v0.25.0-stage-validation/2026-07-12-server-smoke`.
    The smoke rate is non-binding. Its first offline inference emitted one
    causal-conv Triton JIT warning, which remains a warmup/trace audit item.
    Online-gate manifests hash pandas package/distribution files plus Ninja and
    reject missing/drifted dependencies before the GPU lock; profiler launches
    prepend the venv `bin` to spawned EngineCore `PATH`.
  - **GPU mutex:** every CUDA test/model/serve/benchmark/profile holds
    `flock /tmp/gpu` for the whole job or multi-arm series WHEN other agents may run GPU work concurrently (sole owner verified idle via `nvidia-smi` may skip), following
    `/home/mudler/_git/skills/sharing-a-gpu-with-flock/SKILL.md`. Compilation,
    source inspection and file transfer do not need the lock. Never kill an
    unowned PID.
  - Disk cleanup 2026-07-10 reclaimed ~368 GB from unrelated cached model sets,
    April-era autoresearch logits/F16-GGUF cache artifacts, the vLLM compile
    cache and stale rebuildable CUDA build trees. Active latency/PR workspaces,
    gate checkpoints, APEX GGUF evidence and sources were preserved; the volume
    had 359 GB free afterward. Maintain at least 200 GB headroom before adding
    competitor images.
- **Apple/Metal box**: `ssh 192.168.68.103` — Mac mini, Apple M4 (10 CPU
  cores), 16 GB unified memory, arm64, macOS 26.5.2. Use it for the MLX-backed
  `vt::` backend, Metal op parity, and small-model bring-up. It cannot hold the
  27B/35B gate models; gate-scale Apple performance needs a larger-memory Mac.
  **Re-verified 2026-07-22 (`CLAIM-BACKEND-FANOUT-1`), correcting the stale
  2026-07-10 line:** only the **Command Line Tools** are installed, NOT full
  Xcode — so the offline `metal` shader compiler is absent (`xcrun -sdk macosx
  metal` fails). That does **not** block MSL: runtime compilation via
  `newLibraryWithSource:` was verified working, together with a numerically
  correct dispatched compute kernel. **CMake IS already installed** (brew
  4.1.0 at `/opt/homebrew/bin`, missing from the non-interactive PATH — always
  `export PATH=/opt/homebrew/bin:$PATH` in remote commands); `ninja` is not
  (make works). **MLX is NOT installed** (`brew install mlx` -> 0.32.0, pulls
  `python@3.14`) and is not required for native-MSL bring-up. Device facts:
  `hasUnifiedMemory=YES`, `MTLGPUFamilyApple9` + `Metal3`, SIMD width 32,
  32 KiB threadgroup memory, 11.84 GiB recommended max working set, ~30 GiB
  free disk. Our tree configures AND builds there under AppleClang 21 with
  three Clang-only `-Werror` fixes, and 108,952 portable-tier assertions pass.
  **Updated 2026-07-22 (W0 landed):** the FULL tree (library + every test) now
  builds `-Werror`-clean on the M4 with the Metal backend ON, and the fix count
  is **seven**, not three — a full build surfaced four more than the spike's
  lib-only probe (see the fan-out spec § Work breakdown "W0 landed"). Configure
  with plain `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`: `VLLM_CPP_METAL`
  defaults to `AUTO` and turns itself ON for an Apple host with an ObjC++
  compiler. Add `-DVLLM_CPP_METAL=OFF` for a CPU-only A/B.

  **TWO PRE-EXISTING macOS TEST GAPS — expected failures, not regressions.**
  Both fail identically with `-DVLLM_CPP_METAL=OFF`, i.e. they are unrelated to
  Metal, and neither is fixed yet:
  - `test_serve_low_tools` — the Python bench tooling calls Linux-only
    `os.sched_getaffinity` (`tests/tools/test_gdn_packed_component.py`) and
    `POSIX_FADV_DONTNEED` (`tools/bench/drop_file_cache.py`).
  - `test_safetensors` — `MappingRssKb` reads `/proc/self/smaps`, which macOS
    does not provide, so it returns 0 and the RSS assertions cannot hold.

  `test_capi` and `test_openai_conformance` are ctest-PARALLELISM flakes on this
  box (and on Linux); they pass on rerun. Prefer `ctest -j 3`.

  **LOCALAI WORKER — must be DOWN for any timing/benchmark work on this box
  (user-directed 2026-07-22).** It is a **root LaunchDaemon**, not a container
  and not a user LaunchAgent:

  | | |
  |---|---|
  | Unit | `system/com.localai.worker` |
  | Plist | `/Library/LaunchDaemons/com.localai.worker.plist` (root:wheel) |
  | Program | `/Users/mudler/local-ai/local-ai worker` (a NATS-driven worker) |
  | Properties | `keepalive | runatload` — so `kill`ing the PID is NOT enough, launchd restarts it |
  | Log | `/Users/mudler/local-ai/worker.log` |
  | State observed 2026-07-22 | **running**, PID 327, RSS ~51 MB, up 1d08h, **MEASURED idle: 0.0% CPU, and `ioreg IOAccelerator PerformanceStatistics` reports `Device Utilization % = 0`, `Renderer Utilization % = 0`, `Tiler Utilization % = 0` — it holds NO GPU work.** Log shows only periodic `NATS backend.list` events (~1 per 6 h); no model loaded |

  ```sh
  # inspect (works WITHOUT root)
  launchctl print system/com.localai.worker
  # stop  (NEEDS root; bootout, because KeepAlive would restart a killed process)
  sudo launchctl bootout system/com.localai.worker
  # restore to the observed state
  sudo launchctl bootstrap system /Library/LaunchDaemons/com.localai.worker.plist
  launchctl print system/com.localai.worker | grep state   # expect: running
  ```

  **NOT STOPPED during W0**, for two reasons, both recorded deliberately:
  (1) stopping it needs an interactive `sudo` password and this box has no
  passwordless sudo (`sudo -n true` -> "a password is required"), so an agent
  cannot do it unattended; (2) W0 took **no timing measurement whatsoever** —
  every gate is a functional/correctness assertion — so contention could not
  affect any recorded result. **The next agent doing MLX-vs-ours benchmarking
  MUST get the user to run the bootout above first; any Metal timing taken with
  this daemon up is VOID.** Note also three `actions.runner.localai-org-*` GitHub
  Actions runners as user LaunchAgents (PIDs 599/600/601) which can start CI jobs
  on this box at any time — quiesce those too before a benchmark series
  (`launchctl bootout gui/$UID/actions.runner.localai-org-<name>.<label>`).

  **STILL NOT STOPPED as of the 2026-07-22 MLX baseline run** — same reason
  (no passwordless sudo). The MLX numbers in
  [docs/BENCHMARKS.md](../docs/BENCHMARKS.md) are therefore recorded
  **`BLOCKED-ON-SUDO` / INDICATIVE, not binding**; the recipe is a one-command
  re-run once the user boots the daemon out. **Second contender found the same
  session and not anticipated by the earlier note:** the desktop **aerial video
  wallpaper** — `WallpaperAerialsExtension` (PID 472, **8.2% CPU**) plus
  `VTDecoderXPCService` (PID 518, 2.2%) — decodes video continuously and touches
  the GPU; it is the actual source of the ~1.47 load average on an otherwise idle
  box. Disable it (System Settings -> Wallpaper, or log the console user out)
  before any binding run. It was left untouched.

  **MLX IS NOW INSTALLED (2026-07-22), via the venv route as recommended** —
  brew was NOT used, so `python@3.14` never entered `/opt/homebrew/bin` and the
  PATH our macOS builds use is unchanged:

  ```sh
  /usr/bin/python3 -m venv ~/mlx-venv && ~/mlx-venv/bin/pip install -U pip mlx-lm
  ```

  | | |
  |---|---|
  | Resolved versions | **`mlx` 0.29.3, `mlx-metal` 0.29.3, `mlx-lm` 0.29.1** — the CLT python 3.9.6 caps the resolve BELOW brew's 0.32.0. Record this: an unpinned competitor arm is not a floor |
  | Location | `~/mlx-venv` (off every build PATH), `~/hf-cache` (3.2 GB model cache) |
  | Model | `mlx-community/Qwen3-1.7B-bf16` @ rev `9cd6692855d3e06772228e9a962b2606359b2d24` |
  | Ships prebuilt | `mlx/lib/mlx.metallib` **104,894,650 bytes** + `libmlx.dylib` — so CONSUMING MlX needs no Xcode, but BUILDING it from source does (`xcrun metal`), which this box cannot do |
  | Device probe | `mx.metal.device_info()` -> `applegpu_g16g`, `max_recommended_working_set_size` 12,713,115,648 (11.84 GiB), `max_buffer_length` 9,534,832,640 |
  | Removal | `rm -rf ~/mlx-venv ~/hf-cache` — neither is on any PATH our builds consult |

## Benchmark models on dgx.casa

- `~/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4`
  (snapshot complete, ~22G, 3 safetensors shards — re-downloaded 2026-07-03
  after the original snapshot was found incomplete)
- `~/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-NVFP4`
- `~/work/apex/qwen36_35b/Qwen3.6-35B-A3B-APEX-*.gguf` (GGUF-gate inputs)
- `~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B` (fast tests)

## Gate model architecture (from GGUF metadata, arch `qwen35moe`)

40 blocks = 10 × (3 GDN + 1 full-attn); hidden 2048; full-attn GQA 16q/2kv,
partial RoPE 64 dims (MRoPE sections [11,11,10,0]), rope base 1e7; MoE 256
experts top-8 + 1 shared (expert FFN 512); GDN: conv kernel 4, 16 groups,
inner 4096, state 128; context 262144.

## Prior art on dgx.casa (mudler's llama.cpp patch series — mine for GB10 kernels)

- `~/killgate_series/` — NVFP4 W4A4 FP4 MMA prefill, qwen35moe NVFP4
  quant/dedup, MoE decode regraph
- `~/llama-phase93-qwen3next-gqa-bcast`
- `~/llama-phase84-attn-only-source`

## TODO

- Binding immutable `3f256ab` remains **55/124 axes pass, 69 fail** against
  vLLM v0.25.0. Finalized c2 root `179a0fc` already maps the executed path and
  selects the complete **193 vs 97** GDN projection mismatch. W1 merged BA is
  implemented/`GATING`. Clean pushed `581d335` under
  `~/work/vllm.cpp-gdn-ba/immutable-581d335…` passes the exact CUDA 13.0.88 /
  CUTLASS / Triton-AOT build, packed F32/BF16 capture/replay, strict memcheck,
  merged/split 27B and inert native-35B gates; the isolated BF16/decomposed
  control fails the token near-tie. Immutable `0091cd1`, finalized by pushed `8a1f923`, is
  `complete-structural`: both exact-c2 arms pass all 24 local range contracts at
  merged 963/145 versus split 1,011/193, with 48 BF16-only removals and unchanged
  selected non-BF16 families. Clean `f344dec` closes packed W1D2/G2 for
  default+rollback 27B **235/235**, 35B/GGUF inertness and strict safety.
  `benchmark_binding=false`. Close paired node traces and the c2/c16 component
  before qkvz.
  Independently remove **22.920
  GiB** host-weight mirror and overlapping source pages. No 35B performance
  command runs before all 27B axes pass.
- Keep the existing SGLang v0.5.13 P1 evidence immutable. The distinct
  shared-prefix gate pins v0.5.15 `f63458b` and image digest `d0a667e`; its PX1
  deterministic 64k/256k harness/counter work is ready after the priority
  cache-off closure. Write the dedicated `KV-MAMBA-ALIGN` spike before PX2,
  then require matched BF16/no-spec capacity, native hit/no-eviction evidence,
  full axes and traces. Never mutate the vLLM oracle while provisioning SGLang.
- ~~Bootstrap CMake + MLX on the M4 host before the Metal backend bring-up.~~
  **RESOLVED/SUPERSEDED 2026-07-22** by the [backend fan-out
  spike](specs/backend-fanout-metal-vulkan-xpu.md): CMake is already present,
  and MLX is **not** a bring-up prerequisite (native MSL compiles at runtime
  with CLT only, so E2 precedes E1). The real prerequisite is spike work item
  `W0` — chiefly the `CMakeLists.txt:304-306` Apple `-force_load` fix, without
  which every static registrar is silently dropped on macOS and even the CPU
  backend fails to register. `brew install mlx` is deferred to work row `M5`.
  **CLOSED 2026-07-22: `W0` LANDED** — the `-force_load` fix is in and
  `test_backend` is 7/7 on the M4, so the M4 is fully usable for backend work.
  **REOPENED in a different role:** MLX must now be installed on the M4 as the
  **competitor BENCHMARK arm** (user directive; `BACKEND-GATE-METAL-MLXLM`),
  which is independent of its demotion as an implementation path. Use the venv
  route recorded in the M4 entry above, and stop the LocalAI worker daemon
  first.
- **Vulkan runtime is already usable and needs no acquisition.** dgx GB10
  enumerates as a real Vulkan `INTEGRATED_GPU` at API 1.4.312 (loader 1.4.328 +
  NVIDIA ICD) with `VK_KHR_cooperative_matrix` v2 and `VK_NV_cooperative_matrix2`;
  the dev box enumerates `llvmpipe` (Vulkan 1.4.318, CPU) for GPU-free CI. Still
  to install before Vulkan work: `libvulkan-dev`, `vulkan-tools`, and a
  **current-SDK** `glslc` — Ubuntu's shaderc 2023.8 is too old for the coopmat2
  feature probe and fails silently into the slow path.
- **No Intel GPU exists on any box here**, so `BACKEND-XPU` end-to-end work is
  HW-BLOCKED; only policy-port, compile coverage and oneAPI CPU-device unit
  numerics are available.
