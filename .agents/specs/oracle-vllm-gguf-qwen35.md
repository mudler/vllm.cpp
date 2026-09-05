# Can the pinned vLLM serve our Qwen3.8-27B Q4_K_M GGUF, and displace llama.cpp as this arm's oracle?

Owning issue: [#2624](https://github.com/mudler/vllm.cpp/issues/2624). No
roadmap row owns it; the un-run measurement is listed under `## Owed` below,
which is what `AGENTS.md` §"Every change starts from an issue" requires of an
issue whose `Row:` line is a dash.

## The question, and why it is not academic

`AGENTS.md` §"When vLLM has no implementation" admits a secondary oracle **only
where vLLM implements nothing**. The `Qwen3.8-27B-Q4_K_M.gguf` arm is gated
against llama.cpp `b10451` today, and that gate reads `FAIL` on 5 of 6 prompts
([#2534](https://github.com/mudler/vllm.cpp/issues/2534),
`docs/bench-evidence/qwen38-27b-q4km-token-gate-20260823.md`), every divergence
a rank-2 loss under 0.18 logits with 282 of 288 steps at rank 1. It blocks
[#2497](https://github.com/mudler/vllm.cpp/issues/2497) and, under §Gates, every
speed and memory number this arm could produce on any backend.

If the pinned vLLM can read that same GGUF, then llama.cpp was never the
admissible denominator for it and the gate is pointed at the wrong side. That
is a different claim from "our arm is correct" and this spec establishes
neither; it establishes only whether a better denominator is reachable.

## Scope

- **IN:** source feasibility of `vllm-project/vllm-gguf-plugin` against our
  pin and our fleet, cited to `file:line`; a build and a generation run inside
  an `rc` lease; the oracle record and the registry argument.
- **OUT, and stated because the temptation is real:** any speed, latency or
  memory number, and any cross-engine ratio. §Gates forbids a performance
  result while this arm's token gate has not passed, and nothing here changes
  that. Also out: touching the llama.cpp gate's verdict or its evidence
  documents, advancing the vLLM parity pin, and `src/` or `include/`.

## Upstream anchors

| What | Where |
|---|---|
| GGUF left vLLM's tree for a first-party repository | `${VLLM_SOURCE}/docs/features/quantization/gguf.md:7`, install `:12`, serve recipe `:19`, `--hf-config-path` variant `:40-46` — all at the parity pin `5559679229` |
| the deprecation this migration implements | `vllm-project/vllm#39583`, cited by the plugin's own `README.md:3-5` |
| the architecture, registered in-tree at our pin | `${VLLM_SOURCE}/vllm/model_executor/models/registry.py:572` (`Qwen3_5ForConditionalGeneration`), `:647` (`Qwen3_5MTP`) |
| the plugin revision this spec pins | `vllm-project/vllm-gguf-plugin` `d4c1f0d082fc7cd4350da56689109a01c1f29d6c`, 2026-08-31 |
| the adapter that knows this architecture | `vllm_gguf_plugin/weights_adapter/qwen3_5.py`, added by `4ec8d61` (2026-08-19, `#98`), renamed by `c5e3717` (`#124`) |

## Design

Nothing is written. The work is: establish feasibility from source, then run
the thing, then record what it did.

**The artifact.** `Qwen3.8-27B-Q4_K_M.gguf` on the house NAS, which a leased
worker reaches as `/workspace/ckpt/qwen38-27b-q4km/Qwen3.8-27B-Q4_K_M.gguf`,
17,106,775,008 bytes, `gguf_sha256` `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
— **measured 2026-09-03 over the worker-visible copy**, and identical to the
value #2534 records, so the staged file and the gated file are one artifact
rather than two with the same name — from
`unsloth/Qwen3.8-27B-GGUF` @ `fe1e2a23d973adb629709749dc4f6756df66ef10`. Its
own `config.json` sits beside it: `model_type = "qwen3_5"`,
`architectures = ["Qwen3_5ForConditionalGeneration"]`, 64 text layers,
`full_attention_interval = 4`, `linear_num_key_heads = 16`,
`linear_num_value_heads = 48`, a `vision_config`, and `mtp_num_hidden_layers = 1`.
`mmproj-BF16.gguf` (931,146,432 B) is the vision tower.

**The recipe, read out of the plugin rather than guessed.** A model path
ending in `.gguf` triggers `plugin.py:56-76`, which sets `quantization`,
`load_format` and `config_format` to `gguf`, moves the path into
`model_weights`, and rewrites `model` to a **config source**. That source is
`hf_config_path` if given, else the `tokenizer` when it is not itself a GGUF
reference, else the GGUF's own parent directory
(`plugin.py:34-46`, `config_parser.py:62-64`). This run passes the HF
checkpoint as `tokenizer`, so the HF config is what is read; a `config.json`
was staged beside the `.gguf` as well, and on this path it is a fallback that
is not consulted. The tokenizer comes from the HF checkpoint rather than from
GGUF vocab conversion, which vLLM's own doc calls "time-consuming and
unstable" (`gguf.md:38`). A `vision_config` without an `mm_proj` is a hard
error by name (`weights_adapter/qwen3_5.py:210-215`), and an explicit path is
accepted through `model_loader_extra_config={"mm_proj": ...}`
(`loader.py:92-99`, `gguf_utils.py:147-159`); the `mmproj-BF16.gguf` beside
the checkpoint is staged and passed that way.

**Where it runs.** `dgx:gpu0` inside an `rc` lease, never `ssh`. The job reuses
two things already measured on that box rather than rediscovering them: the
`FLASHINFER-ONLY` vLLM wheel at `/workspace/oracle-vllm`
(`README-WHEELS.md` — vLLM's default `FLASH_ATTN` carries no `sm_12x` SASS
there and cannot run at all), and the toolchain, venv, watchdog and
postcondition shape of `/workspace/a2q1-neartie/job.sh`, which ran green.

**Ordering is a design decision, not a detail.** The job builds the plugin
wheel and copies it to `/workspace` **before** anything loads a model, so a job
killed at the 90-minute ceiling still leaves a durable artifact and a
registration proof rather than nothing. `.agents/specs/oracle-wheel-in-lease.md`
records a venv-staging job killed mid-copy that left only a partial tree.

## Risks

| Risk | What it would look like | Mitigation |
|---|---|---|
| the box OOM-reboots | `.agents/specs/mtp-k-gt-1.md` records the host consumed in the step **after** `torch.compile`, at `gpu_memory_utilization` 0.75 **and** 0.30, the 0.30 run rebooting the box | `enforce_eager=True` skips compilation entirely; a 1 Hz `MemAvailable` watchdog kills the process **group**, because vLLM forks an `EngineCore` that survives a single-pid kill |
| eager mode read as a benchmark configuration | someone quotes a number out of this run | no timing is taken, `gen.py` records none, and §Gates' ban on `--enforce-eager` is about a **denominator**; this asks a correctness question |
| an instrument failure wearing the shape of a verdict | `ModuleNotFoundError: No module named 'vllm'` 13 minutes after the job looked healthy (#1416) | identity is asserted from `cd /` before anything else, the wheel is copied to a PEP 427-conforming name first, and the job aborts with an explicit "this is an INSTRUMENT failure" line |
| the released plugin wheel is silently the wrong one | a config-parse failure that reads like "vLLM cannot do qwen35" | measured: `v0.0.5` predates the adapter; the job builds `d4c1f0d` from a staged archive whose sha256 it asserts before untarring |
| aarch64 turns out to be an x86-only project, as `exllamav3` did | seven translation units of host intrinsics and no aarch64 wheel | measured false, see below |

## Tests and gates

There is no unit test to write; the gate is the oracle's own bar. `AGENTS.md`:
the oracle must **demonstrably build and run the model**, and constructing a
config proves nothing.

| Gate | Command | State |
|---|---|---|
| the registry stays consistent both ways | `python3 scripts/check-oracle-pins.py` | **PASS**, 14 oracles. Red-before/green-after was taken on this change: with `pinned_on` omitted it printed `vllm-gguf-plugin.md: oracle-pin is missing required key 'pinned_on'` and `oracle-pins FAILED (1 error(s))`, and adding the key turned it green. The checker **exits 1** on that failure and 0 on the pass: `raise SystemExit(main())` at `scripts/check-oracle-pins.py:311-317`, re-measured by the same mutation on 2026-09-03. An earlier draft of this line, and the commit message of `43b1678e`, said it "exits 0 either way"; that was false and neither can be rewritten, so it is corrected here |
| the plugin builds on aarch64 + CUDA | `pip wheel . --no-build-isolation` in an `rc` lease | **PASS**, measured on `thor:gpu0` 2026-09-03. `PLUGIN_BUILD_RC=0` in all five jobs under nvcc 13.0.88. The cold build is job 1's 29.128 s; the cited run (job 5) rebuilds over a warm tree in 3.113 s |
| the plugin registers, and its CUDA extension imports | `import vllm_gguf_plugin._C_gguf`, entry-point listing | **PASS**, same lease. `REGISTRATION_RC=0` |
| the GGUF loads into the pinned vLLM | the engine's own weight loader | **PASS**, same lease. 16.3 GiB in 530.73 s in the cited run (745.17 s in job 3, cold cache), no unmapped tensor reported |
| the model emits tokens | `/workspace/ggufplugin/gen.py`, six prompts x 48 greedy tokens | **NOT REACHED on `thor:gpu0`**; the forward throws inside the plugin's own op, and the launch that failed is `_C.silu_and_mul` in vLLM's own `sm_110`-less wheel, by elimination over a closed window (below). `PENDING` on `dgx:gpu0`, `rc` job `7a45427e-ad86-4042-aeea-cdf8db535a54` |
| the six-prompt token gate against this oracle | not attempted | **NOT REACHED** |

## Evidence

**ESTABLISHED, offline, no lease.**

1. **aarch64 is upstream-supported, so `exllamav3`'s failure mode does not
   apply.** `.github/workflows/release.yml:192` and `:247` build `cu129` and
   `cu130` aarch64 wheels on `ubuntu-24.04-arm` in
   `pytorch/manylinuxaarch64-builder`, and `:354` uploads the aarch64 wheel to
   PyPI. Every release from `0.0.2` (2026-06-11) carries a
   `...-manylinux_2_28_aarch64.whl` on PyPI, and the `v0.0.5` GitHub release
   carries four wheels, two of them aarch64. **The dispatch that opened this
   work recorded "the only published wheel is x86_64" and "the wheel installs
   nowhere as-is"; both are false.**
2. **No x86 assumption anywhere in the extension.**
   `grep -rn '__x86_64__|__aarch64__|_M_X64|__AVX|__SSE|immintrin|x86intrin|__ARM_NEON|_mm_|_mm256|__builtin_ia32' vllm_gguf_plugin/csrc/`
   returns **zero** hits over all eleven files, and `__aarch64__` appears
   nowhere either — there is no host-CPU intrinsic to port. The device code
   does carry 25 `__CUDA_ARCH__` guards, and an earlier draft of this claim
   said it carried none; they are all **lower bounds and none excludes
   `sm_121`**: 23 in `csrc/gguf/vecdotq.cuh` read
   `#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 610 || defined USE_ROCM`
   around the `__dp4a` dot products (`:48-58` is the first), and 2 in
   `csrc/gguf/ggml-common.h:1086,1090` gate `__float2bfloat16` at `>= 800`.
   `sm_121` clears both. The only other architecture branch in `csrc/` is CUDA
   versus ROCm (`csrc/cuda_compat.h`, `csrc/gguf/mmq.cuh`). Contrast
   `.agents/oracles/exllamav3.md`, where seven translation units are x86-only
   host code behind `#ifdef __linux__`.
3. **It imports against OUR pin.** All 44 `from vllm...` symbol imports in
   `vllm_gguf_plugin/` resolve in the pinned checkout at `5559679229`. 42 were
   resolved by walking each target module's top-level definitions and its star
   re-exports; the walk is depth-limited and reported
   `get_tensor_model_parallel_rank` and `get_tensor_model_parallel_world_size`
   as missing from `vllm/distributed/__init__.py`, and both were then confirmed
   by hand at `vllm/distributed/parallel_state.py:2039,2044`, re-exported by
   that `__init__.py:5`'s `from .parallel_state import *`. Two apparent misses
   that were an instrument limit, not a gap — said here rather than smoothed
   into "all 44 resolved".
   The two that matter most are `weights_adapter/qwen3_5.py:14-15`,
   `vllm.transformers_utils.configs.qwen3_5.Qwen3_5Config` and
   `...qwen3_5_moe.Qwen3_5MoeConfig`; both modules exist at the pin. This is
   static resolution, not a runtime signature check.
4. **The released wheel cannot serve this architecture.** `v0.0.5` was
   published 2026-08-10; the Qwen3.5 adapter landed 2026-08-19 in `4ec8d61`.
   `git ls-tree v0.0.5 vllm_gguf_plugin/weights_adapter/` lists `base`,
   `default`, `diffusion` and `gemma3` and nothing else. So the pin has to be
   a commit. What the release would do instead is NOT measured: with no
   matching adapter, `config_parser.py:44-52` falls back to `transformers`'
   `MODEL_FOR_CAUSAL_LM_MAPPING_NAMES` and the generic `default` adapter,
   whose name map knows nothing of `ssm_alpha`, `ssm_out` or `attn_qkv`, and
   whether that surfaces as a config error or a weight-mapping error was not
   tried.
5. **Every tensor of our artifact maps. MEASURED 2026-09-03.** Running the
   adapter's own `build_qwen35_text_mapper`, `build_qwen35_vision_mapper` and
   `build_qwen35_mtp_mapper` over the 866 backbone and 334 mmproj tensor
   names, 1200 in union: **1185 mapped, 0 unmapped**, 15 held back as the
   `blk.64` MTP block, and the MTP mapper maps all 15. `WeightsMapper`'s
   `_map_name_with_shard` was copied verbatim from
   `${VLLM_SOURCE}/vllm/model_executor/models/utils.py:81-135` at the pin,
   because `vllm` is not importable on the coordinating host; the three name
   dictionaries and the three builders were executed out of the plugin
   checkout itself. Samples:
   `blk.0.attn_qkv.weight -> model.language_model.layers.0.linear_attn.in_proj_qkv.weight`,
   `blk.0.ssm_a -> ...linear_attn.A_log`,
   `v.blk.0.attn_qkv.weight -> model.visual.blocks.0.attn.qkv.weight`.
   `_gdn_value_head_layout` (`qwen3_5.py:169-184`) resolves to
   `heads_per_group = 3` here, so the GDN head-tiling restore is armed rather
   than skipped.
   **This supersedes the dispatch's "862 of 866" figure**, which came from
   `gguf.get_tensor_name_map(MODEL_ARCH.QWEN35, 65)` — a mechanism the plugin
   does not use for this architecture.
   The script that did this is deliberately NOT committed: it needs a clone of
   an upstream repository this tree does not vendor, so in-tree it would be a
   file that cannot run. The method above is the reproduction instruction.
6. **Every quantized type in the artifact has a kernel. MEASURED.** Backbone
   866 tensors: `Q4_K` 294, `F32` 456, `Q6_K` 67, `Q5_K` 48, `Q8_0` 1. mmproj
   334: `F32` 224, `BF16` 110. All four quantized types are in the plugin's
   dequantize dispatch table (`triton/dequantize/interface.py:74,77,78,79`)
   and its GEMM table (`triton/gemm/interface.py:64,67,68,69`), so nothing in
   this checkpoint lands in an unimplemented arm. Counting the types is not
   running them.
7. **Upstream tests this shape on smaller siblings, and says so.**
   `tests/test_multimodal_gguf.py:103-115` declares `QWEN35_CONFIG`
   (`unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M` against `Qwen/Qwen3.5-0.8B`) and
   `QWEN35_MOE_CONFIG` (`...35B-A3B...`), both multimodal Q4_K_M, and
   `README.md`'s coverage table lists "Qwen 3.5 — Q4_K_M backbone with BF16
   projector". `QWEN35_MODELS_TO_TEST` (`:121-124`) marks only the MoE case
   `pytest.mark.slow`; the dense 0.8B case is unmarked, and the params at
   `:118-119` are Gemma3's. Neither was run here: this is upstream's
   declaration, not a green from this session.
8. **The two oracles are not offered the same model.**
   `.agents/oracles/llama-cpp.md` records `b10451` loading 851 of 866 tensors
   and ignoring all 15 of `blk.64`, 289,527,808 bytes. The plugin maps all 15,
   and `Qwen3_5MTP` is registered at our pin. So the vLLM side can run an MTP
   head the llama.cpp side cannot, and any future paired run must say which
   side ran what.

## What the lease measured, on `thor:gpu0`, 2026-09-03

Five `rc` jobs ran the same staged `/workspace/ggufplugin/job.sh` on
`thor:gpu0` (aarch64, NVIDIA Thor, compute capability 11.0, driver 595.78,
125,748 MB of unified memory). The last is `40a1d8dd-529b-456b-8e46-2789354cce5a`,
run directory `/workspace/ggufplugin/20260903T012806Z`. **The cited run's
engine log, its `rc` job log, `job.sh`, `gen.py` and the offline arch
measurement are committed under
`docs/bench-evidence/vllm-gguf-plugin-thor-20260903/`.** They were on the NAS
and in a scratchpad when this record was first written, which is why two
numbers came from the wrong run and the stack was read one frame short: no gate
could see either. Every job took the
device through `rc`; no `ssh` was used.

**PASS — the plugin builds from source on aarch64 + CUDA.** `PLUGIN_BUILD_RC=0`,
**29.1 s** of wall clock under nvcc `13.0.88` (CUDA release 13.0), producing
`vllm_gguf_plugin-0.0.5-cp310-abi3-linux_aarch64.whl`. Step 1's whole question
is answered YES, empirically and not by inference. `exllamav3`'s failure mode
does not occur here.

**PASS — the pinned vLLM wheel runs on a second aarch64 capability.** The
`FLASHINFER-ONLY` wheel was built on GB10; on Thor it imports as
`vllm.__version__ = 0.1.dev1+g555967922` from `site-packages` and reports
`cuda True NVIDIA Thor`. `IDENTITY_RC=0`.

**PASS — the plugin registers, and its kernels are reachable.**
`REGISTRATION_RC=0`. The entry point is `('gguf', 'vllm_gguf_plugin:register')`,
`_C_gguf.abi3.so` imports, and `torch.ops._C_gguf` exposes `ggml_dequantize`
and `ggml_moe_a8_vec`. In the engine, vLLM logs
`Registered model loader <class 'vllm_gguf_plugin.loader.GGUFModelLoader'> with
load format 'gguf'` and `Registered config parser
<class 'vllm_gguf_plugin.config_parser.GGUFConfigParser'> with config format
'gguf'`.

**PASS — the config resolves to the in-tree architecture.**
`Resolved architecture: Qwen3_5ForConditionalGeneration`, engine config
`load_format=gguf, quantization=gguf, dtype=torch.bfloat16`,
`served_model_name=/workspace/ckpt/qwen38-27b-q4km/Qwen3.8-27B-Q4_K_M.gguf`.
The config source resolved to the HF checkpoint exactly as `plugin.py:34-46`
predicts when a non-GGUF `tokenizer` is given.

**PASS, and this is the substantive one — THE 17 GB Q4_K_M GGUF LOADS.**
`Model loading took 16.3 GiB memory and 745.171284 seconds`, and the adapter
logged **no** `No HF name for N Qwen3.5 GGUF tensor(s), skipping` warning, which
is the line it emits for any tensor it cannot map. The offline mapping result
is therefore confirmed on the real loader. The engine selected
`Using Triton/FLA GDN prefill kernel (requested=auto, head_k_dim=128)` and
`AttentionBackendEnum.FLASHINFER`, so the Gated Delta Net path is the one that
was built.

**PASS — the tokenizations agree with the llama.cpp gate's.** vLLM's tokenizer
over the six recorded prompts gives lengths `[6, 5, 6, 7, 11, 7]`, matching the
`PROMPT_IDS` lengths `docs/bench-evidence/qwen38-27b-q4km-token-gate-20260823.md`
records for llama.cpp `b10451` exactly. The ids are in the run's `gen.out`. That
removes tokenizer divergence as a confound for any future comparison, and it is
a stronger statement than the length match alone because the two sides read
different vocab sources (GGUF vocab there, HF tokenizer here).

**NOT REACHED on this device, and where it throws is not where it failed.**
The memory profiling forward dies with
`CUDA error: no kernel image is available for execution on the device`.

**Where it throws.** Inside the plugin's own quantized GEMM, not inside
PyTorch. The Python frames, from
`docs/bench-evidence/vllm-gguf-plugin-thor-20260903/gen-20260903T012806Z.log`:

```text
vllm/model_executor/models/qwen2_moe.py:115   out, _ = self.down_proj(out)
vllm/model_executor/layers/linear.py:1763     self.quant_method.apply(...)
vllm_gguf_plugin/quantization/linear.py:261   fused_mul_mat_gguf_op(...)
vllm_gguf_plugin/quantization/linear.py:49    ops.ggml_mul_mat_a8(...)
vllm_gguf_plugin/ops.py:207                   torch.ops._C_gguf.ggml_mul_mat_a8
```

and the C++ frames continue **into** the plugin before they reach torch:
frames #15-#17 are in `vllm_gguf_plugin/_C_gguf.abi3.so`, which calls
`torch_call_dispatcher("aten::new_zeros")` at
`torch/include/torch/csrc/stable/ops.h:975`; `new_zeros` -> `zero_` ->
`fill__Scalar` -> `fill_kernel_cuda` -> frame #2
`at::native::gpu_kernel_impl_nocast<at::native::FillFunctor<c10::BFloat16>>`.
That `new_zeros` is the **first device-touching statement** of
`ggml_mul_mat_a8` (`csrc/gguf/gguf_kernel.cu:228`), before the plugin launches
anything of its own: the five lines above it read `X.sizes()`, compute the
padding and take a `DeviceGuard`, and none of them touches a kernel. The previous record stopped one frame short of the caller and read the
bottom of the stack as the cause.

**Why the bottom frame is not the cause.** `cudaErrorNoKernelImageForDevice`
is handed to the launching thread and stays in the last-error slot until some
`cudaGetLastError` collects it; torch runs one after every kernel it launches,
and almost nothing else does. The plugin's `csrc/` has **zero** launch checks
across all eleven files, and vLLM's `LAUNCH_ACTIVATION_GATE_KERNEL`
(`csrc/libtorch_stable/activation_kernels.cu:235-288`) has none either. So the
frame that raises is only the next frame that looked.

**Measured offline, on the exact artifacts, with a positive control.**
`docs/bench-evidence/vllm-gguf-plugin-thor-20260903/fatbin_arch.py` reads the
`.nv_fatbin` section of an ELF and lists the SM version of every fatbin entry.
Its control is the plugin's own object, whose arch coverage the lease
established independently with `cuobjdump --list-elf`
(`_C_gguf.abi3.1.sm_110.cubin`): the parser reports exactly `ELF sm_110` over
100% of the section, so it agrees with `cuobjdump` where both ran.

| object | source | SASS archs |
|---|---|---|
| `vllm_gguf_plugin/_C_gguf.abi3.so` | built in the lease, `TORCH_CUDA_ARCH_LIST=11.0` | `110` (control: matches `cuobjdump`) |
| `torch/lib/libtorch_cuda.so` | `torch 2.13.0+cu130`, `cp312-manylinux_2_28_aarch64`, `git_version cf30153c4c` | `80, 90, 100, 103, 110, 120, 121` — **508 `sm_110` entries** |
| `vllm/_C_stable_libtorch.abi3.so` | the GB10-built pinned wheel | `80, 89, 90, 120` — **no `sm_110`**; 64 of 70 blocks are `sm_120`-only with no PTX |
| `vllm/_moe_C_stable_libtorch.abi3.so` | same wheel | `80, 120` — **no `sm_110`** |

So the claim that the stock aarch64 torch wheel "carries no `sm_110`" is false.
`.agents/specs/lease-runtime-staging.md:70-78` had already recorded
`torch 2.13.0+cu130` running `CUBLAS_OK (bf16 1024x1024 matmul executed)` on
this very box at `capability (11, 0)` — from a relocated tree staged on
`/workspace` rather than from this job's pip-installed wheel, so it
corroborates the table above and is not what establishes it. The object in the process that has no
image for `sm_110` is **vLLM's own extension**: the fatbin block holding
`vllm::act_and_mul_kernel` (the `silu_and_mul` that `Qwen2MoeMLP.forward` runs
at `qwen2_moe.py:114`, between the two GGUF projections) and the block holding
`vllm::rms_norm_kernel` are both `sm_120` ELF only.

**Which launch set the flag: `torch.ops._C.silu_and_mul`, by elimination over
a closed window.** The window is bounded, and that is the first result:
`gate_up_proj` (`qwen2_moe.py:113`) took the identical `_fused_mul_mat_gguf`
route two lines earlier and its `new_zeros` bf16 fill did **not** raise, so the
last-error slot was clean at that point and the failing launch happened between
the two `new_zeros` calls. Exactly three launches sit in that window, and
nothing else in it launches a kernel torch checks — `SiluAndMul.forward_cuda`
allocates with `torch.empty`, which does not fill, and the reshapes in the
plugin's `apply()` are metadata-only.

| in the window | object | checked? | has an `sm_110` image? |
|---|---|---|---|
| `quantize_row_q8_1_cuda` | `_C_gguf.abi3.so` | no | **yes** |
| `ggml_mul_mat_*_q8_1_cuda` | `_C_gguf.abi3.so` | no | **yes** |
| `_C.silu_and_mul` (`:114`) | `_C_stable_libtorch.abi3.so` | no | **no** |

`cudaErrorNoKernelImageForDevice` is returned exactly when no compatible image
exists for the current device, so a launch out of the plugin's object cannot
produce it: that object's coverage was measured twice and independently,
`cuobjdump --list-elf` in the lease and the offline parse here, and both say
`sm_110`. `vllm::act_and_mul_kernel` is `sm_120` ELF with no PTX, so on
`sm_110` it can neither load nor JIT. It is also the **first** vLLM `_C` kernel
the whole forward reaches, which is why a 16.3 GiB load and every preceding
norm and GDN attention layer ran first.

**The counter-argument that kept this open is refuted, by execution.** It was
that `rms_norm_kernel` sits in the same arch class, so a uniformly imageless
`_C` should have killed layer 0 long before the MLP at `qwen3_next.py:554`.
Those norms never enter `_C`. Qwen3.5's norm is `GemmaRMSNorm` under an alias
(`qwen3_next.py:28`), and `GemmaRMSNorm` widens its weight to `f32` before
handing it to the IR op (`layernorm.py:157`), while `vllm/kernels/vllm_c.py`
registers its `rms_norm` and `fused_add_rms_norm` impls with `supports_args`
predicates that require `weight.dtype == x.dtype`. `vllm/ir/op.py:344-351` then
skips an impl whose `supports_args` is `False` and takes the next entry of the
priority list, which the run's own logged `kernel_config` gives as
`['vllm_c', 'native']`.

`ir_dispatch_probe.py` proves that by execution rather than by reading: it
locates those three constructs in the pinned checkout by regex, `exec`s the
exact source bytes, and evaluates the predicate on a bf16 activation and the
`f32` weight `GemmaRMSNorm` produces. It prints `False` for `rms_norm`, `False`
for `fused_add_rms_norm`, and a `True` **positive control** on a dtype-matched
weight, which is what a plain `RMSNorm` passes. A probe that could only ever
print `False` would prove nothing; the control is what makes the `False`
readable. Upstream renaming any of the three makes the probe fail to find it
rather than answer for a version that no longer exists.

`rotary_embedding_kernel` is `sm_120`-only too, which constrains the crashing
layer rather than feeding the deduction: had this been one of Qwen3-Next's
full-attention layers, rotary would have raised earlier, so the crash is in a
linear-attention layer, which carries no rotary embedding at all.

**What is still owed is the direct observation.** This is a deduction from a
closed enumeration plus two offline measurements, not something anyone watched
happen. `CUDA_LAUNCH_BLOCKING=1`, or a `cudaGetLastError` after each launch,
sees it directly, and this record does not have that run. The deduction makes a
falsifiable prediction: `dgx:gpu0` does **not** hit this wall, because it is
`sm_121a` and the wheel's `sm_120` SASS covers it — the wheel was built there
and has generated text there before. The queued job is what reads that
prediction out, and it is also what answers the token question.

Either way the wall is the pinned vLLM wheel's own arch list, which is a build
input. It is not GGUF, not the plugin, and not this checkpoint. Deduction,
instruments and transcripts:
`docs/bench-evidence/vllm-gguf-plugin-thor-20260903/no-kernel-image-attribution-20260903.txt`.

### Three instrument failures, recorded because each read like a verdict

Each cost a run, and each would have been reported as "vLLM cannot serve this
GGUF" by anyone reading only the last line.

1. **`pip install --no-deps` starved the plugin of its own dependencies.**
   `pyproject.toml` declares `gguf>=0.17.0` and `huggingface_hub>=1.26.0`;
   `--no-deps` was used to stop pip re-resolving `vllm` and `torch`, and it
   dropped those two as well. The job died on
   `ModuleNotFoundError: No module named 'gguf'` at registration. Fixed by
   naming the two dependencies explicitly and keeping `--no-deps`.
2. **No `__main__` guard.** vLLM V1 spawns its `EngineCore` with
   multiprocessing `spawn`, which re-imports the driver script in the child.
   Without a guard the module body ran again there and the engine reported
   `Engine core initialization failed` under a `freeze_support()` bootstrap
   message. Fixed by moving `gen.py`'s body into `main()`.
3. **`TORCH_CUDA_ARCH_LIST` unset, then a same-version reinstall that did
   nothing. This one is an instrument repair that did NOT change the result,
   and it must not be read as a second, fixed failure.** The first draft of the
   job left the arch list unset on the theory that torch's `cpp_extension`
   derives it from the present device. Setting it from
   `nvidia-smi --query-gpu=compute_cap` produced a wheel carrying `sm_110`, and
   `pip install` into the reused venv then printed "already satisfied" and kept
   the arch-less object, because the plugin's version is always `0.0.5`. Adding
   `--force-reinstall` plus a `cuobjdump --list-elf` postcondition on the
   INSTALLED object made the installed object provably `sm_110`. **A flag is
   not a postcondition, and a wheel is not what gets called** — that lesson
   stands. What it did not do is change the failure: jobs 3, 4 and 5
   (`20260903T010058Z` arch-less, `011701Z` arch-listed, `012806Z`
   force-reinstalled) throw at the identical site with a byte-identical call
   chain, differing only in timestamps and pids. So the section above describes
   **one** unchanged failure, not two, and the arch of the plugin's installed
   object is now known not to be its cause.

**NOT ESTABLISHED, and this is what keeps `gateable = no`.** The bar
`AGENTS.md` sets is that the oracle demonstrably **builds AND runs** the model.
The build half is now measured and green. The run half is not: **no token has
been emitted.** A load is not a generation, and the section above says exactly
where it stopped and why. The oracle record therefore reads `gateable = no`
with `#2624` named, which is what `AGENTS.md` asks of a lane whose measurement
is still owed. Also still unestablished: any runtime API incompatibility
between the plugin's HEAD and our five-weeks-older pin beyond the paths this
run exercised, and anything at all about `dgx:gpu0`, where the only job is
still queued.

## The registry decision, and its argument

**A new id, `vllm-gguf-plugin`, with `role = secondary`.** The alternative was
to fold it into `vllm.md`, and that fails on two counts. Mechanically,
`scripts/check-oracle-pins.py` requires **exactly one** `oracle-pin` block per
file, and `vllm.md:14-24` already spends that one block on the vLLM pin
itself, so a second pin cannot live there. (An earlier draft said `vllm.md`
"holds no pin of its own", following `.agents/oracles/README.md:18-20`, which
says `vllm.md` "points at it rather than restating it". Both are inaccurate:
`vllm.md:19` does hold `pin = 5559679229...`, restated as identity while
`.agents/upstream-sync.md` stays the block the sync cycle advances. The
one-block-per-file conclusion does not depend on that, and the mutation below
proves the mechanic directly.) Substantively, the plugin is a separate
repository with its own release cadence and its own revision, which is the
definition of a thing that needs its own pin.

**`role = secondary` is a registry mechanic and NOT a claim about rank.** The
checker admits one `primary` and requires it to be `vllm`
(`check-oracle-pins.py:64,183`), so every other file takes the other value
whatever its provenance. `vllm-omni.md` is the precedent and it is exact: a
first-party `vllm-project` repository, `role = secondary`, its own id, its own
file. This plugin supplies no behavior of its own — it adapts weights into
vLLM's own `Qwen3_5ForConditionalGeneration` — so where it runs, the answer is
vLLM's answer, and the `scope` line says so rather than leaning on the role
word.

**Adding a row to the AGENTS.md table is a rule change, and here is the
argument for it.** The table's own rule is that a secondary oracle is
admissible only where vLLM implements nothing. At `5559679229` vLLM implements
no in-tree GGUF: it was deprecated and moved out. So GGUF is precisely the case
the rule describes, and the row is an application of the rule rather than an
exception to it. The row's "reach for it when" column says the consequence out
loud, because that consequence is the whole reason this work exists: **a GGUF
path this plugin serves is a path the primary oracle serves, and llama.cpp is
not admissible for it.**

## Owed

- **THE TOKEN. Everything else is done; this is not.**
  [#2624](https://github.com/mudler/vllm.cpp/issues/2624). Build, install,
  registration, config resolution and a complete 16.3 GiB weight load are all
  measured green on `thor:gpu0`; the forward is blocked there at the first
  quantized GEMM's launch check, reporting a failure attributed above to
  vLLM's own `_C.silu_and_mul`, whose object carries no `sm_110`. `rc` job
  `7a45427e-ad86-4042-aeea-cdf8db535a54` is queued on `dgx:gpu0`, which is
  `sm_121a` and is where that wheel was built, running the identical
  `bash /workspace/ggufplugin/job.sh`. Until it emits tokens, `gateable = no`
  stands and nobody may say vLLM serves this checkpoint.
- **Whether `thor:gpu0` can be made to run this stack at all**, and if so
  with which vLLM build. Out of scope here and unowned; it is a
  fleet-capability question, not a GGUF one. Note that the in-tree record does
  **not** support the earlier reading of it: `.agents/environment.md:409-424`
  and `.agents/specs/lease-runtime-staging.md:11-12,70-78` record a staged
  `torch 2.13.0+cu130` on that same box that "imports torch, initializes CUDA,
  **runs a bf16 matmul**, and compiles and executes a Triton kernel", logging
  `CUBLAS_OK (bf16 1024x1024 matmul executed)` at `capability (11, 0)`. The
  bf16 matmul is not a JIT, so that record is evidence *for* torch working at
  `sm_110`, not a Triton-only exception to it. The concrete next question is
  the `sm_110`-less pinned vLLM wheel, which is a build input rather than a
  property of the box.
- **Whether the llama.cpp gate should move.** Not owned here and not decided
  here. It depends entirely on the item above, and #2534 stays exactly as
  recorded until then.
- **Runtime API compatibility between plugin HEAD and our pin.** Only static
  import resolution was checked. A signature change in
  `FusedMoEMethodBase`, `LinearMethodBase` or `ConfigParserBase` between
  2026-07-26 and 2026-08-31 would surface only at run time.
- **The x86_64 side is untested here** and is not needed. The three CUDA
  fleet devices are `dgx:gpu0` (GB10), `thor:gpu0` (NVIDIA Thor) and
  `orin:gpu0` (AGX Orin) (`.agents/environment.md:68-70`), and they are
  aarch64: the leased worker runs ELFs through
  `/lib/ld-linux-aarch64.so.1` (`:330`) and the staged oracle wheel is
  `...-cp312-cp312-linux_aarch64.whl` (`:456`). `strix:gpu0` is the x86_64
  box and it is ROCm `gfx1151`, which vLLM does not support, so the plugin's
  ROCm branch is unreachable on this fleet too. That is why the published
  wheel's platform tag mattered at all, and why finding an aarch64 one is the
  finding it is.

## Stop conditions

- The plugin does not build on aarch64: record it, name the failing file and
  the compiler message, leave `gateable = no`, and stop. That is a complete
  and publishable answer.
- The model loads but emits nothing, or the box reboots: record the step it
  reached and the `MemAvailable` trace, leave `gateable = no`, stop. An
  absence is never a result.
- Tokens come out: `gateable = yes` needs the evidence file to exist in this
  tree, so write `docs/bench-evidence/` first and only then flip the flag.
  Even then, this spec does not move #2534's verdict; a paired run does.

## Now

No roadmap row changes state. This change adds one oracle record, one
AGENTS.md registry row and this spec. The measurement is submitted and
unfinished, and every surface says so.
