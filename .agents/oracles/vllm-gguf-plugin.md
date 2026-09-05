# `vllm-gguf-plugin` — vLLM's own GGUF support, moved out of tree

**This is not a competitor's implementation of GGUF. It is vLLM's.** In-tree
GGUF support was deprecated in vLLM (`vllm-project/vllm#39583`) and migrated to
`vllm-project/vllm-gguf-plugin`, in the same organisation. At our parity pin
`5559679229`, `docs/features/quantization/gguf.md:7` names that repository as
where GGUF support went, `:12` gives the install line and `:19` the serve
recipe. So a GGUF path that this plugin **serves** is a path the PRIMARY oracle
serves, and `AGENTS.md` §"When vLLM has no implementation" would not admit a
secondary oracle for it. That consequence is conditional on the serving, and it
is now established on exactly one device and no other: see "It has emitted
tokens, on gfx1151" below. `gateable = no` stays as recorded, because the
registry value is one flag for the whole oracle and the CUDA measurement
[#2624](https://github.com/mudler/vllm.cpp/issues/2624) owes is still owed.
`llama-cpp` remains the admissible denominator for the Q4_K_M arm. See "What
this record does NOT license".

```oracle-pin
id = vllm-gguf-plugin
role = secondary
upstream = https://github.com/vllm-project/vllm-gguf-plugin
scope = GGUF quantization for the pinned vLLM, which carries no in-tree GGUF at 5559679229; never a behavior source, because vLLM's own model implementations are what it feeds
pin = d4c1f0d082fc7cd4350da56689109a01c1f29d6c
pin_label = post-v0.0.5 HEAD, 2026-08-31
pinned_on = 2026-09-03
gateable = no
evidence = #2624
```

**`role = secondary` is a registry mechanic, not a claim about rank.** The
registry admits exactly one `primary` and it is `vllm`
(`scripts/check-oracle-pins.py:64,183`), so every other file takes the other
value whatever its provenance. `vllm-omni.md` sits in the same position: a
first-party `vllm-project` repository recorded as `secondary`. Read the `scope`
line, not the role word. This plugin supplies no behavior of its own — it feeds
weights to vLLM's own `Qwen3_5ForConditionalGeneration`, so where it runs, the
answer it produces is vLLM's answer.

## Why the pin is a commit and not `v0.0.5`

**The released wheel cannot serve this architecture, and that is measured
rather than assumed.** `v0.0.5` was published 2026-08-10; the Qwen3.5/3.6
adapter landed in `4ec8d61` on 2026-08-19 (`[Models] support Qwen3.5/3.6
multimodal GGUF and MTP (#98)`), and `git ls-tree v0.0.5
vllm_gguf_plugin/weights_adapter/` lists only `base`, `default`, `diffusion`
and `gemma3`. Point the release at a `qwen35` GGUF and no adapter matches, so
`config_parser.py:44-52` falls back to `transformers`'
`MODEL_FOR_CAUSAL_LM_MAPPING_NAMES` and the generic `default` adapter, whose
name map knows nothing of `ssm_alpha`, `ssm_out` or `attn_qkv`. Whether that
presents as a config error or as a weight-mapping error is NOT measured here;
what is measured is that the file which knows those names is absent from the
release. The pin is therefore the HEAD that
carries the adapter, `d4c1f0d`, whose `weights_adapter/qwen3_5.py:45-64` is
the file that knows `ssm_alpha`, `ssm_beta`, `ssm_conv1d`, `ssm_out` and
`attn_qkv` are one Gated Delta Net layer.

## What is established, and what is not

**ESTABLISHED — source feasibility on this fleet, no lease needed.**

- **aarch64 is upstream-supported.** `.github/workflows/release.yml:192,247`
  build `cu129` and `cu130` aarch64 wheels on `ubuntu-24.04-arm` inside
  `pytorch/manylinuxaarch64-builder`, and `:354` uploads the aarch64 wheel to
  PyPI. PyPI carries `...-manylinux_2_28_aarch64.whl` for every release from
  `0.0.2` onward. This is the opposite of
  [`exllamav3.md`](exllamav3.md)'s finding, where seven translation units are
  x86-only host code and the project publishes no aarch64 wheel at all.
- **No x86 assumption in the extension.**
  `grep -rn '__x86_64__|__aarch64__|__AVX|__SSE|immintrin|_mm_|_mm256'
  vllm_gguf_plugin/csrc/` returns zero hits over all eleven files: there is no
  host-CPU intrinsic to port. The device code carries 25 `__CUDA_ARCH__`
  guards and **every one is a lower bound that `sm_121` clears** — 23 in
  `csrc/gguf/vecdotq.cuh` at `>= 610` around `__dp4a`, 2 in
  `csrc/gguf/ggml-common.h:1086,1090` at `>= 800` around `__float2bfloat16`.
  The only other architecture branch is CUDA versus ROCm: `USE_ROCM`
  appears in five files, `csrc/cuda_compat.h`, `csrc/gguf/vecdotq.cuh`,
  `csrc/gguf/mmq.cuh`, `csrc/gguf/moe.cuh` and `csrc/gguf/ggml-common.h`.
  Nothing excludes `sm_121a` by construction.
- **It imports against OUR pin, not only against its own contemporary.** All
  44 `from vllm...` symbol imports in `vllm_gguf_plugin/` resolve in the
  pinned checkout at `5559679229`: 42 by an automated walk of each module's
  top-level definitions and star re-exports, and the two the depth-limited
  walk missed, `get_tensor_model_parallel_rank` and
  `...world_size`, by hand at `vllm/distributed/parallel_state.py:2039,2044`.
  `weights_adapter/qwen3_5.py:14-15` needs
  `vllm.transformers_utils.configs.qwen3_5.Qwen3_5Config` and
  `...qwen3_5_moe.Qwen3_5MoeConfig`, and both files exist at the pin.
- **The architecture is registered at the pin.**
  `vllm/model_executor/models/registry.py:572` registers
  `Qwen3_5ForConditionalGeneration`, and `:647` `Qwen3_5MTP`. The artifact's
  own `config.json` declares `model_type = "qwen3_5"`, which is the first
  entry of the adapter's `QWEN35_MODEL_TYPES` (`qwen3_5.py:29-34`) and maps to
  that architecture in `QWEN35_ARCHITECTURES` (`:38-43`).
- **Every tensor of our artifact maps. MEASURED 2026-09-03, offline.**
  Running the adapter's own `build_qwen35_text_mapper`,
  `build_qwen35_vision_mapper` and `build_qwen35_mtp_mapper` over the tensor
  names of `Qwen3.8-27B-Q4_K_M.gguf` (866) and `mmproj-BF16.gguf` (334),
  1200 names in union: **1185 mapped, 0 unmapped**, 15 held back as the
  `blk.64` MTP block, and all 15 of those mapped by the MTP mapper. Samples:
  `blk.0.attn_qkv.weight -> model.language_model.layers.0.linear_attn.in_proj_qkv.weight`,
  `blk.0.ssm_a -> ...linear_attn.A_log`,
  `v.blk.0.attn_qkv.weight -> model.visual.blocks.0.attn.qkv.weight`.
  `_gdn_value_head_layout` (`qwen3_5.py:169-184`) resolves on this config to
  `heads_per_group = 3` from `linear_num_value_heads = 48` over
  `linear_num_key_heads = 16`, so the GDN head-tiling restore is armed rather
  than skipped.
- **Every quantized type in the artifact has a kernel. MEASURED.** The
  backbone's 866 tensors are `Q4_K` 294, `F32` 456, `Q6_K` 67, `Q5_K` 48 and
  `Q8_0` 1; the mmproj's 334 are `F32` 224 and `BF16` 110. All four quantized
  types appear in the plugin's dequantize dispatch table
  (`triton/dequantize/interface.py:74,77,78,79`) and in its GEMM table
  (`triton/gemm/interface.py:64,67,68,69`), so no tensor of ours falls into an
  unimplemented arm. Counting the types is not running them.
- **Upstream tests this shape, on smaller siblings.**
  `tests/test_multimodal_gguf.py:103-115` declares `QWEN35_CONFIG`
  (`unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M` against `Qwen/Qwen3.5-0.8B`) and
  `QWEN35_MOE_CONFIG` (`...35B-A3B...`), both multimodal, both Q4_K_M, and the
  README's coverage table lists "Qwen 3.5 — Q4_K_M backbone with BF16
  projector" — the exact shape of our checkpoint. `QWEN35_MODELS_TO_TEST`
  (`:121-124`) marks only the MoE case `pytest.mark.slow`; the dense 0.8B case
  carries no mark, and the `:118-119` params are Gemma3's. Neither was run
  here, so this is upstream's declaration of coverage and not a green from this
  session.
- **That 15-tensor block is exactly what the llama.cpp oracle drops.**
  [`llama-cpp.md`](llama-cpp.md) records `b10451` loading 851 of 866 tensors
  and ignoring all 15 of `blk.64`. The plugin maps all 15. The two oracles are
  therefore not offered the same model, and the vLLM side can run the MTP head
  the llama.cpp side cannot.

**MEASURED IN A LEASE, on `thor:gpu0`, 2026-09-03.** Five `rc` jobs ran the
same staged script on `thor:gpu0` (aarch64, NVIDIA Thor, capability 11.0,
driver 595.78); the last is `40a1d8dd-529b-456b-8e46-2789354cce5a`, run dir
`/workspace/ggufplugin/20260903T012806Z`. No `ssh` was used. That run's engine
log, its `rc` job log, the driver, the recipe and the offline arch measurement
are committed under
[`docs/bench-evidence/vllm-gguf-plugin-thor-20260903/`](../../docs/bench-evidence/vllm-gguf-plugin-thor-20260903/),
so every number below can be checked without the share.

- **It BUILDS.** `PLUGIN_BUILD_RC=0` under nvcc 13.0.88 in every one of the
  five jobs, producing `vllm_gguf_plugin-0.0.5-cp310-abi3-linux_aarch64.whl`.
  The **cold** from-source build is job 1's, 29.128 s; jobs 2-5 reuse that
  worker's build tree and take 3.142, 3.085, 3.125 and 3.113 s, the last being
  the cited run. Job 1 is the one that answers the source-feasibility question,
  and it is also the job that then died on `ModuleNotFoundError: No module
  named 'gguf'` at registration (`REGISTRATION_RC=1`), so its build number and
  its registration verdict come from the same run and must be quoted together.
  Each job does rebuild rather than serve a cached wheel — all five wheel
  sha256 values differ (`c5a77c04`, `8c8c1463`, `ec8ad598`, `9464269f`,
  `a79bcc2d`) — but the build is not byte-reproducible, so the sha256 does not
  by itself identify which arch list produced which wheel; the
  `cuobjdump --list-elf` postcondition on the INSTALLED object does.
- **It REGISTERS.** `REGISTRATION_RC=0`; entry point
  `('gguf', 'vllm_gguf_plugin:register')`; `_C_gguf.abi3.so` imports; and
  `torch.ops._C_gguf` exposes `ggml_dequantize` and `ggml_moe_a8_vec`. In the
  engine, vLLM registers the plugin's `GGUFModelLoader` for `load_format=gguf`
  and its `GGUFConfigParser` for `config_format=gguf`.
- **The pinned vLLM wheel runs on a SECOND aarch64 capability.** Built on GB10,
  it imports on Thor as `0.1.dev1+g555967922` and reports
  `cuda True NVIDIA Thor`.
- **THE 17 GB Q4_K_M GGUF LOADS.** `Resolved architecture:
  Qwen3_5ForConditionalGeneration`, then, in the cited run,
  `Model loading took 16.3 GiB memory and 530.729566 seconds`
  (`gen-20260903T012806Z.log:41`). The three jobs that got this far read
  745.171284 s, 533.210486 s and 530.729566 s in that order; the first is a
  cold page cache and none of the three is a benchmark. Neither
  `No HF name for N Qwen3.5 GGUF tensor(s), skipping` (`qwen3_5.py:257`) nor
  the separate MTP line `No HF name for N Qwen3.5 MTP tensor(s), skipping`
  (`:424`) appears in any of the five `gen.out` files, so no backbone, vision
  or MTP tensor went unmapped on the real loader. The engine chose
  `Triton/FLA GDN prefill kernel (head_k_dim=128)` and the `FLASHINFER`
  backend, so the Gated Delta Net path is the one that ran.
- **The tokenizations agree with the llama.cpp gate's**, `[6, 5, 6, 7, 11, 7]`
  over the six recorded prompts, from two different vocab sources.

**NOT ESTABLISHED, and it is the half that decides the flag: NO TOKEN.**
`AGENTS.md` requires the oracle to build **and run** the model. A 16.3 GiB
load is not a generation.

- **The forward is blocked on `thor:gpu0`. Where it THROWS is not where it
  FAILED.** Memory profiling dies with `CUDA error: no kernel image is
  available for execution on the device`. The throw happens **inside the
  plugin's own quantized GEMM**:
  `linear.py:1763` -> `vllm_gguf_plugin/quantization/linear.py:261` ->
  `:49 _fused_mul_mat_gguf` -> `vllm_gguf_plugin/ops.py:207 ggml_mul_mat_a8` ->
  `torch.ops._C_gguf.ggml_mul_mat_a8`, with C++ frames #15-#17 inside
  `vllm_gguf_plugin/_C_gguf.abi3.so` calling
  `torch_call_dispatcher("aten::new_zeros")` (`torch/csrc/stable/ops.h:975`).
  Only below that does `at::native::FillFunctor<c10::BFloat16>` appear, at
  frame #2. That fill is the **first device-touching statement** of
  `ggml_mul_mat_a8` (`csrc/gguf/gguf_kernel.cu:228`; the five lines above it
  read shapes and take a `DeviceGuard`, and none of them launches anything),
  and what raised is the launch check torch runs after it.
- **A launch check reports whichever launch last failed, not necessarily its
  own.** `cudaErrorNoKernelImageForDevice` is returned to the launching thread
  and sits in the last-error slot until some `cudaGetLastError` collects it.
  The plugin's `csrc/` contains **zero** launch checks across all eleven files,
  and vLLM's own `LAUNCH_ACTIVATION_GATE_KERNEL`
  (`csrc/libtorch_stable/activation_kernels.cu:235-288`) checks none either. So
  the frame that raised need not be the frame that failed, and reading the
  bottom of this stack as "PyTorch's bf16 fill has no image" is not a
  conclusion the stack supports.
- **PyTorch is not the missing image, and that is now measured rather than
  asserted.** The wheel the job installs (`job.sh:99`, `pip install -q
  torch==2.13.0`, no index URL) is `torch 2.13.0+cu130`,
  `cp312-cp312-manylinux_2_28_aarch64`, `git_version
  cf30153c4c131c8164ee7798e5022d810682e2cb`. Its `torch/lib/libtorch_cuda.so`
  carries SASS for `sm_80, 90, 100, 103, 110, 120, 121` — **508 `sm_110` cubin
  entries over 449 fatbin blocks**. The same box has run `torch 2.13.0+cu130`
  at `capability (11, 0)` before: `CUBLAS_OK (bf16 1024x1024 matmul executed)`,
  `.agents/specs/lease-runtime-staging.md:70-78`. That leg staged its own
  relocated tree rather than pip-installing the wheel, so it corroborates the
  fatbin measurement and does not replace it.
- **Of the three objects measured, the one with no image for this device is
  vLLM's own wheel.** `vllm/_C_stable_libtorch.abi3.so`, built on GB10 and
  installed from `/workspace/oracle-vllm/`, carries `sm_80, 89, 90, 120` SASS
  and **no `sm_110`**; 64 of its 70 fatbin blocks are `sm_120`-only with no PTX at all.
  The block holding `vllm::act_and_mul_kernel` (`silu_and_mul`, the op
  `Qwen2MoeMLP.forward` calls between the two GGUF projections) and the block
  holding `vllm::rms_norm_kernel` are both `sm_120` ELF only.
  `_moe_C_stable_libtorch.abi3.so` is `sm_80` + `sm_120` only. The plugin's own
  object, by contrast, carries `sm_110` and nothing else — `cuobjdump
  --list-elf` said so in the lease (`_C_gguf.abi3.1.sm_110.cubin`,
  `INSTALLED_CUBIN_ARCH_OK`) and the offline parse agrees exactly, which is
  what makes the parse trustworthy for the other two objects. Instrument,
  inputs and output:
  `docs/bench-evidence/vllm-gguf-plugin-thor-20260903/fatbin-arch-20260903.txt`.
- **The failing launch is `torch.ops._C.silu_and_mul`, by elimination over a
  closed window.** `gate_up_proj` (`qwen2_moe.py:113`) took the identical route
  two lines earlier and its own `new_zeros` did NOT raise, so the last-error
  slot was clean there and the failing launch is between the two `new_zeros`
  calls. That window holds exactly three launches: `gate_up`'s
  `quantize_row_q8_1_cuda` and `ggml_mul_mat_*_q8_1_cuda`, both unchecked, and
  `act_fn` at `:114`, which is `torch.ops._C.silu_and_mul`, also unchecked.
  Nothing else in it launches a checked kernel. The first two are in an object
  that carries an `sm_110` image, measured twice and independently, and
  `cudaErrorNoKernelImageForDevice` is returned exactly when no compatible
  image exists — so they cannot produce it. The third is
  `vllm::act_and_mul_kernel`, `sm_120` ELF with no PTX. It is also the FIRST
  vLLM `_C` kernel the whole forward reaches, which is why the load and every
  preceding norm and GDN attention layer ran first.
- **The counter-argument that kept this open is refuted, by execution.** It was
  that `rms_norm_kernel` is in the same arch class, so a uniformly imageless
  `_C` should have killed layer 0 long before the MLP. Those norms never enter
  `_C`. Qwen3.5's norm is `GemmaRMSNorm` under an alias
  (`qwen3_next.py:28`), which widens its weight to `f32` before calling the IR
  op (`layernorm.py:157`), while `vllm/kernels/vllm_c.py` admits its `rms_norm`
  and `fused_add_rms_norm` impls only for `weight.dtype == x.dtype`;
  `vllm/ir/op.py:344-351` then skips the impl and takes `native`, the next
  entry of the `['vllm_c', 'native']` priority the run's own logged
  `kernel_config` sets. `ir_dispatch_probe.py` executes those exact source
  constructs out of the pinned checkout and reports `False`, `False` and a
  `True` positive control on a dtype-matched weight.
- **What is still owed is the direct observation.** This is a deduction from a
  closed enumeration plus two offline measurements, not something anyone
  watched happen. `CUDA_LAUNCH_BLOCKING=1`, or a `cudaGetLastError` after each
  launch, sees it directly. The deduction predicts that `dgx:gpu0` does not hit
  this wall, since it is `sm_121a` and the wheel's `sm_120` SASS covers it, and
  the queued job there reads that prediction out. Working or not, the wall is
  the pinned vLLM wheel's own arch list — a build input, not GGUF, not the
  plugin, and not this checkpoint. Deduction, instruments and transcripts:
  `docs/bench-evidence/vllm-gguf-plugin-thor-20260903/no-kernel-image-attribution-20260903.txt`.
- **`dgx:gpu0` is untried.** It is `sm_121a`, the capability the vLLM wheel
  was built for and the one its `sm_120` code actually covers; `rc` job
  `7a45427e-ad86-4042-aeea-cdf8db535a54` is queued there with the identical
  script and is what owes the token.
- **The `dgx` history is a caution, not a promise.**
  [#1129](https://github.com/mudler/vllm.cpp/issues/1129) recorded that no vLLM
  leg could run a model there by a lease-compliant path, and
  `.agents/specs/mtp-k-gt-1.md` records the host consumed in the step after
  `torch.compile` at two `gpu_memory_utilization` values, the lower one
  rebooting the box. `/workspace/oracle-vllm/README-WHEELS.md` states the
  `FLASHINFER-ONLY` wheel later generated coherent tokens there; that is
  another session's claim, read off the share and not re-derived here, and it
  was a bf16 checkpoint rather than a GGUF one.

The measurement is owed by
[#2624](https://github.com/mudler/vllm.cpp/issues/2624) and the method is in
[`../specs/oracle-vllm-gguf-qwen35.md`](../specs/oracle-vllm-gguf-qwen35.md).

## It has emitted tokens, on gfx1151

This record was written while no leg of this plugin had produced a token, and
that is no longer true. On 2026-09-03, inside `rc` leases on `strix:gpu0`
(`gfx1151`, RDNA 3.5), this plugin at **this exact pin**
`d4c1f0d082fc7cd4350da56689109a01c1f29d6c`, under the pinned vLLM
`5559679229`, loaded the gated 17,106,775,008-byte Qwen3.8-27B Q4_K_M GGUF and
generated 48 greedy tokens for each of the six gate prompts, in four legs, two
per configuration, byte-identical within each configuration. Evidence:
[`../../docs/bench-evidence/oracle-vllm-gfx1151-20260903.md`](../../docs/bench-evidence/oracle-vllm-gfx1151-20260903.md),
issue [#2740](https://github.com/mudler/vllm.cpp/issues/2740).

Read the scope narrowly. It says the plugin serves this checkpoint on one AMD
board. It says nothing about `thor:gpu0` or `dgx:gpu0`: the CUDA question, the
`cudaErrorNoKernelImageForDevice` attribution above, and the token #2624 owes
are all untouched by an AMD run, and `gateable` stays `no` until that
measurement exists.

## What this record does NOT license

It does not move the Q4_K_M arm's token gate off llama.cpp. That gate reads
`FAIL` against `b10451` ([#2534](https://github.com/mudler/vllm.cpp/issues/2534),
`docs/bench-evidence/qwen38-27b-q4km-token-gate-20260823.md`) and stays exactly
as recorded. Nothing in #2740 rescores it, and nothing here does either.

The gfx1151 tokens above do not by themselves re-declare that gate. What they
remove is one argument for keeping it: a denominator that has never emitted a
token cannot replace one that has, and on this one device that objection no
longer applies. Re-declaring the gate is a spec-and-fresh-review job and belongs
to [#2546](https://github.com/mudler/vllm.cpp/issues/2546). It is worth stating
plainly what #2740 measured and what it did not: on the same board, the same
artifact and the same six prompts, the pinned vLLM diverges from llama.cpp
`b10451` on 3 of 6 compiled and 4 of 6 eager, against our own ROCm arm's 3 of 6.
Two engines disagreeing with a third shows the third is not ground truth. It
does not show either of the two is right.
