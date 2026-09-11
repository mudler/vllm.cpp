ID: ISSUE-GH-2392
Title: GGUF residency policy reads the platform probe, not the engine's resolved device
Row: ENG-RESIDENCY-CONFIG
State: OPEN
Kind: UNKNOWN
GitHub: 2392
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-31
Updated: 2026-08-31
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `ENG-GGUF-RESIDENCY-RESOLVED-DEVICE`
>
> Two seams answer "which device will this load run on", and they answer it differently by construction.
>
> **The engine resolves it.** `LoadedEngine::ResolveExplicitDeviceType` (`src/vllm/entrypoints/model_loader.cpp`) returns `kCPU` for an explicit `--device cpu`, with its own comment: *"Explicit CPU never consults the accelerator probe: even on a CUDA-capable build/process this selects the CPU queue."*
>
> **The GGUF residency policy probes it.** `GgufLoadPolicy::FromEnv`, `RouteGgufTensor` and `qwen4_exp_registry.cpp` each call `vllm::platforms::CurrentPlatform().device_type()`, which answers `kCUDA` on any process where `src/vllm/platforms/cuda.cpp`'s `Registrar` saw a usable GPU — regardless of what the caller asked for.
>
> So on a CUDA-capable process, `--device cpu` selects the CPU queue and gets the CUDA residency policy. This affects **every GGUF model**, not one:
>
> - `RouteGgufTensor`'s per-tensor device gate (`DeviceKeepQuantSupported`, `DeviceQuantGatherSupported`) decides residency. `DeviceQuantGatherSupported` is true for `kCPU` alone, so a keep-quant embedding table expands to bf16 that a CPU load would have kept.
> - Four of `FromEnv`'s flags (`keep_quant`, `keep_f16`, `nvfp4_fp4`, `elem_kn_repack`) resolve for the probed device.
> - On `qwen4_exp` it becomes a refusal the user cannot satisfy. The PLE guard (`qwen4_exp_weights.cpp`) throws on any device with no block gather, and its remedy text reads *"Load this model with `--device cpu`"* — the thing the user just did.
>
> **Why it was latent.** Every run that exercised these paths was a CPU-only build, where the probe also answers `kCPU`, so the two sources agreed by accident. It goes live the moment CUDA is enabled in the same build.
>
> **It is on live ground.** A sibling wave landed four CUDA `qwen4_exp` op arms and proved by reachability mutation that they are not reached: on a CUDA build the loader refuses first, with `device 'cuda' has no block-decoding gather kernel ... Load this model with --device cpu`. That refusal is exactly this path.
>
> **Same defect class as #1136**, one level down. `model_loader.cpp` resolves the device once for `CheckDeviceWeightFit` and hands it over, then builds the residency policy for the same load from the probe fifty lines later — the bound and the policy the bound describes name different devices.
>
> **Measured, not inferred.** Reverting the single `qwen4_exp_registry.cpp` line to the probe reds exactly 1 case / 8 assertions of `test_qwen4_exp_gguf_weights`, while the other 3079 assertions stay green — which is the measurement of why this survived.
>
> Fix: thread the resolved device as a value (`ModelSource::device`, required by `FromGguf`; `GgufLoadPolicy::FromEnv(dev)` and `RouteGgufTensor(..., dev)` with no default) rather than widening a global.
>
> Spec: `.agents/specs/gguf-residency-resolved-device.md`

## Resolution

-
