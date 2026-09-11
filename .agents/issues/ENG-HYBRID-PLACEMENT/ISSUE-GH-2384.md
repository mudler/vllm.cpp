ID: ISSUE-GH-2384
Title: --fit is parsed, validated and announced, but nothing ever reads the answer
Row: ENG-HYBRID-PLACEMENT
State: CLOSED
Kind: UNKNOWN
GitHub: 2384
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-31
Updated: 2026-08-31
Closed: 2026-08-31

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## What
>
> `ResolvePlacementFit()` (`include/vllm/config/weight_residency.h:565`, `src/vllm/config/weight_residency.cpp:1171`) has **no production caller**. Declaration, definition, two test assertions — that is all.
>
> ```
> $ git grep -n "\bResolvePlacementFit\b" -- src include
> include/vllm/config/weight_residency.h:565:bool ResolvePlacementFit();
> src/vllm/config/weight_residency.cpp:1171:bool ResolvePlacementFit() {
> ```
>
> Its sibling `ResolvePlacementOverrides()` is live at `model_loader.cpp:177` and `:218`. This one was never picked up.
>
> ## Why it looks alive — the #2382 property, again
>
> Three signals tell an operator it works:
>
> 1. The mutual-exclusion refusal is real (`weight_residency.cpp:698-701`): setting `fit` alongside a manual placement gives a precise by-name startup refusal, which reads as proof that `fit` is an enforced live input.
> 2. `WeightResidencyConfig::Describe()` prints `placement_fit=on` (`weight_residency.cpp:493`) on the startup line.
> 3. `DescribeEnvOverrides()` names `VT_PLACEMENT_FIT (placement.fit)` in the "the environment OVERRIDES" line.
>
> So the user sets `--fit`, watches the engine echo it back twice, and no code ever asks the resolver what the answer was. `docs/ENVIRONMENT.md:174` claims it **"Places routed experts on six architecture families."** It places nothing.
>
> ## This is NOT a small wiring job
>
> Upstream `--fit` at pin `b10451` (`common/fit.cpp`) is an iterative **measure-and-retry loop over repeated dry loads**: `no_alloc` load, `llama_get_memory_breakdown`, per-device free/total from `ggml_backend_dev_memory`, then context reduction (step 2), a cpu-moe probe and back-to-front layer fill by the method of false position (step 3), then front-to-back dense promotion with fractional layer spilling (step 4). It also tunes `n_ctx` and `n_gpu_layers`. Its output is exactly the `-ot` override list this tree already builds by hand.
>
> **The blocking divergence:** upstream fits against **live free** memory. At our install point (`model_loader.cpp:2511` GGUF, `:2905` safetensors) only **static total** is reachable — `GetPlatform(dev).residency_policy().device_memory_total_bytes`. `vt::Backend::DeviceMemoryInfo` needs a constructed backend, and the queue is created by `SelectQueueForModel` *after* both install sites; worse, CUDA does not override it at all (`include/vt/backend.h:100-102`), so it returns unknown there. `interface.h:64-72` deliberately prefers total over free, "because free at load time carries the page cache and whatever else the box is doing".
>
> So a port either fits against a different denominator than the flag it mirrors, or first needs a load-time memory-breakdown equivalent this tree does not have.
>
> ## Also owed: the reachability test cannot reuse the existing observable
>
> `InstallMoePlacementPlan` is unconditional, so a fit-installed plan, a `cpu_moe`-installed plan and a bare plan all yield `resolved_layer_count() == num_hidden_layers`. A test copied from `test_placement_reach.cpp` would pass against a no-op. Discriminating fit needs a **provenance** observable on `MoePlacementPlan` (resolved-by-fit vs stated), or an observable "declined, budget unknown" status.
>
> ## Two holes in the exclusivity refusal, found while reading
>
> - The refusal lives inside the single-document parse; the multi-document merge (`weight_residency.cpp:888-896`) copies `fit` and the override fields field-by-field and **never re-runs the check**, so document A setting `cpu_moe` then document B setting `fit` merges into a state the parser would have refused.
> - `VT_PLACEMENT_FIT=1` beside `VT_CPU_MOE=1` is refused nowhere. Harmless only because nothing reads the bit.
>
> Row: `ENG-HYBRID-PLACEMENT`

## Resolution

GitHub links pull request #2436 as closing issue #2384 on 2026-08-31.
