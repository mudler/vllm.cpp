ID: ISSUE-GH-937
Title: Integrated AMD GPUs: any request with temperature above 0 crashes the process
Row: BACKEND-ROCM
State: OPEN
Kind: UNKNOWN
GitHub: 937
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-15
Updated: 2026-08-15
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## Summary
>
> On an AMD integrated GPU (the GPU built into the CPU chip, like Strix Halo), vllm.cpp crashes as soon as it generates a token using random sampling. Greedy decoding (`--temperature 0`) works fine. Anything else kills the process with a GPU memory error.
>
> This matters because normal API requests use a temperature above 0 by default. So the model appears to work when you test it the usual way, and then dies the moment a real client talks to it.
>
> ## How to reproduce
>
> Same model, same build, one flag different.
>
> Works:
>
> ```
> ./build-hip/examples/vllm-cli --model ~/models/Qwen3-0.6B \
>   --prompt 'The capital of France is' --max-tokens 8 --temperature 0
>
>  Paris. The capital of Italy is Rome
> ```
>
> Crashes:
>
> ```
> ./build-hip/examples/vllm-cli --model ~/models/Qwen3-0.6B \
>   --prompt 'The capital of France is' --max-tokens 8 --temperature 0.7
>
> Memory access fault by GPU node-1 on address 0x7fd150bef000.
> Reason: Page not present or supervisor privilege.
> core dumped
> ```
>
> `ctest -R test_capi` fails the same way.
>
> ## What is going wrong
>
> Some computers let the CPU and the GPU share one pool of memory instead of each having their own. That sharing is not automatically symmetric, and on AMD's RDNA chips it is not:
>
> - The CPU **can** read memory that the GPU allocated.
> - The GPU **cannot** read memory that the CPU allocated.
>
> The code has a single true/false flag, `UnifiedMemory()`, and uses it to answer both of those questions. On these chips one answer is true and the other is false, so whichever question it gets asked second gets the wrong answer.
>
> The crash is one place where that goes wrong. In `src/vllm/v1/sample/sampler.cpp:328`:
>
> ```cpp
> DeviceScratch t(logits.device, q, temp.data(), vt::DType::kF32, {n});
> vt::ApplyTemperature(q, logits, t.tensor(), sm.all_random);
> ```
>
> `temp` is a list of temperature values living in CPU memory. `DeviceScratch` checks the flag, sees "shared memory, true", and decides it can skip copying the list to the GPU and just hand the GPU a pointer to it. The GPU then tries to read CPU memory, which it cannot do, and the process dies.
>
> Backtrace, with `AMD_SERIALIZE_KERNEL=3` so the fault is attributed to the call that caused it:
>
> ```
> vt::rocm::ApplyTemperatureKernelRocm
> vllm::v1::Sampler::sample
> vllm::v1::Sampler::forward
> vllm::v1::GPUModelRunner::sample_tokens
> vllm::v1::EngineCore::step
> ```
>
> ## Why this has not been noticed until now
>
> Three things had to line up, and they only recently did:
>
> 1. A ROCm sampling kernel has to exist. `src/vt/rocm/rocm_sample.hip` is new. Before it, sampling ran on the CPU, where using a CPU pointer was perfectly legal.
> 2. The GPU has to be an integrated one. Discrete cards answer "false" to the shared-memory question, so they copy the data across properly and are unaffected.
> 3. You have to sample rather than use greedy decoding. Every test run posted on #41 so far, mine included, used `--temperature 0`, which skips this code path completely.
>
> ## Who is affected
>
> - Integrated AMD GPUs. Confirmed on gfx1151 (Strix Halo, Ryzen AI MAX+ 395).
> - gfx1103 is very likely affected for the same architectural reason, but I have not tested it. One command would confirm.
> - Discrete AMD cards are not affected.
> - NVIDIA is not affected, because on those chips both directions genuinely do work.
>
> ## Suggested fix
>
> Split the one flag into two, because there are two different questions:
>
> - `UnifiedMemory()` keeps its current meaning: can the CPU read GPU memory. This is what lets unimplemented operations fall back to CPU code.
> - A new `DeviceCanAccessHostMemory()` answers the other one: can the GPU read CPU memory.
>
> If the new one defaults to returning `UnifiedMemory()`, then every existing backend behaves exactly as it does today and nothing about NVIDIA, CPU, Metal or Vulkan changes. Only the ROCm backend overrides it, answering false on RDNA. `DeviceScratch` then asks the new question, and copies the data across instead of handing over a pointer the GPU cannot use.
>
> The same reasoning applies to weight loading, where the ROCm platform currently reports that it does not need to copy weights to the GPU.
>
> I have this implemented and tested locally and can open a PR.
>
> ## Environment
>
> ```
> GPU:        AMD Radeon 8060S (Ryzen AI MAX+ 395), gfx1151, integrated
> OS:         Arch Linux
> ROCm:       7.2.4, HIP 7.2.53211, AMD Clang 22.0.0
> Host gcc:   16.1.1
> Tree:       main @ 5a0ffe9e
> Build:      cmake -S . -B build-hip -DVLLM_CPP_HIP=ON -DCMAKE_BUILD_TYPE=Release -DROCM_PATH=/opt/rocm
> Model:      Qwen/Qwen3-0.6B (bf16 safetensors)
> ```
>
> Two local changes were needed to get the tree to build on gcc 16, neither related to this crash and neither included in any of the output above:
>
> 1. `-DCMAKE_CXX_FLAGS="-Wno-error=array-bounds"`, for false-positive warnings inside libstdc++ and the vendored JSON library. Same workaround @arch-btw reported on #41.
> 2. Adding `#include <unistd.h>` to five files that call `::getpid` without it, one of which is `src/vllm/entrypoints/openai/server_main.cpp`. I will report that separately.
>
> ## NOTE
>
> This report was written with the help of Claude AI and reviewed by the reporter before submission.
>

## Resolution

-
