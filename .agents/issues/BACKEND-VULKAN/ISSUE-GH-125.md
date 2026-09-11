ID: ISSUE-GH-125
Title: Vulkan on AMD Strix Halo (gfx1151) does not load
Row: BACKEND-VULKAN
State: OPEN
Kind: bug
GitHub: 125
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-07
Updated: 2026-08-29
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `BACKEND-VULKAN`
>
> Tried FP8:
>
> ```
> $ ./vllm.cpp/build/examples/vllm-cli --model ./models/Qwen3.6-27B-FP8 --prompt "Draw an SVG image of a skateboarding cat"
> 2>&1 | tee cat.svg
> vllm-cli: loading model from ./models/Qwen3.6-27B-FP8
> vllm-cli: model load failed (status 2): vllm_engine_load: vt: dense loader: expected BF16 for model.language_model.layers.
> 0.linear_attn.in_proj_qkv.weight at /home/brent/vllm.cpp/include/vllm/model_executor/models/dense_weight_loaders.h:109
> ```
>
> Then BF16:
>
> ```
> $ ./vllm.cpp/build/examples/vllm-cli --model ./models/Qwen3.6-27B --prompt "Draw an SVG image of a skateboarding cat" 2>&1
>  | tee cat.svg
> vllm-cli: loading model from ./models/Qwen3.6-27B
> vllm.cpp: Asynchronous scheduling is enabled (max_concurrent_batches=2)
> engine-fatal: EngineCore busy loop threw: vt: vulkan: embedding: table points outside every Vulkan allocation — Vulkan ker
> nels can only bind memory obtained from vt::GetBackend(DeviceType::kVULKAN).Alloc() at /home/brent/vllm.cpp/src/vt/vulkan/
> vulkan_backend.cpp:144
> async-llm: output handler saw engine death: EngineCore encountered an issue. See stack trace (above) for the root cause. [
> vt: vulkan: embedding: table points outside every Vulkan allocation — Vulkan kernels can only bind memory obtained from vt
> ::GetBackend(DeviceType::kVULKAN).Alloc() at /home/brent/vllm.cpp/src/vt/vulkan/vulkan_backend.cpp:144]
> vllm-cli: completion failed (status 3): vllm_complete: EngineCore encountered an issue. See stack trace (above) for the ro
> ot cause. [vt: vulkan: embedding: table points outside every Vulkan allocation — Vulkan kernels can only bind memory obtai
> ned from vt::GetBackend(DeviceType::kVULKAN).Alloc() at /home/brent/vllm.cpp/src/vt/vulkan/vulkan_backend.cpp:144]
> ```
>
> I am on Slackware-current. I have the Vulkan libraries (`vulkan-sdk-1.4.341.1-x86_64-2 `) and am able to use `llama.cpp`'s Vulkan backend.
>
> ```
> $ ls /usr/lib64/libvulkan*
> /usr/lib64/libvulkan.so          /usr/lib64/libvulkan_intel.so        /usr/lib64/libvulkan_nouveau.so
> /usr/lib64/libvulkan.so.1        /usr/lib64/libvulkan_intel_hasvk.so  /usr/lib64/libvulkan_radeon.so
> /usr/lib64/libvulkan.so.1.4.341  /usr/lib64/libvulkan_lvp.so          /usr/lib64/libvulkan_virtio.so
> ```
>
> I usually run `llama.cpp` like this:
>
> ```
> GGML_VK_PREFER_HOST_MEMORY=1 AMD_VULKAN_ICD=RADV ~/llama.cpp/build/bin/llama-server \
>   --host 0.0.0.0 \
>   --no-ui \
>   --metrics \
>   --no-models-autoload \
>   --models-max 1 \
>   --models-preset $HOME/resources/presets.ini 2>&1 | tee $HOME/local/var/log/llama.cpp.log
> ```
>
> And load Qwen3.6 with these presets, for example:
>
> ```
> [11]
> hf = unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q4_K_M
> host = 0.0.0.0
> c = 262144
> spec-type = draft-mtp
> spec-draft-n-max  = 2
> chat-template-file = /home/brent/resources/froggeric_tpl/chat_template.jinja
> chat-template-kwargs = {"preserve_thinking":true}
> np = 1
> ctk = q4_0
> ctv = q4_0
> ctkd = q4_0
> ctvd = q4_0
> temp = 1.0
> top-k = 20
> top-p = 0.95
> min-p = 0
> ```
>
> ```
> $ uname -a
> Linux beelink.home 7.1.2 #1 SMP PREEMPT_DYNAMIC Sat Jun 27 17:46:51 CDT 2026 x86_64 AMD RYZEN AI MAX+ 395 w/ Radeon 8060S AuthenticAMD GNU/Linux
> ```
>
> Cheers

## Resolution

-
