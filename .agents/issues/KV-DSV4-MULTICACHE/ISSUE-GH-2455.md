ID: ISSUE-GH-2455
Title: DeepSeek-V4 cannot load on the default configuration: its own factory publishes fp8_ds_mla and ApplyCacheDType refuses it (no store/read, owed to W5)
Row: KV-DSV4-MULTICACHE
State: OPEN
Kind: bug
GitHub: 2455
Mirror: DIVERGED
Availability: FULL
Created: 2026-09-01
Updated: 2026-09-11
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `KV-DSV4-MULTICACHE`
>
> MEASURED on the real 97.68 GiB `nvidia/DeepSeek-V4-Flash` EXL3 artifact under an
> `rc` lease on `thor:gpu0`, 2026-09-01, source `693f17e08`, through `vllm-cli`
> with NO flags:
>
> ```
> [vt load] dsv4-exl3: coalesced TP1 tower resident_bytes=87994957824 (81.952 GiB)
>   over 43 layers, tp4->tp1, 3-bit trellis; carried host tower
>   host_bytes=16885558876 (15.726 GiB); host MemAvailable=101.407 GiB
> vllm-cli: model load failed (status 2): vllm_engine_load: vt: cache_dtype: an MLA
>   KV cache has its own quantized page formula upstream (fp8_ds_mla,
>   kv_cache_interface.py:398-410). W1 landed that page formula but no fp8_ds_mla
>   store or read, so requesting it here would size the page for bytes nothing
>   writes; run the MLA model on --kv-cache-dtype auto
> ```
>
> **Nobody requested it.** `vllm-cli` has no `--kv-cache-dtype` flag and none was
> passed. The refusal fires on the DEFAULT path, and the message misattributes the
> cause to an operator request.
>
> The mechanism is a contradiction between two correct-looking pieces:
>
> - `MakeDeepseekV4KVCache` publishes an **fp8_ds_mla** topology on purpose,
>   mirroring upstream, where `use_fp8_ds_mla_layout` is `ClassVar[bool] = True`
>   and `_resolve_dsv4_kv_cache_dtype` writes `cache_dtype = "fp8_ds_mla"` back
>   onto the cache config (`attention.py:89-119, 140`). Its specs carry
>   `vt::DType::kI8`.
> - `ApplyCacheDType`'s early-out needs `spec.dtype == resolved.storage`
>   (`kv_cache_interface.cpp:432-435`). On `auto`, `resolved.storage` is the model
>   dtype (bf16), the spec's is `kI8`, so the early-out misses and
>   `RetypeAttentionSpec` refuses every `MLAAttentionSpec`.
>
> So the model's own factory declares a layout the retype path then refuses, and
> the operator has no way to ask for anything different.
>
> **The refusal is CORRECT and must not be widened to make this load.** W1 landed
> the fp8_ds_mla page formula and neither the store nor the read
> (`kv_cache_interface.cpp:377-388` says so). Accepting it would size every MLA
> page at 584 bytes per token while the attention block still writes a bf16 latent
> into it -- wrong tokens, not a crash. Making this green by relaxing the guard is
> exactly the "never make a red gate green by widening its scope" case.
>
> What is owed is the **store and read side**, which `kv_cache_interface.cpp:388`
> already names as W5's.
>
> Two things worth separating for whoever takes it:
>
> 1. The message should stop saying "requesting it here". On this path nothing was
>    requested, and the next reader will look for a flag they never passed.
> 2. The load-blocking question is whether DeepSeek-V4 can serve at all before W5.
>    If the plain 512B-aligned bf16 MLA arm is servable, publishing it is the
>    smaller path to a first token; that arm is currently NOT published and is
>    recorded as owed to W5 in `deepseek_v4_registry.cpp:210-211`.
>
> Found by the default-configuration load probe for #2441, which this now
> supersedes as the load blocker: #2441's `block_size` refusal is GONE from this
> run (0 occurrences), and the load proceeds past the KV geometry to here.
>

## Resolution

-
