ID: ISSUE-GH-2445
Title: MODEL-MM-GLM53-FLASH: --device cuda on the real artifact dies in the KV binding, not at the forward's device refusal the spec names
Row: MODEL-MM-glm5-next-glm5-next-for-conditional-generation
State: OPEN
Kind: UNKNOWN
GitHub: 2445
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-31
Updated: 2026-08-31
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `MODEL-MM-glm5-next-glm5-next-for-conditional-generation`
>
> Observed as a by-product of W9c-0's device gate ([#2415](https://github.com/mudler/vllm.cpp/issues/2415), PR #2432), job `79aa5bb5-7536-43fe-a051-ed73ac1302e1` on `dgx:gpu0`, 2026-08-31T23:30Z, on `4034c368c`.
>
> ## This is the first time `--device cuda` has actually been attempted on this artifact
>
> The row has said for several waves that `--device cuda` "still refuses by name at `glm5_next_forward.cpp:231-238`". That statement had never been measured on the real checkpoint. The one prior attempt passed the checkpoint **directory** rather than a shard, so it died in `hf_config` looking for a `config.json` a GGUF artifact does not carry, five seconds in, and captured nothing.
>
> Driven correctly — first shard, `/workspace/ckpt/GLM-5.3-Flash-UD-Q2_K_XL/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf` — it loads, auto-fits the KV cache, enters the engine, and dies **somewhere else entirely**:
>
> ```
> INFO auto-fit max_model_len: reduced from 1048576 to 8192 to fit the KV cache (256 blocks x 32 tokens).
> INFO recurrent-state budget: reduced max_num_seqs from 32 to 1. The KV pool (256 blocks) holds 1 unified
>      pages of 4288 tokens (one page = one 4390912-byte GDN state), and each sequence owns 1 of them.
> vllm.cpp: Asynchronous scheduling is enabled (max_concurrent_batches=2)
> engine-fatal: EngineCore busy loop threw: glm5_next KV binding:
>   'model.layers.3.self_attn.indexer.k_cache' resolved to attn_kv index 45 but only 22 cache(s) arrived;
>   the name index and attn_kv disagree. See .agents/specs/glm5-next-flash.md and issue #2348.
> ```
>
> `CLI_DEVICE_CUDA=1`.
>
> ## What this corrects
>
> **The refusal at `glm5_next_forward.cpp:231-238` is still in the tree and it never fires.** `ResolveAttnCache` (`glm5_next_kv.cpp:127`) throws first, before the forward is entered. So the row's recurring sentence describes code that exists but not behaviour any user can reach, and the device arm's FIRST obstacle is the KV binding rather than the forward's device guard. That matters for W9c-3, which was scoped as "the compose that deletes the refusal at `glm5_next_forward.cpp:231-238`" — it will meet this instead, and earlier.
>
> The name index resolves `model.layers.3.self_attn.indexer.k_cache` to index **45** while `attn_kv` carries **22**. W5b-2c ([#2348](https://github.com/mudler/vllm.cpp/issues/2348)) records 22 as the expected count — 11 latents plus 11 indexer caches — so the name index is producing indices from a larger space than the vector it indexes into.
>
> ## What is NOT established, and is deliberately not guessed
>
> **Whether `--device cpu` reproduces this was not measured.** This lease drove the CUDA arm only. O30's ` Paris.` on `--device cpu` predates several waves and a different resolved config, and the auto-fit above (256 blocks, `max_model_len` 8192, `max_num_seqs` 1) is itself memory-dependent and may differ per device. So this is filed as "observed on `--device cuda`", not as "device-specific". Establishing which requires one `--device cpu` run at the same shard and config, and that is the first thing whoever picks this up should do — a general KV-binding defect and a device-only one have very different owners.
>
> This is also **not** a claim about the k-pool ops PR #2432 lands. Nothing on this path reaches them; they are unreached by construction and the spec's O36 says so.
>
> ## Owed
>
> Not fixed in flow: it is not W9c-0's scope, the fix is in the multi-KV index mapping W5b-2c owns, and choosing between "the name index is wrong" and "the publication is short" needs the `--device cpu` measurement above first. Listed under `## Owed` in `.agents/specs/glm5-next-flash.md`.

## Resolution

-
