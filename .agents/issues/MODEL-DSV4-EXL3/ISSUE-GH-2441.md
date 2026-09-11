ID: ISSUE-GH-2441
Title: DeepSeek-V4-Flash does not load on the default block size (32 cannot page a compress_ratio-128 layer)
Row: MODEL-DSV4-EXL3
State: OPEN
Kind: UNKNOWN
GitHub: 2441
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-31
Updated: 2026-08-31
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `MODEL-DSV4-EXL3`
>
> The DeepSeek-V4-Flash artifact does not load on the engine's default
> configuration. `EngineParams::block_size` is 32, and a `compress_ratio == 128`
> layer cannot be paged at that size: `storage_block_size` is
> `block_size / compress_ratio`, which is 0, so `MakeDeepseekV4KVCache` refuses by
> name rather than publish a pool that cannot hold a token.
>
> The refusal is correct. What is wrong is that nothing raises the block size for
> it. Upstream never asks an operator for this number -- it derives the whole
> geometry at 256 (`vllm/v1/attention/backends/mla/sparse_swa.py:76-83`,
> `vllm/models/deepseek_v4/compressor.py:174-178`). `vllm-server` happens to accept
> `--block-size`; `vllm-cli` does not, so the CLI cannot serve this architecture at
> all.
>
> `AGENTS.md` §"Nothing lands dead" measures reachability on a production entry
> point's DEFAULT configuration, so a model that loads only when an operator
> guesses 256 is not reachable. Adding a flag to the CLI would move the problem
> rather than fix it.
>
> The fix is a model-declared floor that the engine honours before it builds the
> cache, mirroring the fact that upstream derives this from the model.
>
> Found by the load probe recorded in `.agents/specs/model-dsv4-exl3.md`.
>

## Resolution

-
