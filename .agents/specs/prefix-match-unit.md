# Spec: `--prefix-match-unit` (fine-grained prefix-cache matching unit)

Row: `KV-PREFIX-MATCH-UNIT` (engine-matrix). Claim: `CLAIM-PREFIX-MATCH-UNIT`.
Upstream pin: vLLM `555967922` (0.26.0.dev0).

## Scope

In: port the `prefix_match_unit` CLI/config knob and the exact
`resolve_kv_cache_block_sizes` semantics that turn it into the `hash_block_size`
(a.k.a. matching granularity / "prefix match unit") used by the prefix-cache
block hasher. W1 (this brick) lands the resolver + config field + RED-first unit
tests that pin the `=16`-vs-default behaviour at the resolution site and at the
hasher granularity.

Out (deferred, honestly recorded as later bricks): the full fine-grained
matching path where `hash_block_size < block_size` — i.e. the
`BlockHashListWithBlockSize` coarser-view re-use in `block_pool` /
`single_type_kv_cache_manager` (both already throw "not yet ported" in our tree,
see `KV-BLOCK-POOL` residual "align path where `block_size != hash_block_size`"),
the scheduler threading of a resolved `hash_block_size != block_size`, and the
mamba partial-tail cache-hit stop. Those require the deferred align path and are
NOT owed by W1.

## What `--prefix-match-unit 16` actually does (grounded)

- CLI arg: `vllm/engine/arg_utils.py:696` (`prefix_match_unit = get_field(CacheConfig, "prefix_match_unit")`) and `:1222` (`--prefix-match-unit`, in the cache arg group); forwarded to `CacheConfig` at `:1940`.
- Config field: `vllm/config/cache.py:56` — `prefix_match_unit: int | None = Field(default=None, gt=0)`. Docstring `:57-67`: "The finest token boundary (in tokens) a prefix-cache hit can land on. Prefix-cache keys are computed every `prefix_match_unit` tokens. It can be set finer than the physical KV cache block sizes ... as long as every KV cache group's `block_size` is divisible by it ... controls matching granularity only, not how often states are stored. This equals the `hash_block_size` used throughout the KV cache code." Default `None`.
- Consuming site: `vllm/v1/core/kv_cache_utils.py:626-688` `resolve_kv_cache_block_sizes(kv_cache_config, vllm_config) -> (scheduler_block_size, hash_block_size)`. Exact semantics:
  1. `<= 1` KV group: `bs = cache_config.block_size * dcp`; return `(bs, bs)` (single-group models ignore the knob).
  2. Multi-group: `group_block_sizes[g] = spec.block_size * dcp` if `isinstance(spec, AttentionSpec)` else `spec.block_size`; `scheduler_block_size = lcm(group_block_sizes)`.
  3. If neither `enable_prefix_caching` nor a KV connector is active: return `(scheduler_block_size, scheduler_block_size)` (hashes only consumed by prefix cache / connectors).
  4. If any Mamba group has `block_size != cache_config.block_size` (mamba_cache_mode != "align"): back off, return `(scheduler_block_size, scheduler_block_size)`.
  5. `hash_block_size = prefix_match_unit if set else gcd(group_block_sizes)`.
  6. If any group block size is NOT divisible by `hash_block_size`: `raise ValueError("Invalid prefix_match_unit=... all KV cache group block sizes must be divisible by prefix_match_unit ...")` (`:682-687`).
  7. return `(scheduler_block_size, hash_block_size)`.
- `hash_block_size` feeds `get_request_block_hasher(hash_block_size, ...)` (`:691-748`) — block hashes are computed every `hash_block_size` tokens and chained. A finer unit ⇒ more, finer-grained hashes ⇒ cache hits can land on token boundaries *inside* a physical block. Coarser group block sizes re-use these fine hashes via `BlockHashListWithBlockSize` (`single_type_kv_cache_manager.py:697`).
- Downstream wiring: `vllm/v1/engine/core.py:154` calls the resolver and threads `hash_block_size` into the scheduler (`scheduler.py:76,268-270,282`, `block_pool.hash_block_size`). Scheduler's `mamba_partial_cache_hit` (`scheduler.py:315-318`) fires only when `hash_block_size < block_size`.
- Net effect of `=16` vs default: for a HYBRID/multi-group model (e.g. attention `block_size=16` + mamba block `1024`), default `None` ⇒ `gcd(16,1024)=16`; but if a mamba group diverges the default backs off to `lcm`. Setting `=16` FORCES matching granularity to 16 tokens (finer than a 1024-token physical block), enabling intra-block prefix-cache hits, provided 16 divides every group's block size. For a single-group (dense) model the flag is inert (path 1). It changes matching granularity ONLY, not storage cadence.
- New in 0.26: yes. Not present at the prior `e24d1b24`/0.25.0 pin (the `resolve_kv_cache_block_sizes`/`hash_block_size` split and `prefix_match_unit` are 0.26-era). Related flags: `--block-size`, `--mamba-block-size`, `--mamba-cache-mode` (`align` unlocks the divisibility precondition), `--enable-prefix-caching`.

## Upstream chain (file:line)

- `vllm/engine/arg_utils.py:696,1222,1940`
- `vllm/config/cache.py:56-67,223`
- `vllm/v1/core/kv_cache_utils.py:626-688` (resolver), `:691-748` (hasher)
- `vllm/v1/engine/core.py:154` (call), `vllm/v1/core/sched/scheduler.py:76,268-270,282,312-318`
- `vllm/v1/core/single_type_kv_cache_manager.py:683,697,760-777,1268-1290,1653-1662` (BlockHashListWithBlockSize, fine-grained view — DEFERRED)

## Our baseline

- `hash_block_size` is ALREADY plumbed as a parameter through `get_request_block_hasher(int hash_block_size, ...)` (`src/vllm/v1/core/kv_cache_utils.cpp:577`), `KVCacheManager` / `BlockPool` (`src/vllm/v1/core/block_pool.cpp:43`), and the `Scheduler` ctor (`src/vllm/v1/core/sched/scheduler.cpp:83`) — but the scheduler HARDCODES `hash_block_size = block_size`.
- No `resolve_kv_cache_block_sizes` and no `prefix_match_unit` config field exist yet.
- The fine-grained path (`block_size != hash_block_size`) is explicitly DEFERRED and throws in `src/vllm/v1/core/block_pool.cpp:93,220` (guarded by `include/vllm/v1/core/block_pool.h`). `alignment_tokens` machinery is partly ported in `single_type_kv_cache_manager.cpp`.

## Port map

| Upstream | Local | Notes |
|---|---|---|
| `kv_cache_utils.py:626-688` `resolve_kv_cache_block_sizes` | `include/vllm/v1/core/kv_cache_utils.h` + `src/vllm/v1/core/kv_cache_utils.cpp` (new free function) | Explicit-parameter signature (cache_block_size, prefix_match_unit, enable_prefix_caching, connector_enabled, dcp) instead of reading `VllmConfig`, since our config surface is threaded, not one dataclass. Returns `std::pair<int,int> {scheduler_block_size, hash_block_size}`. |
| `config/cache.py:56` `prefix_match_unit` field | `EngineParams`/`model_loader` field (W2) | Deferred to W2; W1 exercises the resolver directly. |
| scheduler threading of resolved hash_block_size | `scheduler.cpp` + `model_loader.cpp` (W3) | LANDED 2026-09-12 as the DeepSeek-V4 serve fix: `LoadedEngine::ResolveSchedulerBlockSizes` derives `min(group block_size)` then calls the resolver (`engine/core.py:335-338`, `:158-170`), and the pair rides through `MakeScheduler` into the `Scheduler` / `AsyncScheduler` ctors. Only the case where `hash_block_size` is STRICTLY FINER than a group's block size still needs the block_pool align path. |

## Tests to port

- Upstream has NO direct unit test for `resolve_kv_cache_block_sizes` (it is exercised indirectly by `tests/v1/core/prefix_cache/test_partial_prefix_cache_primitives.py`, which needs the deferred `BlockHashListWithBlockSize` path). W1 therefore ports the SEMANTICS as hand cases in `tests/vllm/v1/test_prefix_match_unit.cpp`, each traceably citing `kv_cache_utils.py:626-688`: single-group inert; multi-group default = gcd; `=16` override; `=32` finer-than-1024-block; non-divisible → throw; mamba-non-align back-off; prefix-off+no-connector back-off; plus a hasher-granularity RED case proving unit-16 vs default produce different hash counts via `get_request_block_hasher`.
- The full `test_partial_prefix_cache_primitives.py` port is checked in SKIPPED-with-reason under the deferred align path (tracked in `KV-BLOCK-POOL`).

## Gates

- W1: CPU `-Werror` build clean; `test_prefix_match_unit` green; RED-first proven (default ≠ `=16`). No engine behaviour change on the default path (single-group inert; scheduler still passes `block_size`). SUPERSEDED for the second sentence on 2026-09-12: the engine now derives the pair and the scheduler is threaded (see the port map). The dense default is still byte-identical, because one group at the configured block size resolves to the pair the scheduler was already given.
- Benchmark: a real matching-unit throughput/hit-rate A/B is a later brick — `docs/BENCHMARKS.md` PENDING.

## W-breakdown

- W0: this spec + records (records-only commit).
- W1: `resolve_kv_cache_block_sizes` + tests (this brick).
- W2: `prefix_match_unit` config/CLI/ABI field + engine-core call site.
- W3: scheduler threading of `hash_block_size != block_size` (needs block_pool align path from `KV-BLOCK-POOL`) + mamba partial-tail stop; port `test_partial_prefix_cache_primitives.py` un-skipped. THE THREADING HALF LANDED 2026-09-12 (DeepSeek-V4 serve fix): the engine derives `(scheduler_block_size, hash_block_size)` from the built groups and passes both to the scheduler. What remains is the FINER-THAN-A-GROUP case, which is the part the align path gates, plus the mamba partial-tail stop and the un-skipped port.
- W4: benchmark the matching-unit effect (hybrid model, hit-rate + throughput A/B vs vLLM).
