# Spike: prefix-caching modes and coordinator selection

**Row:** `KV-PREFIX-CACHE` · **state:** accepted; implementation remains
`PARTIAL`.

## Scope

This spike owns the native V1 automatic-prefix-caching control plane: model
capability/default resolution, explicit enable/disable override, request block
hashes, cached-block lookup/allocation/eviction, and the unitary, hybrid, and
no-prefix coordinator choices. It covers text-generation KV groups already in
the tree (full attention, GDN/mamba, sliding-window, and chunked-local).

The current checkpoint ports the no-prefix coordinator and makes hybrid or
attention-free models default to it, matching vLLM. Partial-block caching,
cache salts, retention intervals, KV connectors, cross-attention, distributed
context parallelism, metrics/events breadth, and model-positive APC performance
remain open. This checkpoint does not claim APC breadth or mark the row `DONE`.

## Upstream chain

- Default support is decided by `vllm/config/model.py:1805-1854`; generative
  hybrid and attention-free models return unsupported. Tri-state CLI/default
  resolution is `vllm/engine/arg_utils.py:2470-2510`, with
  `--[no-]enable-prefix-caching` constructed at `:1158-1165`.
- `vllm/v1/core/kv_cache_manager.py:103-236,420-456` builds the coordinator,
  returns no computed blocks when caching is off, and gates cache writes.
- `vllm/v1/core/kv_cache_coordinator.py:94-375` owns common allocation/free
  fanout. `KVCacheCoordinatorNoPrefixCache` at `:377-425` supports any group
  count and returns an empty block tuple plus zero hit/common-prefix length.
  Unitary and hybrid cache-hit selection follow at `:428-779`; factory dispatch
  is `:782-855`.
- `vllm/v1/core/single_type_kv_cache_manager.py:31-82,191-231` and
  `block_pool.py:141-190,548-568` propagate `enable_caching` into manager and
  pool behavior. Request hashes originate in `vllm/v1/core/kv_cache_utils.py`.
- Runtime proof must trace the server-selected cache policy because reused
  prompts can otherwise erase prefill work while leaving logical input counts
  unchanged. The online gate therefore records explicit policy, closed-loop
  admission, model length, max sequences, and token budget on both engines.

## Our baseline

- Hashes, pool, single-type managers, unitary/hybrid coordination, and eviction
  exist in `src/vllm/v1/core/{kv_cache_utils,block_pool,
  single_type_kv_cache_manager,kv_cache_coordinator,kv_cache_manager}.cpp`.
- Before this checkpoint, `get_kv_cache_coordinator` threw when caching was
  false and `LoadedEngine` always enabled caching. That contradicted the pinned
  hybrid-model default and contaminated repeated-prompt traces.
- The standard online corpus is globally disjoint and shares fewer than one
  32-token block, so the old standard grid is not a cache-performance result;
  its warmup/first-request and repeated trace policy were nevertheless not
  strictly equivalent to vLLM and remain non-binding.
- Partial-prefix primitives and broader APC semantics are incomplete; the row
  stays `PARTIAL`.

## Port map

| Upstream | Local | Disposition |
|---|---|---|
| `config/model.py:1805-1854`; `engine/arg_utils.py:2470-2510` | `LoadedEngine::ResolveEnablePrefixCaching`, `EngineParams`, server CLI | model-capability default plus explicit tri-state override |
| `kv_cache_coordinator.py:377-425,782-855` | `KVCacheCoordinatorNoPrefixCache` and `get_kv_cache_coordinator` | direct port; arbitrary group count, no hits/common prefix |
| `kv_cache_manager.py:118-148` | existing `KVCacheManager` constructor | now receives the resolved policy instead of a hard-coded true |
| `kv_cache_utils.py` request hasher selection | `LoadedEngine::block_hasher_` | null when caching is disabled; SHA-256/CBOR when enabled |
| vLLM server/LLM config | `scripts/dgx-online-serving.sh`, `profile_vllm_online_gate.py` | both arms explicitly cache-off for hybrid gate traces/runs |

Deviation: vLLM derives the model attention type through Python model metadata;
our registry already exposes `ModelInfo.is_hybrid` and `has_inner_state`, so the
same decision is made without importing a model class.

## Tests to port

| Upstream test | Local tier / evidence |
|---|---|
| `tests/v1/core/test_prefix_caching.py:1483` and `tests/v1/core/test_single_type_kv_cache_manager.py:511` cache-off allocation | `tests/vllm/v1/test_kv_cache_coordinator.cpp`: hybrid no-prefix type, empty hits/zero common prefixes, normal allocate/free |
| `tests/v1/core/test_scheduler.py:502` caching on/off scheduler parameterization | existing scheduler/KV suites plus loaded-engine cache-default test; remaining scheduler matrix is open |
| `tests/v1/e2e/test_hybrid_chunked_prefill.py:56` hybrid cache-off/on e2e | both real Qwen paged-engine gates with cache off/on; GPU handoff open |
| `tests/benchmarks/sweep/test_param_sweep.py:37,90` Boolean optional flag | server help/parse contract and online command/status validators |
| repeated-prompt runtime behavior | corrected paired 27B then 35B trace: three stable cache-off repetitions on each side |

## Gates

1. CPU: clean warnings-as-errors build and full CTest; focused coordinator,
   loaded-engine, server-help, and online-tool tests pass.
2. Correctness: both real-model greedy gates pass with default cache-off and an
   explicit cache-on A/B. Default off must match the pinned hybrid behavior.
3. Trace: same prompt IDs, cache off, production max model length, max-seqs and
   token budget, closed-loop concurrency, three repetitions within 20% duration;
   every artifact and selected policy is hashed.
4. Performance: cache-off is the binding parity configuration. Cache-on is a
   separate opt-in feature A/B and may not supply an oracle denominator when
   upstream disables it.
5. Memory/lifecycle: no-prefix allocation/free returns GPU and host memory under
   the online gate's existing process/memory-return contract.
6. Backends: CPU behavior is mandatory; GB10/sm_121a is the runtime gate. Other
   backends inherit `KV-PREFIX-CACHE`'s existing open coverage.

## Dependencies

- Rows: `SERVE-GATE-ONLINE` consumes the cache-off default;
  `ENG-ASYNC-SCHED` consumes placeholder-aware cache lengths separately.
- Existing `KVCacheManager`, `BlockPool`, registry metadata, server CLI, and
  exact online corpus are required. No new library or data dependency.
- DGX GB10 is required only for real-model correctness/trace/performance gates.

## Work breakdown

| Work | Deliverable | State |
|---|---|---|
| W0 | no-prefix coordinator, model-default/CLI resolution, cache-off online contract and CPU tests | implemented; GPU gating open |
| W1 | cache salt/skip-reading and partial-block semantics/tests | open |
| W2 | retention interval, connector/event/metrics surfaces | open |
| W3 | model-positive APC on supported decoder families and cache-on performance/memory gates | blocked on a supported non-hybrid model family |
| W4 | distributed/context-parallel and cross-attention breadth | open |

## Risks and decisions

- Logical input-token counters do not reveal prefix hits. Runtime policy plus
  stable repeated durations are mandatory trace metadata.
- Cache-on for hybrid models is an experimental opt-in divergence; it cannot be
  used to beat a cache-off vLLM denominator. The default remains vLLM's answer.
- Async placeholders subtract from cacheable computed length. That behavior is
  owned by `ENG-ASYNC-SCHED` and must not be approximated in this leaf.
- Disabling hashing must not disable ordinary allocation or recurrent-state
  lifetime. The no-prefix class reuses the base fanout and overrides only hit
  and common-prefix reporting.
