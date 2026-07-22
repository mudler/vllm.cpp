# Spike: prompt / prefix caching — full vLLM parity, and beyond

**Rows owned:** `KV-PREFIX-CACHE`, `KV-BLOCK-POOL`, `KV-HYBRID-COORD`,
`KV-MAMBA-ALIGN`, `KV-EVENTS`, `ENG-CASCADE-ATTN`.
**Claim:** `CLAIM-PREFIX-PROMPT-CACHING`. **SPIKE ONLY — no implementation.**

Cross-referenced, NOT owned: `KV-OFFLOAD`, `KV-EXTERNAL-CACHE`, `KV-CONNECTORS`
(`ROAD-V1-D4`), `SERVE-METRICS`, `SERVE-UTILITY-ENDPOINTS`,
`BACKEND-GATE-CUDA-SGLANG-PREFIX`.

This spike answers the user's question — *"do we support prompt caching, like
llama.cpp does? what about vLLM? is it in roadmap_v1?"* — under the binding bar
the user then set: **the same featureset as vLLM, and better.** It supersedes
nothing in [prefix-caching.md](prefix-caching.md), which remains the accepted
leaf spike for the narrower cache-policy/coordinator-selection question; this
document is the umbrella parity surface over the whole caching feature.

### Scope

The complete prompt/prefix-caching feature surface, enumerated against pinned
vLLM `e24d1b24`, with a DONE / PARTIAL / MISSING verdict per feature grounded in
our source rather than in a matrix cell. It covers: automatic prefix caching
(APC) block hashing and its extra keys, the block pool and its eviction, the
cache-hit coordinators, the per-spec managers (full attention, sliding window,
chunked-local, Mamba/GDN), prefix-cache-aware scheduling and its chunked-prefill
interaction, the config/CLI/API surface, hit-rate metrics, KV-cache events,
`reset_prefix_cache`, cascade attention, and the boundary with external/offloaded
KV (`ROAD-V1-D4`). It also covers llama.cpp's `--prompt-cache` / slot
save-restore as a *competitor capability*, and names the concrete axes on which
we can exceed vLLM, each with how it would be MEASURED.

Out of scope for this spike's rows (cross-referenced only): the external KV
connector ABI and its tiers, the Prometheus endpoint itself, and the utility
endpoint router — those are owned by `KV-EXTERNAL-CACHE` / `KV-CONNECTORS` /
`KV-OFFLOAD`, `SERVE-METRICS` and `SERVE-UTILITY-ENDPOINTS` respectively. This
spike states what each of them owes the caching feature, and nothing more.

**Headline answer to the user.** We DO have prefix caching, and it is deeper
than the record claimed — the block pool, the chain hashing, all three
coordinators, the full hybrid cross-group intersection and four of five
single-type managers are ported and tested. The honest gaps are narrower and
different from what the matrix implied: block-hash **extra keys are a stub**,
there are **no hit-rate statistics at all**, KV-cache **events are inert**,
**partial-block** caching throws, `cache_salt` does not exist, `reset_prefix_cache`
is implemented but **unreachable** (no endpoint), and only ONE of vLLM's four
hash algorithms is shipped. Two record claims were wrong in our favour and one
against us; all three are corrected here. See §"Our baseline".

### Upstream chain

vLLM pin `e24d1b24` (verified: `git -C /home/mudler/_git/vllm log --oneline -1`).
The caching core is ~6.0k lines of Python: `kv_cache_utils.py` 2229,
`single_type_kv_cache_manager.py` 1560, `kv_cache_coordinator.py` 834,
`block_pool.py` 723, `kv_cache_manager.py` 622.

#### The complete enumerated surface, with our status

Status key: **DONE** = ported with tests / **PARTIAL** = ported with a named
omission / **MISSING** = absent / **N-A** = verified not owed on the path we
mirror.

| # | Feature | Upstream `file:line` | Ours | Status |
|---:|---|---|---|---|
| 1 | `enable_prefix_caching`, default `True` | `vllm/config/cache.py:93` | `model_loader.cpp:128-138` explicit-wins, else `!is_hybrid && !has_inner_state` | **DONE** |
| 2 | Per-model default (`is_prefix_caching_supported`; hybrid + attention-free OFF) | `vllm/config/model.py:1805-1854` | same predicate over `ModelInfo` `model_loader.cpp:134-137` | **DONE** |
| 3 | Tri-state `--[no-]enable-prefix-caching` | `vllm/engine/arg_utils.py:1158-1165,2470-2510` | `examples/server/main.cpp:158-163`, duplicate-rejecting | **DONE** |
| 4 | `prefix_caching_hash_algo` — FOUR algos, default `sha256` | `vllm/config/cache.py:39,95`; `vllm/utils/hashing.py:26,43,70,76` | only `sha256_cbor`, hardwired `model_loader.cpp:145`; no CLI flag | **PARTIAL** — 3 of 4 absent, and our default deviates from upstream's |
| 5 | `init_none_hash` / `NONE_HASH` chain seed | `vllm/v1/core/kv_cache_utils.py:99-114` (`os.urandom(32)` unless `PYTHONHASHSEED`) | `kv_cache_utils.cpp:305,307`; called UNSEEDED at `model_loader.cpp:140-146` | **DONE — FIXED 2026-07-22 (see the note below).** This cell previously read "DONE (and deterministic)" and that was FALSE: the unseeded branch fills `NONE_HASH` from `std::random_device` (`kv_cache_utils.cpp:312-318`) and the only production caller passes no seed (`model_loader.cpp:144`), so block hashes differ across processes exactly as upstream's do — and we lack upstream's `PYTHONHASHSEED` escape hatch and its unset-warning. See [kv-persistence-lmcache.md](kv-persistence-lmcache.md) §Our baseline Correction 1. **FIXED in the same era:** `init_none_hash` now resolves explicit > `$VLLM_PREFIX_CACHING_HASH_SEED` > `$PYTHONHASHSEED` > a fixed default, making us deterministic across processes by DEFAULT where upstream is random-by-default; `tests/vllm/v1/test_none_hash_determinism.cpp` proves it across separate process invocations |
| 6 | `hash_block_tokens` (parent-chained, falls back to `NONE_HASH` on falsy parent) | `vllm/v1/core/kv_cache_utils.py:577-604` | `kv_cache_utils.cpp:331` | **DONE** |
| 7 | `hash_request_tokens` | **REMOVED at this pin** — superseded by the closure factory below | `kv_cache_utils.cpp:363` still ports it | **STALE** — we mirror a function upstream deleted (see §Risks R6) |
| 8 | `get_request_block_hasher` closure factory — incremental, resumes at `len(block_hashes) * hash_block_size`, trailing partial block never hashed | `vllm/v1/core/kv_cache_utils.py:673-730,688,709-711` | `kv_cache_utils.cpp:392`; `request.cpp:57,66,72,127,134,139-143` | **DONE** |
| 9 | `BlockHash` / `BlockHashWithGroupId` — `NewType` over `bytes`; group id is a **4-byte big-endian concat**, not a tuple | `vllm/v1/core/kv_cache_utils.py:44,49,57-66,69-76` | `kv_cache_utils.h:88,93`; packing `:279-301` | **DONE** |
| 9b | `hash_block_size` decoupled from physical `block_size` + `resolve_kv_cache_block_sizes` (collapses when neither APC nor a connector is live, or on non-`align` Mamba) | `vllm/config/cache.py:56`; `vllm/v1/core/kv_cache_utils.py:607-670,2156-2226` | pool carries `hash_block_size`; the differing-size path throws (#20) | **PARTIAL** |
| 9c | `skip_reading_prefix_cache` per-request read bypass (defaults true when `prompt_logprobs` is set) | `vllm/sampling_params.py:343,500-504`; `vllm/v1/request.py:186,266-276`; enforced `kv_cache_manager.py:218` | absent | **MISSING** |
| 10 | **`generate_block_hash_extra_keys`** — fixed order **lora → mm → cache_salt → prompt_embeds** | `vllm/v1/core/kv_cache_utils.py:539-574,567-569`; mm `:431-495` (emits the *relative* in-block offset `:482`), LoRA `:498-510` (adapter **name**), salt `:560-562` (block 0 only), embeds `:513-536` (always sha256) | **hardcoded no-op** `kv_cache_utils.cpp:323-329`; the encode path `:347-356` is correct and unused | **MISSING** — the single most important gap (see §Risks R1) |
| 11 | `cache_salt` per request (API → Request → hash) | `vllm/v1/request.py:154`; API `openai/chat_completion/protocol.py:422`, `responses/protocol.py:235` | absent; noted unported `request.h:17`, `types.h:17`, `protocol.h:32` | **MISSING** |
| 12 | BlockPool free list + LRU (`FreeKVCacheBlockQueue`) | `vllm/v1/core/block_pool.py` | `kv_cache_utils.cpp:463-645` intrusive queue | **DONE** |
| 13 | Refcounts, `touch`, `free_blocks` unhashed-front/hashed-tail split | `vllm/v1/core/block_pool.py:559-575,614-627` | `block_pool.cpp:243,254-275` | **DONE** |
| 14 | `cached_block_hash_to_block`, `get_cached_block` (group-aware) | `vllm/v1/core/block_pool.py:185,199-224` | `block_pool.h:212,220`; `block_pool.cpp:60` | **DONE** |
| 15 | `cache_full_blocks`, `_maybe_evict_cached_block` | `vllm/v1/core/block_pool.py:144,226` | `block_pool.cpp:77,231,145,184` | **DONE** |
| 16 | `reset_prefix_cache`, `get_usage`, null block | `vllm/v1/core/block_pool.py:542,614` | `block_pool.cpp:277,301,297,56-57` | **DONE** (core); **unreachable** — see #31 |
| 17 | **Partial-block caching** (`cache_partial_block`, boundary hashes) | `vllm/v1/core/block_pool.py:358-457` — **DEAD CODE at this pin: a repo-wide grep of `vllm/` finds no caller; only tests exercise it** | throws `block_pool.cpp:131-137`; partial→full promotion throws `:117-124` | **MISSING, and NOT owed as production behaviour** — see §Risks R7 |
| 18 | **KV-cache events** BlockStored / BlockRemoved / AllBlocksCleared | `vllm/config/kv_events.py:11-52`; `vllm/v1/core/kv_cache_manager.py:121,149` | placeholder struct `block_pool.h:83`; `take_events()` always empty `:311`; emissions omitted `block_pool.cpp:128,239,293` | **MISSING** (`KV-EVENTS`) |
| 19 | `evict_blocks` (connector-driven) | `vllm/v1/core/block_pool.py` | throws `block_pool.cpp:139-143` | **MISSING** |
| 20 | Align path `block_size != hash_block_size` | `vllm/v1/core/block_pool.py` (`BlockHashListWithBlockSize`) | throws `block_pool.cpp:93-97` | **MISSING** |
| 21 | `KVCacheManager.get_computed_blocks` + "recompute at least one token" | `vllm/v1/core/kv_cache_manager.py:221` | `kv_cache_manager.cpp:124`, rule at `:134` | **DONE** |
| 22 | `allocate_slots` (watermark, OOM→None, admission cap, `delay_cache_blocks`) | `vllm/v1/core/kv_cache_manager.py:110,244` | `kv_cache_manager.cpp:144,170-189,198,214,230,237` | **DONE** |
| 23 | **`PrefixCacheStats`** — `queries`/`hits` count **tokens**, plus mutually-exclusive `preempted_*` counters; aggregated by `CachingMetrics` over a 1000-**request** sliding window | `vllm/v1/metrics/stats.py:115,122-142,35-111,107-111`; take-and-swap `kv_cache_manager.py:190-200` | ported `include/vllm/v1/metrics/stats.h`; recorded `src/vllm/v1/core/kv_cache_manager.cpp:139-147`; reset flag `:270-276`; per-step aggregation `src/vllm/v1/core/sched/scheduler.cpp` (end of `schedule()`); accessors on Scheduler/EngineCore/LLMEngine | **DONE** |
| 24 | `KVCacheCoordinatorNoPrefixCache` / `Unitary` / `Hybrid` + factory | `vllm/v1/core/kv_cache_coordinator.py:377-425,428-779,782-855` | `kv_cache_coordinator.cpp:260,289,330,539` | **DONE** |
| 25 | Hybrid cross-group intersection (fixed-point loop, EAGLE bookkeeping, truncation) | `vllm/v1/core/kv_cache_coordinator.py:514,630` | `kv_cache_coordinator.cpp:424-537,281-323,500,507-521` | **DONE** — record was pessimistic |
| 26 | `FullAttentionManager` / `SlidingWindowManager` / `ChunkedLocalAttentionManager` `find_longest_cache_hit` + `remove_skipped_blocks` | `vllm/v1/core/single_type_kv_cache_manager.py:564,669,876`; SWA `reachable_block_mask` `:774-836` | `single_type_kv_cache_manager.cpp:288,377,553` + `:262,470,521,618` | **DONE** |
| 26b | **Seven** single-type subclasses upstream, dispatched by a spec→manager REGISTRY (11 registrations), not a static map | `vllm/v1/core/single_type_kv_cache_manager.py:1455-1496,1499-1560`; `RSWAManager:625`, `SinkFullAttentionManager:1431` | we implement 4 and dispatch statically; `RSWA` / `SinkFullAttention` / `CrossAttention` absent | **PARTIAL** |
| 27 | `MambaManager` + **`mamba_cache_mode`, THREE values `all`/`align`/`none`** | manager `vllm/v1/core/single_type_kv_cache_manager.py:1026,1196-1204,1265-1327,1341,1364`; config `vllm/config/cache.py:38,134`; **resolution** `vllm/model_executor/models/config.py:550-593` (APC-on + `none` → `all` if supported else `align`; `all` on an unsupported model downgrades to `align`; `align` **asserts chunked prefill** `:571-573`; APC-off forces `none`); Qwen raises on `all` `vllm/model_executor/models/qwen3_5.py:297` | allocator half ported: `single_type_kv_cache_manager.cpp:639,687-700,737,773,803,865`; **no config selects align**, `MambaSpec` defaults `"none"` `kv_cache_interface.h:306`; **no runner-side state copy** (zero hits in `src/vllm/v1/worker/gpu/`) | **PARTIAL** — record said `INVENTORIED`, which is WRONG (see §Our baseline) |
| 28 | `CrossAttentionManager` — the only manager that **refuses** prefix caching | `vllm/v1/core/single_type_kv_cache_manager.py:1368,1399,1428` | absent | **MISSING** (T2, `KV-CROSS-ENCODER-SPECS`) |
| 29 | Prefix-cache-aware scheduling + chunked prefill + preemption reuse | `vllm/v1/core/sched/scheduler.py` | `scheduler.cpp:263-270,284-297,629,102,184-233,253`; async `async_scheduler.cpp:67-73` | **DONE** |
| 30 | `num_common_prefix_blocks` on SchedulerOutput | `vllm/v1/core/sched/scheduler.py` | `scheduler.cpp:330-335,381`; `sched/output.h:151` | **DONE** (computed; consumed by nothing — see #32) |
| 31 | `POST /reset_prefix_cache` (`reset_running_requests`, `reset_external`) — **only mounted when `VLLM_SERVER_DEV_MODE=1`**; siblings `/reset_mm_cache`, `/reset_encoder_cache` | `vllm/entrypoints/serve/dev/cache/api_router.py:20-43`; mount gate `vllm/entrypoints/openai/api_server.py:198`; `vllm/entrypoints/llm.py:800`; engine `vllm/v1/core/sched/scheduler.py:2185-2233` | engine-level exists (#16); **no endpoint** — our server exposes 5 routes `api_server.cpp:311-327` | **MISSING**, low priority — dev-mode surface (`SERVE-UTILITY-ENDPOINTS`) |
| 32 | **Cascade attention** | `vllm/config/model.py:238` **`disable_cascade_attn: bool = True`**; impl `vllm/v1/worker/gpu_model_runner.py:504,2544` | absent; only the common-prefix input plumbing (#30) | **N-A — no port owed**, see below |
| 33 | Prometheus `vllm:prefix_cache_queries` / `vllm:prefix_cache_hits` / `vllm:external_prefix_cache_*` | `vllm/v1/metrics/loggers.py:547,558,571` | **no `/metrics` endpoint at all** | **MISSING** (`SERVE-METRICS` + #23) |
| 34 | Multimodal input hashing into block keys — hasher default is **blake3** (`VLLM_MM_HASHER_ALGORITHM`), the mm hash is `MultiModalFeatureSpec.identifier` | `vllm/multimodal/hasher.py:23,37-40,154-162`; `vllm/multimodal/inputs.py:302,322`; `kv_cache_utils.py:431-495` | blocked behind #10; no multimodal models yet | **MISSING** (latent) |
| 34b | Two caching layers ADJACENT to APC, often confused with it: the **MM processor cache** (caches HF-processor output tensors) and the **encoder cache** (caches encoder outputs) | `vllm/multimodal/cache.py:262,326,379,437`; `vllm/config/multimodal.py:123,132`; `vllm/v1/core/encoder_cache_manager.py:17,94,184,269` | absent | **MISSING** — distinct rows, not KV caching; named so they are not double-counted as APC |
| 35 | KV offload tiers: CPU (LRU/ARC) + `fs` / `p2p` / `obj` secondary | `vllm/v1/kv_offload/tiering/factory.py:55-79`; `file_mapper.py:112-121` | absent (`grep -i offload` over `src/`+`include/` = nothing) | **MISSING** (`KV-OFFLOAD`) |
| 36 | External KV connectors (LMCache, Example, NIXL, Multi, Offloading) + `get_num_new_matched_tokens` | `vllm/distributed/kv_transfer/`; V2 runner hook `vllm/v1/worker/gpu/kv_connector.py` | absent; deferral comments `types.h:26`, `sched/output.h:30` | **MISSING** (`KV-EXTERNAL-CACHE`, `ROAD-V1-D4`) |
| 37 | Retention interval (sparse local/SWA retention) | `vllm/v1/core/kv_cache_coordinator.py:124` | env read deferred `kv_cache_coordinator.h:70`; SWA/mamba paths honour it `single_type_kv_cache_manager.cpp:489-511,689-700` | **PARTIAL** |
| 38 | DCP/PCP > 1 and differing per-group block sizes | `vllm/v1/core/kv_cache_coordinator.py` | assert-guarded `kv_cache_coordinator.cpp:344,348-349` | **MISSING** (T2) |

#### Cascade attention — the disposition, and why it is not a gap

`ENG-CASCADE-ATTN` has been carried as an unspiked `INVENTORIED` port. It is
**not owed on the path we mirror**, for THREE independently sufficient reasons,
each verified in the pinned tree rather than inferred:

1. **It is OFF by default.** `disable_cascade_attn: bool = True`
   (`vllm/config/model.py:238`), whose own docstring says users must opt in
   (`:239-244`). Mirroring vLLM's default means cascade-off, so it can never be
   part of a default-configuration parity gate.
2. **It does not exist on the MRV2 runner.** The implementation lives only in
   the legacy `vllm/v1/worker/gpu_model_runner.py:504,2544,2582-2677`. A
   recursive grep for `cascade` over the entire MRV2 tree
   `vllm/v1/worker/gpu/` returns **zero hits**. Per
   [vllm-v1-v2.md](../vllm-v1-v2.md):9 we port MRV2, not the legacy runner.
3. **It is unreachable on our gate hardware even if opted into.**
   FlashAttention is the only backend that implements it
   (`vllm/v1/attention/backends/flash_attn.py:670-671`); FlashInfer hard-returns
   `False` with a "cascade attention doesn't work" TODO (`flashinfer.py:1445-1452`),
   as do Triton `:347-349`, FlexAttention `:1152-1153` and ROCm `:255,706`. On
   Blackwell the backend priority resolves FlashInfer FIRST
   (`vllm/platforms/cuda.py:145-151`), so on GB10/sm_121 — our gate box — the
   cascade path cannot be selected at all.

Additional narrowing even where it IS reachable: it needs `common_prefix_len >=
256`, `num_reqs >= 8`, `dcp == 1`, no alibi/SWA/local
(`flash_attn.py:1307-1319`), and it **forces full CUDA graphs off**
(`gpu_model_runner.py:3869`) — which on our stack would trade a measured
graph win for an unmeasured attention win. The shared-block count is
additionally hard-zero for SWA, chunked-local and Mamba groups
(`single_type_kv_cache_manager.py:873,1023,1184`), i.e. for our hybrid gate
models.

So the correct record state is "verified not owed on the mirrored path", not
"pending port". The scheduler-side input (`num_common_prefix_blocks`, #30) is
already computed and plumbed, so should upstream ever move cascade into MRV2 the
remaining work is the attention-backend half only. The row moves
`INVENTORIED` → `SPIKE` carrying this disposition, and is explicitly NOT
scheduled in the work breakdown.

#### llama.cpp `237ad9b96` — what its "prompt cache" actually is

Verified against `/home/mudler/_git/llama.cpp` (note: this checkout is the
`mudler/llama.cpp` FORK; one prefix-sharing path in it is a local patch and NOT
upstream capability — `server-context.cpp:3563-3593,4031-4041`, local commit
`41541d41d`, env `LLAMA_KV_PAGED`. It must not be cited as llama.cpp behaviour).

Three genuinely upstream mechanisms:

1. **`--prompt-cache` / `--prompt-cache-all` / `--prompt-cache-ro`**
   (`common/arg.cpp:1615-1635`) — **CLI-tool only**
   (`.set_examples({LLAMA_EXAMPLE_COMPLETION})`, so the server never sees it).
   Whole-context state to one file via `llama_state_load_file`
   (`tools/completion/completion.cpp:257-266`); reuse is a **strict linear
   common prefix** token loop (`:352-360`) with the divergent tail trimmed
   (`:372-381`); logits are not stored so the last token is re-decoded
   (`:389-394`).
2. **Server slot save / restore** — `POST /slots/{id}?action=save|restore|erase`
   (`tools/server/server-context.cpp:4797-4824`, route `server.cpp:245`), gated
   on `--slot-save-path` (`common/arg.cpp:3190-3203`), over
   `llama_state_seq_save_file` / `_load_file` (`include/llama.h:860-874`).
   Format `GGSQ` v2 (`src/llama-context.cpp:3137-3153`): token count, token ids,
   then per-sequence memory — KV cells with pos/seq metadata and raw K/V rows
   (`src/llama-kv-cache.cpp:2179-2247,2292-2360`), and for SSM/RWKV the
   recurrent `R`/`S` tensors (`src/llama-memory-recurrent.cpp:736-813`).
   Restore is position-relocatable but requires matching `n_layer`/capacity
   (`:2552-2559`). Documented cost: 1745 tokens → 14,309,796 B (~8.2 KB/token),
   **save 49.9 ms, restore 42.9 ms** (`tools/server/README.md:1069-1108`).
3. **In-memory slot reuse** — slot selection by longest common prefix
   (`server-context.cpp:1634-1677`, `slot_prompt_similarity` default 0.1), a
   RAM-resident `server_prompt_cache` of whole serialized slot states
   (`--cache-ram`, default 8192 MiB; `server-task.cpp:1603-1789`), and
   `--cache-reuse N` KV shifting of matching chunks
   (`server-context.cpp:3383-3445`).

**Verdict.** llama.cpp's reuse machinery is strictly weaker than APC: granularity
is a whole contiguous prefix of ONE sequence, entries are `std::move`d out of the
pool on load (`server-task.cpp:1745-1747`) so there is **no cross-request
sharing, no refcounted dedup and no fan-out** — two concurrent requests on the
same system prompt each compute and hold their own copy. vLLM beats it on every
reuse axis, and vLLM ALSO covers the disk-persistence axis at finer granularity
via the `fs` secondary tier (`vllm/v1/kv_offload/tiering/factory.py:61-65`,
content-addressed per-block files `file_mapper.py:112-121`).

**Exactly one genuine capability exists in llama.cpp and not in vLLM:** an
**explicit, caller-addressable, named per-sequence KV save/restore**. llama.cpp
can be told "serialize slot 3 to `foo.bin`" and later "load `foo.bin` into slot
0". vLLM has no analogue at any layer — its reuse is implicit and
content-addressed, the only handle on a prefix is a derived hash that cannot be
named, pinned or exported, and the only imperative control is bulk
`reset_prefix_cache` (`vllm/v1/core/kv_cache_manager.py:513-518`). `cache_salt`
partitions the namespace but does not name or pin. The corollary property is
that llama.cpp's restore is **synchronous and guaranteed**, whereas a vLLM disk
tier is a best-effort cache subject to eviction. This is the one place where
"mirror vLLM" leaves a real user-facing capability on the table, and where the
user's "and better" has a concrete target (proposal **B3** below).

### Our baseline

Established by reading our source, not the matrix. Three record claims were
wrong and are corrected in this change.

**Correction 1 — `KV-BLOCK-POOL` is better than `ANCHOR-BACKFILL` implied.** The
row's item text ("free list, refcounts, LRU eviction, core block lifecycle")
undersells a port that also covers group-aware hash mapping, the partial-alias
hash map, the unhashed-front/hashed-tail free ordering, reset, usage and
duplicate-hash eviction, under 13 tests (`tests/vllm/v1/test_block_pool.cpp:70`
… `:398`). `ANCHOR-BACKFILL` means "code and tests but no leaf spike" — this
document IS that spike, so the state moves to `PARTIAL`, which is the honest
label given the four deferrals (#17–#20), all throw-if-called or inert and none
reachable from a ported call site. It does not move to `DONE`: nothing in this
change may claim `DONE`, and four upstream behaviours are genuinely absent.

**Correction 2 — `KV-HYBRID-COORD` is better than `PARTIAL` implied.** The
entire hybrid cross-group intersection algorithm is ported verbatim, including
the iterate-to-fixed-point loop, the full-attention downward-closed fast path
with its "not yet looked up" sentinel, EAGLE extra-block bookkeeping, the
`is_simple_hybrid` single-iteration shortcut, final full-attention truncation and
`num_uncached_common_prefix_tokens`
(`src/vllm/v1/core/kv_cache_coordinator.cpp:424-537,281-323,500,507-521`). The
row keeps `PARTIAL` — differing per-group block sizes, DCP/PCP > 1 and
cross-attention remain assert-guarded — but its item text and anchors are
corrected so the residue is not read as breadth.

**Correction 3 — `KV-MAMBA-ALIGN` is recorded as `INVENTORIED` with no code
anchor, and that is WRONG.** The align-mode allocator is substantially ported:
the mode is read at `single_type_kv_cache_manager.cpp:639` and branched at
`:737` (`remove_skipped_blocks` frees the block two steps back), `:773`
(`get_num_blocks_to_allocate`), `:803` (`allocate_new_blocks`) and `:865`
(`pop_blocks_for_free`), with `reachable_block_mask` `:687` and
`retention_interval` `:689-700`. What is genuinely missing is (a) any config
path that SELECTS align — `MambaSpec` defaults `"none"`
(`include/vllm/v1/kv_cache_interface.h:306`) and no caller overrides it, (b) the
**runner-side state copy** the align comment at `:738-740` presumes (a recursive
grep for `last_state_block_idx` over `src/vllm/v1/worker/gpu/` returns nothing),
and (c) align-mode tests — our four Mamba cases
(`tests/vllm/v1/test_single_type_kv_cache_manager.cpp:811-874`) all exercise mode
`"none"`. So the row is materially further along than the record says, and its
remaining work is precisely identified rather than open-ended. It moves
`INVENTORIED` → `SPIKE`.

**Two gaps no row currently owns, now named.** (i) Prefix-cache hit-rate
statistics do not exist at ANY level — not the counters, not the aggregation,
not the endpoint (#23/#33); this blocks a binding gate, see below. (ii)
`reset_prefix_cache` is implemented at both the pool and manager level but is
**unreachable** — no CLI flag, no HTTP route, no engine RPC (#31).

**One stale blocker, cleared.** [prefix-caching.md](prefix-caching.md):112 parks
W3 ("model-positive APC on supported decoder families") as *"blocked on a
supported non-hybrid model family"*. That is no longer true: Qwen3-dense,
Qwen3-32B-NVFP4A16, Qwen3-Coder-30B, OPT and DeepSeek-V2-Lite have all landed
since, and dense models default prefix caching **ON**
(`src/vllm/entrypoints/model_loader.cpp:137`). **No gate has ever been run with
caching on for a dense model** — so our APC has real code, real unit tests, and
zero end-to-end evidence. Closing that is W3 below and it is the largest
correctness-and-speed exposure in this area.

**What the binding gate needs from us.** `BACKEND-GATE-CUDA-SGLANG-PREFIX`
(backend-matrix.md:123) is `READY` and requires proving cache hits/reuse in every
arm; the benchmark protocol makes that mandatory, and vLLM exposes it as
`vllm:prefix_cache_queries`/`_hits`. **We cannot currently produce that number at
all.** Hit-rate statistics are therefore not cosmetic — they are a hard blocker
on a binding competitor gate, which is why they lead the work breakdown.

### Port map

| Upstream | Local target | Disposition |
|---|---|---|
| `kv_cache_utils.py:539,431,498,560-561` `generate_block_hash_extra_keys` | `src/vllm/v1/core/kv_cache_utils.cpp:323-329` (replace the stub) | direct port; the encode path `:347-356` already accepts `extra_keys`, so this is extractor-only |
| `vllm/v1/request.py:154` + `openai/chat_completion/protocol.py:422` `cache_salt` | `include/vllm/v1/request.h`, `include/vllm/v1/engine/types.h`, `include/vllm/entrypoints/openai/protocol.h` | new optional field, first block only |
| `vllm/v1/metrics/stats.py:118,185` `PrefixCacheStats` | new `include/vllm/v1/metrics/prefix_cache_stats.h`; record at `kv_cache_manager.cpp:139`, reset at `:268` | direct port; consumed by `SERVE-METRICS` |
| `vllm/v1/metrics/loggers.py:547,558` counters | deferred to `SERVE-METRICS` (no `/metrics` route exists) | cross-row dependency, not implemented here |
| `vllm/config/kv_events.py:11-52` events | `src/vllm/v1/core/block_pool.cpp:128,239,293`; real `KVCacheEvent` replacing `block_pool.h:83` | direct port; `take_events()` `:311` becomes live |
| `block_pool.py` `cache_partial_block` + boundary hashes | `src/vllm/v1/core/block_pool.cpp:131-137,117-124` (replace throws) | direct port |
| `single_type_kv_cache_manager.py:1032-1072` align + `models/config.py:558-599` resolution | allocator EXISTS `single_type_kv_cache_manager.cpp:737,773,803,865`; add config selection over `kv_cache_interface.h:306` and the runner state copy in `src/vllm/v1/worker/gpu/runner.cpp` | completion, not a fresh port |
| `vllm/utils/hashing.py:70,76` xxhash / xxhash_cbor | `src/vllm/v1/core/kv_cache_utils.cpp` (new `HashFn`s over the existing pluggable seam `kv_cache_utils.h:342`) + a CLI flag | native, no optional dependency (see B1) |
| `entrypoints/serve/dev/cache/api_router.py:18-41` | `src/vllm/entrypoints/openai/api_server.cpp:311-327` | owned by `SERVE-UTILITY-ENDPOINTS`; this spike supplies the engine call, already present at `kv_cache_manager.cpp:264` |
| `vllm/v1/worker/gpu_model_runner.py:2544` cascade | **none** | NOT PORTED — verified not owed (§Upstream chain) |
| `vllm/v1/kv_offload/`, `vllm/distributed/kv_transfer/` | **none** | owned by `KV-OFFLOAD` / `KV-EXTERNAL-CACHE` / `KV-CONNECTORS` (`ROAD-V1-D4`) |

Recorded deviation, carried forward from
[porting-inventory.md](../porting-inventory.md):76: our shipped hash is
`sha256_cbor` while upstream's DEFAULT is `sha256` (pickle), which is
impractical to reproduce in C++. Consequence for interop, newly noted here: a
shared content-addressed tier would NOT interoperate with a stock vLLM unless
that vLLM is launched with `--prefix-caching-hash-algo sha256_cbor`. This must
be stated wherever `ROAD-V1-D4` claims LMCache interop.

### Tests to port

Upstream suite, enumerated at the pin. Counts are `^def test_` per file.

| Upstream test file | Cases | Tier | Disposition |
|---|---:|---|---|
| `tests/v1/core/test_prefix_caching.py` | 56 | T-unit | the executable spec for this feature; ~20 already have counterparts across our KV suites, the rest map to W1–W6 |
| `tests/v1/core/test_kv_cache_utils.py` | 51 | T-unit | hashing/queue/config; 23 ported (`tests/vllm/v1/test_kv_cache_utils.cpp`) |
| `tests/v1/core/prefix_cache/test_partial_prefix_cache_primitives.py` | 10 | T-unit | **entirely unported** — the W5 spec |
| `tests/v1/core/test_single_type_kv_cache_manager.py` | 9 | T-unit | 26 local cases exist (`tests/vllm/v1/test_single_type_kv_cache_manager.cpp`) |
| `tests/v1/core/test_kv_cache_metrics.py` | 1 | T-unit | unported (W1) |
| `tests/v1/core/test_reset_prefix_cache_e2e.py` | 1 | T-e2e | unported (W7) |
| `tests/v1/core/test_contiguous_kv_packing.py` | 0 (class-based) | T-unit | already SKIP-marked to `KV-PREFIX-CACHE` at `tests/vllm/v1/test_single_type_kv_cache_manager.cpp:1050` |
| `tests/v1/core/test_deferred_block_free.py` | 11 | T-unit | deferred free under async connectors — SKIP to `KV-CONNECTORS` |
| `tests/v1/core/test_scheduler_e2e.py` | 2 | T-unit | incl. `test_prefix_cache_stats_is_recorded` (W1) |
| `tests/v1/core/test_scheduler.py` (cache-relevant subset) | ~37 of 78 | T-unit | incl. `test_scheduler_reset_prefix_cache`, `test_external_prefix_cache_metrics`, `test_kv_connector_handles_preemption` |
| `tests/models/language/pooling/test_auto_prefix_cache_support.py` | 3 | T-unit | the pooling default matrix (#2) — SKIP until pooling exists |
| `tests/engine/test_arg_utils.py::test_prefix_cache_default`, `tests/test_config.py::test_is_prefix_caching_supported` | 2 | T-unit | the tri-state and the support matrix — directly gate our `ResolveEnablePrefixCaching` |
| `tests/v1/engine/test_engine_args.py` | 2 | T-unit | hash-algo CLI incl. xxhash (W7/W8) |
| `tests/kernels/attention/test_cascade_flash_attn.py` | 2 | — | **NOT PORTED** — see the cascade disposition |
| `tests/v1/e2e/general/test_mamba_prefix_cache.py` | 2 (`_mrv1`, **`_mrv2`**) | T-e2e | W6; the `_mrv2` case is the one that binds us |
| `tests/v1/e2e/test_hybrid_chunked_prefill.py` | 1 | T-e2e | partially covered by both paged-engine gates |
| `tests/v1/e2e/general/test_cascade_attention.py` | 1 | — | **NOT PORTED** — records the §Upstream-chain disposition as a SKIP marker |
| `tests/v1/core/test_scheduler.py` (prefix-cache params) | — | T-unit | scheduler on/off parameterization; partly covered |

Named cases mapped to work rows (the ones that drive design, not the full 56):

- **W1 stats:** `test_prefix_cache_stats_disabled` (`:1965`),
  `test_kv_cache_metrics_collector_smoke`.
- **W2 extra keys / salt:** `test_generate_block_hash_extra_keys` (`:485`),
  `_no_mm_inputs` (`:517`), `_cache_salt` (`:530`), `_lora` (`:641`),
  `_prompt_embeds` (`:568`), `_prompt_embeds_cached` (`:591`),
  `_different_prompt_embeds` (`:618`), `test_cache_key_salting` (`:1766`),
  `test_mm_prefix_caching` (`:1658`), `test_hash_tokens_different_mm_input`
  (`:695`), `test_null_parent_block_hash` (`:2102`).
- **W3 model-positive APC:** `test_prefill` (`:225`), `test_decode` (`:1252`),
  `test_hash_block_correct_reuse` (`:1374`),
  `test_computed_blocks_not_evicted` (`:1415`),
  `test_basic_prefix_caching_disabled` (`:1475`),
  `test_prefill_not_enough_free_blocks_with_computed_blocks` (`:1846`).
- **W4 events:** `test_kv_cache_events` (`:2040`), `_with_lora` (`:2170`),
  `test_block_stored_event_group_idx` (`:2228`), `_multiple_groups` (`:2282`),
  `_out_of_bounds` (`:2368`), `test_block_removed_event_group_idx` (`:2405`).
- **W5 partial blocks (all 10):** `test_boundary_hashes_reuse_fine_grained_chain`
  (`:90`), `test_cache_partial_block_kv_cache_events` (`:110`),
  `test_partial_block_replacement_emits_remove_then_store_events` (`:172`),
  `test_later_request_hits_cached_partial_tail` (`:236`),
  `test_cache_partial_block_uses_fine_grained_boundary_hash` (`:277`),
  `test_cache_partial_block_requires_hash_boundary` (`:315`),
  `test_cache_partial_block_duplicate_checks_all_blocks_for_hash` (`:336`),
  `test_reset_prefix_cache_clears_partial_entry_metadata` (`:375`),
  `test_evict_cached_block_removes_full_hash_and_partial_entry` (`:392`),
  `test_partial_block_promotes_to_direct_full_block_hash` (`:408`).
- **W6 mamba align:** `test_prefill_hybrid_model_mamba_align` (`:987`),
  `test_hybrid_cache_mamba_align_shared_prefix_detection` (`:1027`),
  `test_hybrid_model_mamba_align_with_dynamic_draft_tokens` (`:1087`), plus e2e
  `test_mamba_prefix_cache_mrv2`.
- **W7 reset:** `test_reset_prefix_cache` (`:1925`), `test_reset_prefix_cache_e2e`.
- **Retention interval (#37):** `test_hybrid_local_kv_retention_interval_*`
  (`:2995`, `:3066`, `:3107`, `:3190`, `:3264`),
  `test_pure_swa_retention_*` (`:3811`, `:3850`, `:3882`),
  `test_mamba_reachable_block_mask_sparsifies_retention` (`:3910`).
- **External-hit interaction (`ROAD-V1-D4`, not this spike):**
  `test_cache_hit_local_and_external*` (`:3524`, `:3645`, `:3656`, `:3682`) —
  SKIP-marked to `KV-EXTERNAL-CACHE`.

Per [test-porting.md](../test-porting.md):6, every case that cannot pass yet is
checked in SKIPPED with a tracked row reason — the convention already used at
`tests/vllm/v1/test_single_type_kv_cache_manager.cpp:1050-1086` — never dropped.

### Gates

1. **CPU (every work row):** clean warnings-as-errors build and full CTest; the
   ported cases above green or explicitly SKIP-marked with a row reason.
2. **Hash byte-exactness (W2, W8):** every shipped `HashFn` reproduces its
   Python counterpart byte-for-byte on the block-hash value shapes, asserted
   against vectors captured from the pinned vLLM — the standard already met by
   `sha256_cbor` (`tests/vllm/v1/test_kv_cache_utils.cpp:550`). A new algorithm
   that is not byte-exact does not ship.
3. **Correctness precondition (W3, W6 — DGX GB10):** token-exact on the
   affected model gates with caching **ON**, **OFF**, and against the pinned
   vLLM oracle. Caching must be output-invariant: a cache hit may change timing,
   never a token. This is a precondition that is never traded for speed.
4. **Cache-hit proof (W1, W3, W6):** every benchmark arm reports queries/hits
   and a non-zero hit rate on a shared-prefix corpus, on BOTH engines. Per
   [prefix-caching.md](prefix-caching.md):117 logical input-token counters do not
   reveal prefix hits, so an arm without a hit counter is void.
5. **Every-axis performance (W3, W6, W9 — DGX GB10):** match or beat vLLM on
   total and output throughput, req/s, TTFT, TPOT/ITL and peak memory at the
   large-concurrency operating point, per
   [benchmark-protocol.md](../benchmark-protocol.md). A caching change that
   raises hit rate but regresses ANY axis is a failed change, not a win. Where
   SGLang is the faster applicable reference, it binds too
   (`BACKEND-GATE-CUDA-SGLANG-PREFIX`).
6. **Default-flip discipline:** per the standing parity-enabler rule, any lever a
   binding grid depends on has its default flipped BEFORE the binding run; the
   grid then measures pure production defaults.
7. **No-regression when unused (W8, W9):** an opt-in surface must be provably
   inert when off — a same-binary A/B with the feature disabled reproduces the
   prior binding numbers within run noise.

### Dependencies

- **Rows:** `SERVE-METRICS` must exist before hit-rate counters are externally
  observable (W1 supplies the engine-side numbers regardless);
  `SERVE-UTILITY-ENDPOINTS` owns the `/reset_prefix_cache` route (W7);
  `KV-EXTERNAL-CACHE` / `KV-OFFLOAD` / `KV-CONNECTORS` own everything external
  (`ROAD-V1-D4`); `BACKEND-GATE-CUDA-SGLANG-PREFIX` CONSUMES W1 and W6;
  `ENG-ASYNC-SCHED` owns placeholder-discounted cacheable lengths and must not
  be approximated here (`async_scheduler.cpp:67-73`).
- **Models:** W3 needs any dense model (Qwen3-dense / Qwen3-32B / Qwen3-Coder /
  OPT / DeepSeek-V2-Lite are all present and default caching ON). W6 needs a
  hybrid gate model (27B/35B).
- **Hardware:** W1, W2, W4, W5, W7 are **CPU-only, no GPU** — they are pure
  behavioural ports gated by CTest on the dev box. W3, W6, W8, W9 need **DGX
  GB10 (sm_121a)** under one `flock /tmp/gpu` per series, on an idle box.
- **No new library dependency.** xxhash (B1/W8) is a ~200-line public-domain
  algorithm implemented natively, exactly as the existing SHA-256 and CBOR
  encoders are (`kv_cache_utils.cpp:114,200-275`) — no third-party package,
  which is itself the parity edge.
- Disk: none. Nothing here downloads a checkpoint that is not already present.

### Work breakdown

Ordered so the highest-value parity gaps land first. Nothing below is claimed
started; every row is `open`.

| W | Deliverable | Gate | HW | State |
|---:|---|---|---|---|
| W1 | **Prefix-cache statistics** — DONE. `BaseCacheStats`/`PrefixCacheStats`/`CachingMetrics` ported 1:1 (`include/vllm/v1/metrics/stats.h`); recorded in `KVCacheManager::get_computed_blocks`, flagged by `reset_prefix_cache`, taken-and-swapped by `make_prefix_cache_stats()`, and folded per step into a 1000-request sliding window exposed as `Scheduler/EngineCore/LLMEngine::prefix_cache_metrics()`. **`log_stats` now DEFAULTS ON** (mirroring upstream's `disable_log_stats=False`), so no benchmark arm is void for want of a counter. `Request::num_preemptions` un-deferred to feed the mutually-exclusive `preempted_*` triple | `tests/vllm/v1/test_prefix_cache_stats.cpp` 12/12 — token-not-request counting, preempted mutual exclusion, empty-observation and window-slide rules, reset ordering, take-and-swap, log_stats-off; plus the **first measured hit rate: 0.75 (1920/2560 tokens over 16 requests)** on a repeated-prefix corpus, with a caching-OFF 0.0 negative control | CPU | **DONE** |
| W2 | **`generate_block_hash_extra_keys` + `cache_salt`** — replace the stub `kv_cache_utils.cpp:323-329`; `cache_salt` through API → EngineCoreRequest → Request → first-block hash | the 11 named extra-key/salt cases; byte-exact hash vectors | CPU | open |
| W3 | **Model-positive APC end-to-end** — the first cache-ON gate on a dense model; clears the stale blocker at [prefix-caching.md](prefix-caching.md):112 | token-exact ON/OFF/oracle + non-zero hit rate + every-axis grid vs vLLM | **DGX** | open |
| W4 | **KV-cache events** (`KV-EVENTS`) — real `KVCacheEvent`, BlockStored/BlockRemoved/AllBlocksCleared emission, live `take_events()` | the 6 named event cases | CPU | open |
| W5 | **Partial-block primitives** — `cache_partial_block`, boundary hashes, partial→full promotion; removes 2 throwing stubs. **Ported as an UNWIRED primitive with tests, mirroring upstream's own dead-code status — no call site is added** (§Risks R7) | all 10 `test_partial_prefix_cache_primitives.py` cases | CPU | open |
| W6 | **Mamba/GDN align completion** (`KV-MAMBA-ALIGN`) — config resolution selecting `align`, the runner-side state copy, align tests | the 3 named align cases + e2e `test_mamba_prefix_cache_mrv2`; then the hybrid cache-on arm of `BACKEND-GATE-CUDA-SGLANG-PREFIX` | **DGX** | open |
| W7 | **`reset_prefix_cache` reachability** (dev-mode gated, mirroring `VLLM_SERVER_DEV_MODE`) + `--prefix-caching-hash-algo` CLI + `skip_reading_prefix_cache` (#9c) | `test_reset_prefix_cache` + `_e2e`; server help contract | CPU | open |
| W8 | **B1/B3 surpass work** — native xxhash/xxhash_cbor, then the named session save/restore surface | §Risks B1/B3 measurement plans; inert-when-off A/B | **DGX** | open |
| W9 | **B4 eviction-policy study** — measured, not asserted | §Risks B4 measurement plan; every-axis no-regression | **DGX** | open |
| — | Cascade attention | **NOT SCHEDULED** — verified not owed (§Upstream chain) | — | dispositioned |
| — | Offload / external tiers | owned by `KV-OFFLOAD`, `KV-EXTERNAL-CACHE`, `KV-CONNECTORS` (`ROAD-V1-D4`) | — | not owned here |

### Risks/decisions

**R1 — the extra-keys stub is a latent correctness trap, not a feature gap.**
`generate_block_hash_extra_keys` returning a constant
(`kv_cache_utils.cpp:323-329`) means two prompts that differ ONLY in multimodal
input, LoRA adapter or cache salt hash to the SAME block key and would serve each
other's KV — wrong output, silently. It is not a live bug today because we have
no multimodal path, no LoRA and no `cache_salt` field, so no request can differ
only in those. It becomes live the moment any of the three lands. Decision: W2
lands BEFORE the first multimodal or LoRA consumer, and the multimodal/LoRA rows
must depend on it explicitly rather than discover it.

**R2 — our hash default deviates from upstream's.** Upstream defaults to
`sha256` (pickle); we ship `sha256_cbor` because pickle's opcode stream is
impractical in C++ (`kv_cache_utils.h:18-31`). Blocks are therefore keyed
differently from a stock vLLM. This is invisible within our own process and
becomes load-bearing only for cross-engine content-addressed sharing, where the
peer must be launched with `--prefix-caching-hash-algo sha256_cbor`. Recorded
here so `ROAD-V1-D4` cannot claim LMCache interop without stating it.

**R3 — a cache result can silently erase the work being measured.** A repeated
prompt with caching on skips prefill while leaving logical input-token counts
unchanged, so a benchmark can look faster for the wrong reason. Mitigation is
the existing rule: runtime policy plus hit counters are mandatory trace
metadata, and cache-off remains the binding parity configuration until a
cache-on arm is proven equivalent on both engines (gate 4).

**R4 — W6 touches the recurrent state path.** Align mode frees a GDN state block
two steps behind and relies on a runner-side state copy that does not exist yet.
Getting it wrong corrupts recurrent state and shows up as wrong tokens, not as a
crash. Decision: W6 lands behind an explicit mode selection with mode `"none"`
byte-identical, and the token-exact gate runs in both modes.

**R5 — hit rate is not an axis.** Every "better caching" idea below can raise hit
rate while losing throughput. The acceptance rule is per-axis, so a hit-rate
improvement that regresses any binding axis is rejected. This is why B4 is a
measured study, not a planned implementation.

**R6 — we port a function upstream has deleted.** `hash_request_tokens`
(`kv_cache_utils.cpp:363`) mirrors an API that no longer exists at the pin; it
was superseded by the `get_request_block_hasher` closure factory
(`kv_cache_utils.py:673-730`), which we ALSO port (`:392`). Behaviour is
equivalent today, so this is drift rather than a defect, but it is exactly the
kind of divergence the upstream-sync cycle exists to catch. Decision: the next
sync pass retires the dead entry point rather than carrying two spellings of one
concept. Logged so it is not rediscovered as a bug.

**R7 — do not port dead upstream code as live behaviour.** `cache_partial_block`
is defined at `block_pool.py:358-457` and has **no caller anywhere in `vllm/`**;
only its test module exercises it. Partial-block caching is therefore a staged
primitive at this pin, not a shipped feature. Porting it *with a call site*
would make us diverge from vLLM in observable behaviour while believing we were
mirroring it — the failure mode the whole-chain rule is meant to prevent. W5
therefore ports the primitive plus its ten tests and wires nothing, holding the
mirror exactly where upstream holds it; when upstream activates it, we activate
it in the same sync.

**R8 — hidden traps that will silently produce a wrong port.** Recorded because
each is a plausible-looking mistake: (a) `BlockHashWithGroupId` is a byte concat
with a 4-byte big-endian suffix, so `get_block_hash` depends on a fixed-length
digest — a variable-length hash breaks it; (b) extra-key ORDER is
lora → mm → cache_salt → prompt_embeds and the salt applies only to block 0
(chaining transitively thereafter); (c) LRU-ness comes from `free_blocks`
ordering, NOT from the queue, which is plain FIFO — a "correct" priority queue
would change eviction order; (d) the `num_tokens - 1` cap can force recompute of
a whole BLOCK, not one token, because `allocate_slots` needs block-aligned
counts; (e) hybrid `find_longest_cache_hit` is an iterative fixed point with
truncation — a single-pass intersection is wrong; (f) `reset_prefix_cache`
legitimately FAILS unless exactly one block (the null block) is in use.

#### The "and better" axes — concrete, with how each is measured

**B1 — native fast hashing with no optional dependency.** vLLM's fast algorithms
(`xxhash`, `xxhash_cbor`, `vllm/utils/hashing.py:70,76`) require the optional
`xxhash` package and are therefore absent from a default install; its shipped
default (`sha256` over pickle) is neither fast nor cross-language. We can ship
all four semantics natively. *Measurement:* (a) a hash microbenchmark in ns per
block over identical token blocks, ours vs all four vLLM algorithms; (b) the
share of prefill wall-time spent hashing, from an nsys/step attribution — **if
that share is below ~1% the lever is closed as NEUTRAL and recorded as such**,
because a faster hash of a negligible cost is not a win; (c) only if the share
justifies it, a TTFT/prefill A/B on the cache-on dense gate. Honest position:
this is a candidate lever whose value is unknown until (b) is measured, and the
portable-lever history on this project says most such levers measure neutral.

**B2 — deterministic cross-process block hashes, with no environment footgun.
CORRECTED 2026-07-22: this was recorded as an existing advantage and it is
actually a DEFICIT.** vLLM seeds `NONE_HASH` with `os.urandom(32)` unless
`PYTHONHASHSEED` is set (`kv_cache_utils.py:99-114`), so its own documentation
requires operators to set that variable on every instance or cross-process cache
sharing silently produces zero hits
(`docs/features/kv_offloading_usage.md:117-120`). The original claim here —
"ours is deterministic by construction" — is **false**: we mirror the random
branch (`kv_cache_utils.cpp:312-318`), our only production caller passes no seed
(`model_loader.cpp:144`), and we expose neither the escape hatch nor the
warning, so we are strictly WORSE than upstream on this axis. The seam already
accepts a seed, so the fix is cheap; it is owned by
[kv-persistence-lmcache.md](kv-persistence-lmcache.md) W1, where it is a hard
blocker on any persisted cache. *Measurement (unchanged, now a target rather
than a claim):* two independently launched processes over the same corpus emit
byte-identical block-hash lists with zero configuration; on a shared tier, an
unconfigured vLLM pair scores 0% cross-process hit rate while ours scores the
full shared prefix. This is a correctness/usability win reported as a hit-rate
number, not a throughput claim.

**B3 — imperative named session save/restore: the one thing vLLM lacks.** Per
§Upstream chain, vLLM has no caller-addressable per-sequence KV export;
llama.cpp does. Proposal: an opt-in surface that exports a request's block list
and re-imports it, giving snapshot-a-conversation-and-resume and session
migration. *Measurement:* (a) restore-vs-recompute TTFT at 8k+ prompt tokens,
target ≥5× faster than full prefill; (b) bytes/token and save/restore
milliseconds against llama.cpp's published floor (~8.2 KB/token, 49.9 ms save /
42.9 ms restore for 1745 tokens, `tools/server/README.md:1069-1108`); (c) a
token-exactness gate — a restored continuation must equal the uninterrupted
continuation; (d) an inert-when-off A/B reproducing the binding grid. *Scope
risk:* this is genuinely beyond vLLM, so there is no oracle to mirror and no
upstream test to port — it must be specified from our own semantics, which is
exactly the situation the discipline rules treat as highest-risk. It is
therefore sequenced LAST (W8), after parity is closed, and it is opt-in.

**B4 — eviction smarter than LRU (a study, not a plan).** vLLM evicts strict LRU
over the free queue. Under skewed shared-prefix workloads an LFU/S3-FIFO/
W-TinyLFU policy may hold hot system prompts that LRU drops. Eviction is
**semantically transparent** — evicting a cached block loses a hit, never changes
a token — so this is a pure-performance, correctness-neutral lever, which is
what makes it safe to explore. *Measurement:* a Zipfian shared-prefix workload
generator; report hit rate AND every binding axis for LRU vs each candidate;
accept only if hit rate rises with no axis regression, against both vLLM and
SGLang arms on the same corpus. *Prior:* unproven; recorded as a hypothesis with
a falsification path, not a claim.

**B5 — cache-on where vLLM defaults off (hybrid/GDN).** vLLM's hybrid default is
caching OFF, and when explicitly enabled Qwen3.5/3.6 must use `align` because
`all` is unsupported (`vllm/model_executor/models/qwen3_5.py:294-301`). Once W6
lands, we can hold a cache-on hybrid configuration. *Measurement:* the
`BACKEND-GATE-CUDA-SGLANG-PREFIX` grid with hits proven in every arm.
*Constraint, non-negotiable:* per [prefix-caching.md](prefix-caching.md):119 a
cache-on arm may NOT be compared against a cache-off vLLM denominator. Beating
vLLM here means beating vLLM's own `mamba_cache_mode=align` configuration, not
beating a configuration upstream disabled.
