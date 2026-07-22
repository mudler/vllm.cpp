# Spike: KV persistent state to disk, and external KV providers (LMCache)

**Rows owned:** `KV-OFFLOAD`, `KV-EXTERNAL-CACHE`, `KV-CONNECTORS`.
**Claim:** `CLAIM-KV-PERSISTENCE-LMCACHE`. **SPIKE ONLY — no implementation.**

Portfolio row: `ROAD-V1-D4` (external KV-cache provider interoperability). Adjacent
and NOT owned: `KV-PREFIX-CACHE`, `KV-BLOCK-POOL`, `KV-EVENTS`, `KV-MAMBA-ALIGN`
(all owned by [prefix-prompt-caching-parity.md](prefix-prompt-caching-parity.md)),
`SERVE-METRICS`, `SERVE-UTILITY-ENDPOINTS`, `KV-MLA-SPEC`.

This spike answers the user's 2026-07-22 ask — *"let's do the KV persistent
state to disk support, and LMCache support too"* — under the standing bar: **the
same featureset as vLLM, and better.** It builds on the caching spike, which
established that vLLM already covers disk persistence through its `kv_offload`
filesystem tier, and does not re-derive anything settled there.

### Scope

The complete pinned-vLLM KV-offload and external-KV-provider surface, enumerated
with `file:line` and a DONE / PARTIAL / MISSING verdict per feature grounded in
our source. It covers: the `kv_offload` core abstractions, the CPU primary tier
and its LRU/ARC policies, the filesystem (disk) secondary tier and its exact
on-disk format, the tiering manager's promotion/cascade/eviction ordering, the
`KVConnectorBase_V1` interface and its full scheduler/worker lifecycle, the
connector registry and out-of-tree module seam, KV-cache events as they are
consumed by connectors, the `KVTransferConfig` surface, and precisely what
LMCache integration requires of a host engine and where that code physically
lives. It then specifies OUR disk-persistence design — which parts mirror
faithfully and which need a C++-native equivalent — and scopes the one genuine
beyond-parity item, imperative named per-sequence save/restore, with its
identity and invalidation safety rules stated explicitly.

Out of scope for this spike's rows: prefill/decode disaggregation as a *serving
topology* (the NIXL/Mooncake/MoRI-IO transports are enumerated for completeness
and dispositioned, not scheduled), the Prometheus endpoint itself
(`SERVE-METRICS`), the utility-endpoint router (`SERVE-UTILITY-ENDPOINTS`), and
the prefix-cache statistics counters (`KV-PREFIX-CACHE` W1) that everything here
depends on for evidence.

**Headline answer to the user.** *KV persistent state to disk* is a MIRROR job
and it is tractable: vLLM's disk tier is 101 lines of `open`/`write`/`readv`
over one raw file per block, with nothing Python-specific in the byte path, and
we can port it essentially verbatim. *LMCache support* is NOT the same kind of
job: the vLLM-facing glue is vendored in-tree, but every byte of storage, the
wire protocol, the CUDA-IPC handoff and the config schema live in an external,
unpinned PyPI package that is not installed anywhere on this project's boxes.
A faithful C++ LMCache client is reverse-engineering against a moving target.
The honest sequencing is therefore: port the connector seam and the disk tier
first (real user value, full upstream oracle), and treat LMCache as an
interop-compatibility target reached over that seam, gated on two blockers we
own — our block hashes are neither byte-compatible with vLLM's default nor
stable across processes. See §Risks R1/R2.

### Upstream chain

vLLM pin `e24d1b24` (verified: `git -C /home/mudler/_git/vllm log --oneline -1`
→ `e24d1b24 Fix Transformers modeling backend usage stats (#47472)`).

**Naming correction to the record.** There is no `vllm/v1/offloading/`. The
subsystem lives in three cooperating trees, and two prior-record assumptions are
wrong at this pin: `SharedStorageConnector` was **renamed `ExampleConnector`**
(`vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py`), and
`P2pNcclConnector` **no longer exists** — no file, no registry entry, no
reference. `vllm/config/offload.py` is *model-weight* offloading
(`UVAOffloadConfig`, `vllm/config/offload.py:12`) and is unrelated to KV; it
belongs to `ENG-WEIGHT-OFFLOAD`, not here.

| Tree | Role |
|---|---|
| `vllm/v1/kv_offload/` | core abstractions, media, managers, policies, tiers, workers |
| `vllm/distributed/kv_transfer/kv_connector/v1/` | the `KVConnectorBase_V1` ABI and all 16 connectors |
| `vllm/v1/simple_kv_offload/` | a SEPARATE, simpler CPU-only offload path behind `SimpleCPUOffloadConnector` |

#### The enumerated surface, with our status

Status key: **DONE** = ported with tests / **PARTIAL** = ported with a named
omission / **MISSING** = absent / **N-A** = verified not owed on the path we
mirror.

| # | Feature | Upstream `file:line` | Ours | Status |
|---:|---|---|---|---|
| 1 | `KVTransferConfig` — `kv_connector`, `engine_id`, `kv_buffer_device`, `kv_buffer_size`, `kv_role`, `kv_rank`, `kv_parallel_size`, `kv_ip`, `kv_port`, `kv_connector_extra_config`, `kv_connector_module_path`, `enable_permute_local_kv`, `kv_load_failure_policy` | `vllm/config/kv_transfer.py:22-75`; `kv_role` REQUIRED whenever `kv_connector` is set `:102-106` | absent — no `--kv-transfer-config`, no config struct | **MISSING** (`KV-EXTERNAL-CACHE`) |
| 2 | `KVConnectorRole` = `SCHEDULER` / `WORKER`; two instances of one class, never sharing memory | `vllm/distributed/kv_transfer/kv_connector/v1/base.py:124` | absent | **MISSING** |
| 3 | `KVConnectorBase_V1` — **exactly 7 abstract methods**: worker `start_load_kv`/`wait_for_layer_load`/`save_kv_layer`/`wait_for_save`, scheduler `get_num_new_matched_tokens`/`update_state_after_alloc`/`build_connector_meta`. All other ~30 hooks have safe defaults | `base.py:171,293,311,325,347,454,489,510` | absent | **MISSING** — this is the minimum viable seam |
| 4 | Scheduler→Worker payload `KVConnectorMetadata` (rides on `SchedulerOutput`); Worker→Scheduler `KVConnectorWorkerMetadata` with an abstract `aggregate(other)` folding all ranks | `base.py:141,150,162` | shapes PRE-CARVED but empty: `include/vllm/v1/core/sched/output.h:30-31`, `include/vllm/v1/engine/types.h:26,30` | **MISSING** (seam exists) |
| 5 | `get_num_new_matched_tokens(request, num_computed_tokens) -> (int\|None, bool)` — **THREE states**: hit count, `None` = "not ready, re-ask next step", and the async-load flag. Must be side-effect free | `base.py:454`; scheduler call `vllm/v1/core/sched/scheduler.py:736-742`; the `None` path pops + `prepend_request` + `continue` at `:745-751` | absent; the accounting term is already NAMED at `include/vllm/v1/core/kv_cache_manager.h:31` (`ext_comp = num_external_computed_tokens`) | **MISSING** (arithmetic seam pre-shaped) |
| 6 | `update_state_after_alloc(request, blocks, num_external_tokens)` — may be called TWICE per request under async load | `base.py:489`; `scheduler.py:933-937` | absent; `include/vllm/v1/core/single_type_kv_cache_manager.h:122` already documents the external-computed-tokens allocation entry point | **MISSING** (seam exists) |
| 7 | `build_connector_meta(scheduler_output)` — must not mutate its argument; RESETS connector internal state | `base.py:510`; `scheduler.py:1118-1119` via `_build_kv_connector_meta` `:1137-1140` | absent | **MISSING** |
| 8 | `request_finished(request, block_ids) -> (bool, dict\|None)` — returning True transfers block-freeing OWNERSHIP to the connector until `get_finished()` reports the id | `base.py:542`; `scheduler.py:2340,2369` | absent | **MISSING** — the deferred-free semantics, see §Risks R6 |
| 9 | `SupportsHMA` marker + `request_finished_all_groups(request, block_ids: tuple[list[int], ...])`; non-HMA connectors ASSERT exactly one KV cache group | `base.py:85,93,117`; `scheduler.py:2371`; factory guard `factory.py:131,145` | absent | **MISSING** — load-bearing for our hybrid models, see §Risks R7 |
| 10 | Worker lifecycle on the V2 runner: `register_kv_caches` / `set_host_xfer_buffer_ops` at init; `pre_forward` = `handle_preemptions` → `bind_connector_metadata` → `start_load_kv`; `post_forward` = `wait_for_save` → `get_finished` → `get_block_ids_with_load_errors` → stats → events → `build_connector_worker_meta` → `clear_connector_metadata`; plus `no_forward` for zero-token steps | `vllm/v1/worker/gpu/kv_connector.py:56,57,61,67,68,72,77,85,87,89,90,91,93,95,98,107` | absent | **MISSING** — this is the MRV2 path we mirror |
| 10b | Legacy V1 mixin doing the same as a `@contextmanager` | `vllm/v1/worker/kv_connector_model_runner_mixin.py:64-112` | absent | **N-A — no port owed**; we port MRV2 ([vllm-v1-v2.md](../vllm-v1-v2.md):9) |
| 11 | LAYERWISE hooks wired by a DECORATOR on attention layers, not by the runner: `wait_for_layer_load(layer_name)` pre-attention, `save_kv_layer(...)` post-attention | `vllm/model_executor/layers/attention/kv_transfer_utils.py:15,39,45,51,57` | absent | **MISSING** — and it forces PIECEWISE cudagraphs, see §Risks R8 |
| 12 | `handle_preemptions(metadata)` — called BEFORE preempted/evicted blocks are overwritten; mandatory for any async-save connector | `base.py:285`; `vllm/v1/worker/gpu_model_runner.py:4078-4081` | absent; our preemption is recompute-only and frees blocks immediately (`src/vllm/v1/core/sched/scheduler.cpp:102-113`) | **MISSING** — a correctness hook, not an optimization |
| 13 | `bind_gpu_block_pool(block_pool)` — hands the connector refcount + prefix-cache iteration | `base.py:443`; `scheduler.py:280` | absent; `BlockPool::evict_blocks` is our only connector-shaped code and it THROWS: `src/vllm/v1/core/block_pool.cpp:140-142` | **MISSING** |
| 14 | `get_required_kvcache_layout(vllm_config) -> "HND"\|"NHD"\|None` (classmethod); `prefer_cross_layer_blocks` + `register_cross_layers_kv_cache` | `base.py:585,177,261`; consumed `vllm/distributed/kv_transfer/kv_connector/utils.py:41` | absent; our full-attention page is a fixed interleaved `[K(bs,h,d)\|V(bs,h,d)]` (`src/vllm/model_executor/models/qwen3_5.cpp:2408-2426`) with no layout negotiation | **MISSING** |
| 15 | `KVConnectorFactory` — LAZY registry (`dict[str, Callable[[], type]]`), `register_connector(name, module_path, class_name)`, dup-registration rejected | `factory.py:28,31,37-38,43,78` | absent | **MISSING** |
| 16 | Out-of-tree connector seam: `kv_connector_module_path` takes PRIORITY over the internal registry; empty string rejected; 3-arg ctor ENFORCED via `supports_kw(cls, "kv_cache_config")` | `factory.py:96,102-123,115-123` | absent | **MISSING** — the Python-plugin half, see §Port map "what does NOT mirror" |
| 17 | **16 registered connectors**: `ExampleConnector`, `ExampleHiddenStatesConnector`, `LMCacheConnectorV1`, `LMCacheMPConnector`, `NixlConnector`/`NixlPullConnector`/`NixlPushConnector`, `MultiConnector`, `MoRIIOConnector`, `OffloadingConnector`, `DecodeBenchConnector`, `MooncakeConnector`, `MooncakeStoreConnector`, `FlexKVConnectorV1`, `SimpleCPUOffloadConnector`, `HF3FSKVConnector` | `factory.py:152,158,164,170,176,182,188,194,200,206,212,218,223,228,233,238` | absent | **MISSING** (`KV-CONNECTORS`) |
| 18 | **`ExampleConnector`** (ex-`SharedStorageConnector`) — the reference "persist KV to a shared filesystem" impl: directory per prompt-hash, one safetensors file per layer, fully SYNCHRONOUS (`wait_for_layer_load`/`wait_for_save` are no-ops), hit test = `os.path.exists` | `example_connector.py:405,422,427,439,181-182,122,109,246,221,203,192,248,380,390,442`; config key `shared_storage_path` `:104` | absent | **MISSING** — the simplest complete disk oracle |
| 19 | **`OffloadingConnector`** — the production tiered path; `class OffloadingConnector(KVConnectorBase_V1, SupportsHMA)`; `prefer_cross_layer_blocks = True`, layout `"HND"` | `offloading_connector.py:46,47-49,181-183`; halves `offloading/scheduler.py:319`, `offloading/worker.py:33` | absent | **MISSING** (`KV-OFFLOAD`) |
| 20 | Tier model: secondary tiers CANNOT touch GPU memory. Store = GPU→CPU→secondary; load = secondary→CPU→GPU | `vllm/v1/kv_offload/tiering/base.py:52-64`; doc `docs/features/kv_offloading_usage.md:15` | absent | **MISSING** — a structural constraint, not a detail |
| 21 | `OffloadKey = block_hash_bytes + group_idx.to_bytes(4, "big")`, deliberately bytes not a tuple (GC overhead) | `vllm/v1/kv_offload/base.py:27-30,35-47` | our `BlockHashWithGroupId` uses the IDENTICAL 4-byte big-endian packing: `src/vllm/v1/core/kv_cache_utils.cpp:279-301` | **DONE (by coincidence of a correct port)** — the key encoding already matches |
| 22 | **Hash agreement is ASSERTED**: `gpu_block_size % hash_block_size == 0`, with the comment "hash_block_size must match what the scheduler uses for `Request.block_hashes`" | `kv_offload/base.py:536-549,538`; `resolve_kv_cache_block_sizes` `vllm/v1/core/kv_cache_utils.py:607-670,646` | our pool carries `hash_block_size` but the differing-size path THROWS (`src/vllm/v1/core/block_pool.cpp:93-97`) | **PARTIAL** — blocks any `block_size_factor > 1` tier |
| 23 | Offload keys are the SAME `Request.block_hashes` prefix caching uses, strided by `hash_block_size_factor` | `offloading/scheduler.py:270-284,199-202`; `vllm/v1/request.py:179,237-240` | our `Request` carries the identical incremental `block_hashes` list (`src/vllm/v1/request.cpp:57,66,72,127,134,139-143`) | **DONE** — the identity that makes an offload tier possible at all |
| 24 | `LoadStoreSpec` + `medium()`; `BlockIDsLoadStoreSpec`; `GPULoadStoreSpec(block_ids, group_sizes, block_indices)` carrying the unaligned first/last offloaded block per group; `CPULoadStoreSpec` | `kv_offload/base.py:88-101,350-359,362-398,363-381`; `cpu/common.py:13-21` | absent | **MISSING** |
| 25 | `OffloadingManager` ABC — `lookup`, `prepare_load`, `touch`, `complete_load`, `prepare_store`, `complete_store`, `on_new_request`, `on_request_finished`, `take_events`, `on_schedule_end`, `has_pending_work`, `reset_cache`, `get_stats`, `shutdown` | `kv_offload/base.py:177-347` (prose contract `:119-143`) | absent | **MISSING** |
| 26 | `LookupResult` = `MISS` / `HIT` / `HIT_PENDING` / `RETRY`; `OffloadPolicy` = `BLOCK_LEVEL` / `REQUEST_LEVEL` | `kv_offload/base.py:56-62,65-71` | absent | **MISSING** |
| 27 | `PrepareStoreOutput(keys_to_store, store_spec, evicted_keys)`; **`prepare_store` returning `None` is a control-flow path**, not an error — eviction could not be satisfied, skip the store | `kv_offload/base.py:104-108`; `cpu/manager.py:169-237,192-194` | absent | **MISSING** — see §Risks R5 |
| 28 | `CPUOffloadingManager(num_blocks, cache_policy="lru"\|"arc", enable_events, store_threshold=1, max_tracker_size=64_000)`; free list + bump allocator; reuse counting in a capped `OrderedDict` | `cpu/manager.py:36,46-53,83-97,116-130,121-124` | absent | **MISSING** |
| 28b | **There is NO `LRUOffloadingManager` at this pin** (the record's row text implies one) — LRU/ARC are pluggable `CachePolicy` objects | `cpu/manager.py:30-33`; `cpu/policies/lru.py:12`, `arc.py:12` | — | record corrected here |
| 29 | `CachePolicy` ABC — `get`/`insert`/`remove`/`touch`/`evict(n, protected)`/`clear` + `mark_evictable`/`mark_non_evictable`. **`evict` is ATOMIC**: returns `None` and mutates nothing if `n` evictions are impossible | `cpu/policies/base.py:36-92,70-73` | our `BlockPool` free-queue eviction is LRU-by-ordering only, with no policy seam (`src/vllm/v1/core/kv_cache_utils.cpp:463-645`) | **MISSING** (policy seam) |
| 30 | `BlockStatus` is a `ctypes.Structure` `{ref_cnt: int32, block_id: int64}` where **`ref_cnt == -1` is the "not ready to read" sentinel**, `is_ready == (ref_cnt >= 0)` | `cpu/policies/base.py:10-33,20-25` | absent | **MISSING** — a tri-state a naive port collapses, see §Risks R5 |
| 31 | CPU tier backing store: pinned `torch.zeros`, or under tiering a shared `/dev/shm` mmap with `O_CREAT\|O_EXCL` creator election, `BLOCK_SIZE_ALIGNMENT = mmap.PAGESIZE`, per-worker strided views, and a zero-copy `memoryview` handed to secondary tiers | `cpu/gpu_worker.py:501-506`; `cpu/shared_offload_region.py:56,66-70,39,120-157,159-171,167` | absent | **MISSING** |
| 32 | **The disk (`fs`) tier** — `FileSystemTierManager` + `FsAsyncLookupManager` | `tiering/fs/manager.py:64-71,95-103,131-137,146-180,217` | absent | **MISSING** (`KV-OFFLOAD`) — the primary target of this spike |
| 33 | **On-disk format: ONE RAW FILE PER BLOCK, no container, no index.** `<root>/<safe_model_name>_<sha256[:12]>_r<rank>/<hhh>/<hh>_g<group_idx>/<hash_hex>.bin` | `file_mapper.py:112-120,15,128-139`; doc `docs/features/kv_offloading_usage.md:103-111` | absent | **MISSING** |
| 34 | Identity digest = canonical JSON over `{model_name, hash_block_size, gpu_blocks_per_file, tp_size, pp_size, pcp_size, dcp_size, dtype, kv_cache_groups, inference_engine}`, sha256 truncated to 12 hex | `file_mapper.py:49-60,128-139` | absent | **MISSING** — and it is WEAKER than it looks, see §Risks R3 |
| 35 | `config.json` is WRITTEN once if absent and **NEVER READ OR VALIDATED** — the only identity check is path-digest agreement | `file_mapper.py:16,122-126`; `tiering/fs/manager.py:131-137`; repo-wide, no reader exists | absent | **MISSING as upstream has it; we should EXCEED it** — §B2 |
| 36 | Byte path: `store_block` = existence-skip → `O_CREAT\|O_EXCL\|O_WRONLY\|O_TRUNC\|O_DIRECT` temp file → single `os.write` → `os.replace` (atomic publish); `load_block` = `O_RDONLY\|O_DIRECT` → `os.readv` → **`os.remove` the file on ANY failure** (self-healing) | `tiering/fs/io.py:32-72,42-43,53-57,59,66,75-101,87,88,92`; `O_DIRECT` Linux-only `:12` | absent — and we have NO byte writer anywhere in the tree (§Our baseline) | **MISSING** — but trivially portable, see §Port map |
| 37 | I/O threading: `DualQueueThreadPool` with `n_read_threads` + `n_write_threads`, each able to drain the other's deque so neither starves; completion via `JobState.task_done` → `_finished_q` | `tiering/fs/thread_pool.py:50-57,41-47,153-180,178` | absent | **MISSING** |
| 38 | Batched async lookup, preferring a C extension that releases the GIL for a whole `faccessat` batch | `tiering/fs/manager.py:64-71`; `csrc/fs_io.cpp:23,40` | N-A — no GIL to release; a C++ port is unconditionally at or above this | **N-A** (and a free win, §B1) |
| 39 | **The disk tier has NO capacity accounting and NO eviction** — files accumulate until reclaimed externally | `tiering/fs/manager.py` implements neither; contrast `cpu/manager.py:169-237` | absent | **MISSING as upstream has it; a named gap to EXCEED** — §B3 |
| 40 | `TieringOffloadingManager` — promotion (`lookup` returns `RETRY`, eager `prepare_write` marks the slot in-flight, deferred batched `submit_load` flushed at `on_schedule_end`), cascade on `complete_store`, and a `reset_cache` that drains secondaries FIRST and deliberately does NOT reset them (persistent stores survive) | `tiering/manager.py:123,238-280,282-329,331-353,408-459,498-556,461-495,202-235,643-681`; aliases `:74-120,100-103` | absent | **MISSING** — the eviction/promotion ORDERING, see §Risks R5 |
| 41 | GPU↔CPU transfer: batched raw-pointer DMA on a SIDE stream, one stream per transfer, serialized by events; pinned descriptor tensors; polled completion via `end_event.query()`, never blocking except `wait(job_ids)` | `cpu/gpu_worker.py:166-172,240-421,153-163,73-120,289-296,366-392,382,387,395-404,423-445,447-451` | absent — we have NO device↔host KV copy path at all (§Our baseline) | **MISSING** |
| 42 | **`is_src_access_order_any = not gpu_to_cpu`** — CPU→GPU may use `CU_MEMCPY_SRC_ACCESS_ORDER_ANY`; GPU→CPU MUST keep STREAM ordering because the compute stream is still writing the source. A correctness requirement, not a knob | `cpu/gpu_worker.py:388-394`; `csrc/libtorch_stable/cache_kernels.cu:139-142` | absent | **MISSING** — see §Risks R4 |
| 43 | Copy kernel selection: GPU→CPU always `ops.swap_blocks_batch` (copy engine beats Triton); Triton only if page size `< 28 KiB` and 8-byte aligned; the C++ op resolves **`cuMemcpyBatchAsync`** via `cuGetProcAddress` (CUDA >= 12.8) and falls back to a `cudaMemcpyAsync` loop | `cpu/gpu_worker.py:35-58,40,47-48,51-56,57`; `cpu/swap_blocks_triton.py:19-21,24-46,59-67`; `csrc/libtorch_stable/cache_kernels.cu:79,111-129,132-135,69-76,154-160` | absent; our raw primitives exist (`include/vt/backend.h:31,27-29,73-74,42,78-90`) | **MISSING** (primitives present) |
| 44 | Store DEFERRAL by one engine step — "so that offloading starts AFTER transfers related to token sampling, thereby avoiding delays to token generation" | `offloading/worker.py:294-302,296-298,283-286,273-276`; rationale `offloading_connector.py:115-117` | absent | **MISSING** — a latency-correctness rule we must mirror, not rediscover |
| 45 | Store completion is QUORUM-based: `pending_count = num_workers`, decremented by per-worker `completed_jobs`; `complete_store` fires exactly once after every rank reports | `offloading/scheduler.py:785,1132-1136,1140,1142` | N-A at world size 1, but the shape is owed | **MISSING** (degenerate today) |
| 46 | Config: everything offloading-specific lives inside `kv_connector_extra_config` — `spec_name`, `spec_module_path`, `cpu_bytes_to_use` (REQUIRED), `block_size`, `eviction_policy`, `store_threshold`, `max_tracker_size`, `offload_prompt_only` (default `true`), `self_describing_kv_events`, `secondary_tiers`. **No `num_cpu_blocks` flag** — count is derived | `factory.py:37,41-45,57-59`; `cpu/spec.py:57-61,106,114,117,73-98,26`; `kv_offload/base.py:521-523,512-514,554-566`; `tiering/spec.py:104-106,72,94-101,173-176`; doc `docs/features/kv_offloading_usage.md:64-82` | absent | **MISSING** |
| 47 | Secondary tier registry: `example`, `fs`, `p2p`, `obj`; `type` popped, rest passed as ctor kwargs | `tiering/factory.py:34-42,57-79` | absent | **MISSING** |
| 48 | Per-request selective offload: `max_offload_tokens` in `kv_transfer_params`; `0` disables, negative/non-int rejected with a warning | doc `docs/features/kv_offloading_usage.md:145-154`; test `offloading_connector/test_scheduler.py:1078` | absent | **MISSING** |
| 49 | Cross-process sharing REQUIRES a fixed `PYTHONHASHSEED` on every instance, else `NONE_HASH = os.urandom(32)` gives different filenames for identical content | `vllm/v1/core/kv_cache_utils.py:96-114,102,111-114`; doc `docs/features/kv_offloading_usage.md:115-121` | we have the same randomness and **NO escape hatch at all** | **MISSING, and we are WORSE than vLLM** — §Our baseline Correction 1 |
| 50 | `parallel_agnostic` folder collapsing, enabled only for a SINGLE `FullAttentionSpec` group, explicitly excluding MLA (latent KV is replicated per rank, never head-sharded) and the V2 runner | `file_mapper.py:46-48,85-96` | absent | **MISSING** — and the MLA exclusion is the upstream confirmation of §Risks R9 |
| 51 | `KVEventsConfig` + `ZmqEventPublisher` (reliable PUB/ROUTER, replay buffer, per-DP port offset); `BlockStored` carries `extra_keys`, `group_idx`, `kv_cache_spec_kind`, `kv_cache_spec_sliding_window` so external consumers can RECONSTRUCT block hashes | `vllm/config/kv_events.py:11,50`; `vllm/distributed/kv_events.py:25,36,48,92,107,111,115,202,237,268,278,305,505,518`; scheduler merge `scheduler.py:154,1791-1805` | inert placeholder — owned by `KV-EVENTS`, spiked in [prefix-prompt-caching-parity.md](prefix-prompt-caching-parity.md) | **MISSING** (cross-row dependency) |
| 52 | `self_describing_kv_events` — block-granular `BlockStored`/`BlockRemoved` from the offloading connector; full-attention groups only, sliding-window/SSM keep the placeholder; rejected by `TieringOffloadingSpec` | doc `docs/features/kv_offloading_usage.md:77`; `kv_offload/base.py:512-514`; `tiering/spec.py:94-101`; tests `offloading_connector/test_events.py:112,246,318` | absent | **MISSING** |
| 53 | `kv_load_failure_policy` = `recompute` \| `fail` (default **`fail`**); `get_block_ids_with_load_errors()` → `_handle_invalid_blocks` | `vllm/config/kv_transfer.py:70-74`; `base.py:375,383-392`; `scheduler.py:1529-1535` | absent | **MISSING** — the fail-safe default matters, §Risks R6 |
| 54 | `MultiConnector` — ordered fan-out with cross-child dedup; HMA only if EVERY child supports it | `multi_connector.py`; `factory.py:145` | absent | **MISSING** (`KV-CONNECTORS`) |
| 55 | `SimpleCPUOffloadConnector` + `vllm/v1/simple_kv_offload/{manager,metadata,worker}.py` — a minimal CPU-only alternative to the tiered path | `simple_cpu_offload_connector.py`; `simple_kv_offload/worker.py:198,204,269` | absent | **MISSING** — a smaller first port target than `OffloadingConnector` |
| 56 | `NixlConnector`/Pull/Push, `MooncakeConnector`, `MooncakeStoreConnector`, `MoRIIOConnector`, `HF3FSKVConnector`, `FlexKVConnectorV1`, `DecodeBenchConnector`, `ExampleHiddenStatesConnector` | `factory.py:176-238`; `nixl/` (14 files) | absent | **MISSING — NOT SCHEDULED** (`KV-CONNECTORS`); each needs an external RDMA/store dependency we do not have and cannot gate on this hardware |
| 57 | **`LMCacheConnectorV1`** — a thin dispatcher: `use_native` selects the in-tree `lmcache_integration` impl, default False selects the EXTERNAL `lmcache.integration.vllm` impl; every worker hook is pass-through; `get_num_new_matched_tokens` always returns `(n, False)` (never async); `update_state_after_alloc` DROPS the `blocks` argument | `lmcache_connector.py:83-115,120-228,128-134,223-228,259,281,230-254,303,342,34,74-81` | absent | **MISSING** (`KV-EXTERNAL-CACHE`) |
| 58 | **`LMCacheMPConnector`** (1226 lines) — the recommended standalone-server mode; still hard-imports `lmcache.integration.vllm.utils`, `lmcache.utils`, and `lmcache.v1.multiprocess.{mq,protocol,custom_types}` at module scope | `lmcache_mp_connector.py:1-50,11-13,26-49`; adapter `lmcache_integration/multi_process_adapter.py:1-20,11-18` | absent | **MISSING** |
| 59 | **In-tree vendored glue: `lmcache_integration/` ≈ 2396 lines** (`vllm_v1_adapter.py` 1429, `multi_process_adapter.py` 736, `utils.py` 211, `__init__.py` 20) — used as a fallback when the external package's own adapter is absent | `vllm/distributed/kv_transfer/kv_connector/v1/lmcache_integration/` | absent | **MISSING** — but see the trap in §"What LMCache actually requires" |
| 60 | LMCache is an EXTERNAL package: `lmcache >= 0.3.9` in an opt-in extras file that `setup.py`/`pyproject.toml` never reference | `requirements/kv_connectors.txt:1`; zero `lmcache` hits in `setup.py`/`pyproject.toml` | not installed in any environment on this project's boxes | **EXTERNAL — confirmed** |

#### What LMCache integration actually requires of a host engine

This is the load-bearing determination, and it is more constraining than the
`ROAD-V1-D4` row implies. vLLM vendors only the *vLLM-facing adapter*; the
vendored code still imports the external package at module scope
(`lmcache_integration/vllm_v1_adapter.py:11-35`, `utils.py:9-10,122`,
`multi_process_adapter.py:11-18`). Everything that actually stores a byte is
external: `LMCacheEngine`/`LMCacheEngineBuilder`
(`vllm_v1_adapter.py:16`), the three `VLLMPagedMemGPUConnectorV2` /
`VLLMPagedMemLayerwiseGPUConnector` / `VLLMBufferLayerwiseGPUConnector` classes
that touch paged memory (`:19-23`), the config schema (`:18`), the lookup and
offload servers (`:25-29`), and — for the MP mode — the ZMQ message queue, the
wire protocol and the **CUDA IPC** handle wrapper
(`multi_process_adapter.py:11-18`, `wrap_kv_caches` at `:23-25`).

What the adapter demands of the engine, each a hard constraint on us:

1. **A flat `slot = block_id * block_size + offset` paged layout**, computed by
   LMCache itself, not asked for (`vllm_v1_adapter.py:368-376`). Our
   `compute_slot_mapping` uses exactly this formula
   (`include/vllm/v1/worker/gpu/block_table.h:104-106`), so we agree — today.
2. **Block tables as flat block-id lists, KV cache group 0 ONLY**
   (`vllm_v1_adapter.py:175-188`). It runtime-sniffs the shape. For a hybrid
   model this silently ignores every non-zero group.
3. **Direct access to per-layer KV tensors off the attention layer objects**
   (`self.kv_caches[layer_name] = attn_layer.kv_cache`, `:781`) — it reaches
   past `register_kv_caches` into layer internals.
4. **Its own token→key mapping**, NOT vLLM block hashes: it injects multimodal
   hashes into the token id tensor (`:344-352`) and keys on the result. So
   LMCache does not consume our block hash at all in the V1 path — but the MP
   path DOES stride vLLM block hashes (`multi_process_adapter.py:28-39`, taking
   the LAST block hash of each chunk).
5. Layer count including MTP/draft layers (`:420,470,676,738`) and TP topology
   (`get_tensor_model_parallel_rank`, `get_tp_group`, `get_world_group`).
6. **Layerwise mode forces PIECEWISE cudagraphs**
   (`lmcache_connector.py:74-81`).

`tests/v1/kv_connector/unit/test_lmcache_integration.py` is explicitly an
interface-stability contract test, and its eight case names are effectively the
enumeration of vLLM internals LMCache depends on.

**Verdict.** There is no specified wire protocol to implement against. A C++
LMCache client means reimplementing an unpinned Python package's ZMQ message
types and CUDA-IPC handshake by reading its source, with no upstream test we
could port that does not itself `import lmcache`. That is the opposite of the
mechanical-port property this project exists to preserve. LMCache is therefore
scoped here as an **interop target reached over a ported connector seam**, not
as a from-scratch client, and it is sequenced last.

#### llama.cpp, revisited for the disk axis only

Settled in [prefix-prompt-caching-parity.md](prefix-prompt-caching-parity.md)
§Upstream chain and not re-derived: llama.cpp's `/slots/{id}?action=save|restore`
writes whole-sequence `GGSQ` v2 state files at ~8.2 KB/token, 49.9 ms save /
42.9 ms restore for 1745 tokens
(`tools/server/README.md:1069-1108`). It is strictly weaker than APC on every
reuse axis, but it is **strictly stronger on one**: the save is imperative,
named, synchronous and guaranteed, where every vLLM tier is a best-effort cache
subject to eviction. That single axis is §B4 below.

### Our baseline

Established by reading our source. **Two record claims are corrected, one of
them a correction to the caching spike that landed hours ago.**

**Correction 1 — the caching spike's §B2 is WRONG, and in our favour is exactly
backwards.** [prefix-prompt-caching-parity.md](prefix-prompt-caching-parity.md)
§B2 claims *"Ours is deterministic by construction"* and offers it as a
beyond-parity win over vLLM's `PYTHONHASHSEED` footgun; feature row #5 of that
table likewise records the chain seed as `DONE (and deterministic)`. Both are
false. `init_none_hash` takes an OPTIONAL seed and, when it is absent, fills
`NONE_HASH` with 32 bytes from `std::random_device`
(`src/vllm/v1/core/kv_cache_utils.cpp:307-319`, random branch `:312-318`) — a
faithful mirror of upstream's `os.urandom(32)`
(`vllm/v1/core/kv_cache_utils.py:111-112`). The single production caller passes
**no seed** (`src/vllm/entrypoints/model_loader.cpp:144`, inside
`EnsureNoneHash`), and every test does the same
(`tests/vllm/v1/test_scheduler.cpp:100,724`). Because hashes chain from
`NONE_HASH` (`kv_cache_utils.cpp:337-339,384`), **every block hash of every
request differs across processes.** We are not merely at parity with the
footgun — we are **worse**, because vLLM at least exposes `PYTHONHASHSEED` as an
escape hatch and logs a warning when it is unset
(`vllm/v1/core/kv_cache_utils.py:102-109`), while we expose neither. This is
spike-BLOCKING: block hashes are the file names of any content-addressed disk
tier, so a persisted cache would score a 0% hit rate on restart. The comment at
`model_loader.cpp:141-143` justifying the unseeded call ("prefix caching stays
inert for prompts shorter than a block") is true only for the sub-block case and
does not generalise. Fixing this is W1 and it is cheap: the seam already accepts
a seed.

**Correction 2 — `KV-OFFLOAD`'s row text names a class that does not exist.**
The row cites `vllm/v1/kv_offload/cpu/manager.py:36` alongside
`cpu/policies/lru.py:12` and `arc.py:12` as though LRU and ARC were manager
variants. At this pin they are pluggable `CachePolicy` objects behind one
`CPUOffloadingManager` (`cpu/manager.py:30-33`), and the row's scope ("CPU
tiering with LRU and ARC") omits the entire secondary-tier half — including the
filesystem tier that is the actual subject of the user's request. The row text
and anchors are corrected in this change; the state moves `INVENTORIED` →
`SPIKE`.

**What we have that is directly reusable.**

- **The key encoding already matches.** Our `BlockHashWithGroupId` packs a
  4-byte big-endian group id onto a 32-byte SHA-256 digest
  (`src/vllm/v1/core/kv_cache_utils.cpp:279-301`), byte-identical to upstream's
  `OffloadKey` (`vllm/v1/kv_offload/base.py:35-37`). Nothing needs to change.
- **`Request.block_hashes` is the same incremental list** the offload scheduler
  strides (`src/vllm/v1/request.cpp:57,66,72,127,134,139-143`).
- **The KV cache can already be host-resident.** `GPUModelRunner::CacheBuffer`
  is either device memory or a plain host `std::vector<uint8_t>`, selected by
  `VT_DEVICE_KV_CACHE` (`src/vllm/v1/worker/gpu/runner.cpp:344-369,513-517`).
  A host-resident buffer is directly `memcpy`-able, which gives W2 a
  GPU-free development and test path.
- **Transfer primitives exist**: `vt::Backend::Copy`, `Alloc`/`Free`/`Memset`,
  `AllocPinned`/`FreePinned`, `Synchronize`, events
  (`include/vt/backend.h:27-29,31,42,73-74,78-90`).
- **The accounting seams are pre-carved**: `ext_comp = num_external_computed_tokens`
  is already named in our KV manager
  (`include/vllm/v1/core/kv_cache_manager.h:31`), the external-computed-tokens
  allocation entry point is documented
  (`include/vllm/v1/core/single_type_kv_cache_manager.h:122`), and
  `SchedulerOutput` / `EngineCoreOutput` have documented empty slots for
  connector metadata and `kv_transfer_params`
  (`include/vllm/v1/core/sched/output.h:30-31`,
  `include/vllm/v1/engine/types.h:26,30`).
- **`BlockPool::cached_block_hash_to_block`** (`include/vllm/v1/core/block_pool.h:213`)
  is the hash→block map a restore path populates.

**What we do not have, verified.**

- `grep -ri "offload\|kv_transfer\|connector\|lmcache" src/ include/` returns 28
  hits, **27 of which are deferral comments** and one of which is executable:
  the `throw` in `BlockPool::evict_blocks`
  (`src/vllm/v1/core/block_pool.cpp:140-142`). Zero hits for `lmcache`.
- **No device↔host KV copy path at all.** Preemption is recompute-only:
  `Scheduler::preempt_request` frees the blocks and abandons their contents
  (`src/vllm/v1/core/sched/scheduler.cpp:102-113`).
- **No byte WRITER anywhere in the tree, and no shared byte-IO abstraction.**
  `grep -rn "ofstream\|fwrite\|O_WRONLY\|O_CREAT" src include tools` returns
  exactly one hit, in a dev harness (`tools/marlin/harness_main.cu:48`). Both
  model readers (`include/vllm/model_executor/model_loader/safetensors_reader.h:29-58`,
  `include/vllm/model_executor/model_loader/gguf_reader.h:72-90`) implement
  their own private read-only `mmap` and expose only typed views. The
  serialization layer is net-new work, not an extension. `safetensors_reader.h`
  is the cleaner template — it already has bounds-checked, overflow-guarded
  header parsing and a windowed page-release facility (`:24-28,69-84`).
- **Quantized KV cannot be persisted today**: every spec's
  `real_page_size_bytes()` throws when `kv_quant_mode != kNone`
  (`src/vllm/v1/kv_cache_interface.cpp:18-20,25,49,64,77`).

**The cache-shape facts a disk format must not assume.**

| Spec | Our page bytes | Anchor |
|---|---|---|
| `AttentionSpec` (base) | `2 * block_size * num_kv_heads * head_size * sizeof(dtype)` | `src/vllm/v1/kv_cache_interface.cpp:24-32` |
| `FullAttentionSpec` | `block_size * num_kv_heads * (head_size + head_size_v) * sizeof(dtype)` | `:48-58` |
| **`MLAAttentionSpec`** | `storage_block_size() * num_kv_heads * head_size * sizeof(dtype)` — **no factor 2, no separate V** | `:63-74` |
| `SlidingWindowSpec` | as full attention | `:76-86` |
| `ChunkedLocalAttentionSpec` | inherits the base symmetric formula | `include/vllm/v1/kv_cache_interface.h:275-296` |
| **`MambaSpec`** | `sum_i(prod(shapes[i]) * sizeof(dtypes[i]))` — SSM + conv state, **not a paged formula at all** | `:102-123` |

Full attention's physical page is `[K(bs,h,d) | V(bs,h,d)]` contiguous, and the
rank-4 view keeps a factor of 2 in the BLOCK stride
(`src/vllm/model_executor/models/qwen3_5.cpp:2408-2426`). MLA's view is
**rank-3** — `{num_blocks, block_size, head_size}` with `head_size = kv_lora_rank
+ qk_rope_head_dim = 576` and `num_kv_heads == 1`
(`src/vllm/model_executor/models/deepseek_v2.cpp:570-571`, asserted `:549-552`;
spec `include/vllm/v1/kv_cache_interface.h:189-238,225-232`; constructed
`src/vllm/model_executor/models/deepseek_v2_registry.cpp:145-163`). Upstream
independently confirms MLA is the exception by excluding it from
`parallel_agnostic` folder sharing (`file_mapper.py:88-96`).

**Sliding window has no production construction site** — `SlidingWindowSpec` is
registered (`src/vllm/v1/kv_cache_spec_registry.cpp:77`) and unit-tested but no
model builds one, so it is untested for persistence and must be treated as such.

**GDN/Mamba state is not addressed by block id at all.** Column 0 of a GDN
group's block-table row is REWRITTEN from a block id to a compact per-request
state slot keyed on `req_id`
(`GPUModelRunner::remap_gdn_state_slots`, `src/vllm/v1/worker/gpu/runner.cpp:602-653`).
Recurrent state is a different persistence problem from paged KV and is
explicitly out of scope for W2–W5.

**Model identity metadata available for a file header.** `HfConfig`
(`include/vllm/transformers_utils/hf_config.h:69-112`) carries `model_type`,
`architectures`, `num_hidden_layers`, `num_attention_heads`,
`num_key_value_heads`, `head_dim`, `sliding_window`, `layer_types`,
`rope_theta`/`rotary_dim`/`RopeParameters` (`:102-104`, struct `:22-57`),
`max_position_embeddings`, `torch_dtype`, and — critically — the entire
`nlohmann::json raw` (`:111`), which makes a digest over the untyped remainder
possible. **Three gaps must be planned around:** there is no quantization field
on `HfConfig` at all (scheme is inferred inside registries/loaders), no model
path or checkpoint hash is retained in any queryable struct (the path exists
only transiently at `include/vllm/entrypoints/model_loader.h:98` and
`include/vllm.h:87-101`), and KV storage dtype is an **env-var A/B**
(`include/vllm/v1/kv_cache_dtype.h:25-29`, `VT_KV_CACHE_F32`) rather than a
config field — so the authoritative value to record is `spec->dtype`, not the
env var.

**Where an API would attach.** The C ABI is `VLLM_ABI_VERSION 1`
(`include/vllm.h:44`) with `vllm_engine_load`/`vllm_complete`/`vllm_request_*`
(`:155-222`) — adding save/restore is an ABI bump. The HTTP server exposes
exactly five routes and **no admin surface**
(`src/vllm/entrypoints/openai/api_server.cpp:311-327`), but dispatch is
socket-free and unit-testable via `handle_*` returning `DispatchResult`
(`include/vllm/entrypoints/openai/api_server.h:69-90`), so a route is cheap to
add and test on CPU.

### Port map

| Upstream | Local target | Disposition |
|---|---|---|
| `vllm/v1/core/kv_cache_utils.py:96-114` seedable `NONE_HASH` + unset warning | `src/vllm/v1/core/kv_cache_utils.cpp:307-319` (seam already accepts a seed); caller `src/vllm/entrypoints/model_loader.cpp:140-146`; new CLI flag in `examples/server/main.cpp` | **W1** — mirror the env-var escape hatch and the warning; ours becomes deterministic-by-default only if we deliberately EXCEED upstream (§B1), which is a recorded deviation |
| `vllm/v1/kv_offload/base.py:27-47` `OffloadKey` | none — `src/vllm/v1/core/kv_cache_utils.cpp:279-301` already packs identically | **NO WORK** — verified equivalent |
| `vllm/v1/kv_offload/tiering/fs/io.py:32-101` `store_block`/`load_block` | new `src/vllm/v1/kv_offload/fs_io.cpp` + `include/vllm/v1/kv_offload/fs_io.h` | **direct port, and the cleanest in the whole surface** — `open`/`write`/`readv`/`rename`/`unlink` with `O_DIRECT`; nothing Python-specific. Also introduces the byte-writer the tree lacks |
| `vllm/v1/kv_offload/file_mapper.py:112-139` naming + digest | new `src/vllm/v1/kv_offload/file_mapper.cpp` | direct port of the layout; the digest field set is EXTENDED (§B2) and the deviation recorded |
| `vllm/v1/kv_offload/tiering/fs/thread_pool.py:50-180` dual-queue pool | `src/vllm/v1/kv_offload/fs_io.cpp` over `std::thread` + two deques | structural port; no GIL, so the C-extension batch-lookup fast path (`csrc/fs_io.cpp:23,40`) is unconditionally unnecessary (§B1) |
| `vllm/v1/kv_offload/cpu/policies/base.py:36-92` `CachePolicy` + `BlockStatus` tri-state | new `include/vllm/v1/kv_offload/cache_policy.h` | direct port; the `ref_cnt == -1` sentinel and the ATOMIC `evict` are both load-bearing (§Risks R5) |
| `vllm/v1/kv_offload/cpu/policies/{lru,arc}.py` | `src/vllm/v1/kv_offload/policies/` | direct port; ARC is 171 lines and self-contained |
| `vllm/v1/kv_offload/cpu/manager.py:36-309` `CPUOffloadingManager` | new `src/vllm/v1/kv_offload/cpu_manager.cpp` | direct port incl. the `prepare_store -> None` control path |
| `vllm/v1/kv_offload/tiering/manager.py:123-681` promotion/cascade/reset ordering | `src/vllm/v1/kv_offload/tiering_manager.cpp` | direct port — the ORDERING is the correctness content, not the data structures |
| `vllm/v1/kv_offload/cpu/gpu_worker.py:240-421` batched DMA on a side stream | `src/vllm/v1/kv_offload/gpu_worker.cu` over `vt::Backend::Copy` + events (`include/vt/backend.h:31,78-90`) | port the STRUCTURE (side stream, event chaining, polled completion); the `cuMemcpyBatchAsync` fast path is a later optimization, the `cudaMemcpyAsync` loop is the correct first implementation. The `is_src_access_order_any` asymmetry (`cpu/gpu_worker.py:388-394`) is mandatory |
| `vllm/distributed/kv_transfer/kv_connector/v1/base.py:171-703` the ABI | new `include/vllm/v1/kv_offload/kv_connector.h` — an ABSTRACT C++ BASE CLASS, 7 pure-virtual methods + defaulted hooks | **mirror the SEMANTICS, not the plugin mechanism** — see below |
| `factory.py:96-123` `kv_connector_module_path` dynamic Python import | **none** | **NOT PORTED** — a Python-import ABI has no C++ equivalent worth having; the C++ analogue is compile-time registration, mirroring our existing `REGISTER_VLLM_MODEL` seam. Recorded deviation |
| `vllm/v1/worker/gpu/kv_connector.py:61-107` `pre_forward`/`post_forward` | `src/vllm/v1/worker/gpu/runner.cpp` | direct port of the MRV2 hook points |
| `vllm/model_executor/layers/attention/kv_transfer_utils.py:15-57` layerwise decorator | `src/vllm/model_executor/models/*` attention call sites | **deferred to W5** — it forces PIECEWISE cudagraphs and would forfeit our measured decode-graph win (§Risks R8) |
| `vllm/v1/core/sched/scheduler.py:736-742,933-937,1118-1119,2340-2371` | `src/vllm/v1/core/sched/scheduler.cpp` | direct port; the `None` third state at `:745-751` is the trap |
| `vllm/v1/core/block_pool.py` `evict_blocks` | `src/vllm/v1/core/block_pool.cpp:139-143` (replace the throw) | direct port; owned jointly with `KV-BLOCK-POOL` |
| `example_connector.py:109-248` synchronous safetensors-per-layer disk connector | **none** | **NOT PORTED as a product** — it is a debug reference (`_generate_foldername_debug`); it is the READING oracle for W3's semantics, not a port target |
| `lmcache_integration/` (2396 lines) + `lmcache_connector.py` + `lmcache_mp_connector.py` | **none in W1–W5** | see §Risks R2; W6 is a compatibility study, not a port |
| `nixl/`, `mooncake/`, `moriio/`, `hf3fs/`, `flexkv_connector.py` | **none** | **NOT SCHEDULED** — each needs an external RDMA/store dependency absent from our boxes |

**What mirrors faithfully vs. what needs a C++-native equivalent.** The
distinction the user asked for, stated plainly:

*Load-bearing for correctness, must mirror exactly:* block identity (the
`OffloadKey` byte packing — already matching), **hash agreement** (the offload
tier and the prefix cache must key on the same `Request.block_hashes`, asserted
upstream at `kv_offload/base.py:536-549`), the tri-state `LookupResult` and the
`ref_cnt == -1` not-ready sentinel, the ATOMIC `evict` contract, the
`prepare_store -> None` control path, eviction/promotion ORDERING in
`TieringOffloadingManager` (eager `prepare_write` before a deferred `submit_load`
so a slot reads as in-flight within the same step), the GPU→CPU stream-ordering
requirement, the one-step store deferral, `handle_preemptions` firing before
blocks are overwritten, deferred block-free ownership after `request_finished`,
and the `fail`-by-default load-failure policy.

*Incidental to Python, replace with a native equivalent:* the dynamic
`importlib` connector-module path (→ compile-time registration), `ctypes.Structure`
for `BlockStatus` (→ a plain struct), the GIL-releasing `faccessat` C extension
(→ unnecessary), `msgspec` event structs (→ our existing JSON/CBOR encoders),
`torch.Tensor` as the KV handle (→ our `PagedKvCache` POD /
`CacheBuffer::data()`), and the two-parallel-worker-path split
(V1 mixin vs V2 `gpu/kv_connector.py`) — we port MRV2 only.

### Tests to port

Upstream suite at the pin. Counts are `^def test_` plus indented/async forms.

| Upstream test file | Cases | Tier | Disposition |
|---|---:|---|---|
| `tests/v1/kv_offload/tiering/test_fs_tier.py` | 12 | T-unit | **the executable spec for W3** — roundtrip, integrity, missing-file failure, partial-job failure, shutdown-discards-pending, idle-wait, batch-lookup dispatch |
| `tests/v1/kv_offload/test_file_mapper.py` | 8 | T-unit | **the executable spec for W3's naming/identity** — full path structure, run-config fields, config path, and the four `parallel_agnostic` gating cases incl. `test_parallel_agnostic_excludes_mla` |
| `tests/v1/kv_offload/cpu/test_manager.py` | 16 | T-unit | W2 — incl. `test_already_stored_block_not_evicted_during_prepare_store` (parametrized over policy), `test_prepare_load_preserves_key_order`, `TestARCPolicy`, `test_evictable_cache_block_count` |
| `tests/v1/kv_offload/test_factory.py` | 12 | T-unit | W2 config/registry; the `spec_module_path` dynamic-import cases (`:194,217`) are SKIP-marked with the recorded deviation |
| `tests/v1/kv_offload/tiering/test_tiering_offloading.py` | 25 | T-unit | W4 — promotion/cascade/reset ordering |
| `tests/v1/kv_offload/tiering/test_async_lookup.py` | 11 | T-unit | W4 |
| `tests/v1/kv_offload/tiering/test_factory.py` | 8 | T-unit | W4 tier registry |
| `tests/v1/kv_offload/cpu/test_shared_offload_region.py` | 27 | T-unit | W2 — SKIP the `/dev/shm` multi-worker half until multi-rank exists; the layout/stride/overflow cases port now |
| `tests/v1/kv_offload/cpu/test_gpu_worker.py` | 2 | T-e2e | W2 — `test_transfer`, `test_transfer_multi_group`; needs a GPU |
| `tests/v1/kv_offload/cpu/test_swap_blocks_batch.py` | 2 | T-e2e | W2 — default vs dedicated stream |
| `tests/v1/kv_offload/cpu/test_swap_blocks_triton.py` | 1 | — | **NOT PORTED** — Triton-specific fast path |
| `tests/v1/kv_offload/tiering/test_obj_tier.py` | 17 | — | **NOT PORTED** — object-store tier, no dependency |
| `tests/v1/kv_offload/tiering/p2p/*` (4 files) | ~50 | — | **NOT PORTED** — NIXL/ZMQ P2P tier |
| `tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py` | 62 | T-unit | **the largest and most important suite** — W4/W5; incl. `test_request_preemption:148`, `test_two_groups_full_and_sliding_window:413`, `test_two_groups_different_block_sizes:519`, `TestMaximalPrefixLookup:685`, `TestSlidingWindowLookup:748`, `test_fence_at_update_state_after_alloc:933`, `test_complete_store_called_per_job:1038`, `test_max_offload_tokens_validation:1078`, `test_offload_prompt_only:1172`, `test_reset_cache:1222`, `test_pending_transfer_defers_prefix_lookup:1357`, `test_swa_alignment_skip:1452`, `test_skip_reading_prefix_cache:1648`, `TestEagle:1701` |
| `tests/v1/kv_connector/unit/offloading_connector/test_worker.py` | 2 | T-unit | W5 `register_kv_caches` |
| `tests/v1/kv_connector/unit/offloading_connector/test_worker_metadata.py` | 4 | T-unit | W5 aggregation |
| `tests/v1/kv_connector/unit/offloading_connector/test_events.py` | 9 | T-unit | **SKIP to `KV-EVENTS`** — self-describing events need the event payload that row owns |
| `tests/v1/kv_connector/unit/offloading_connector/test_metrics.py` | 20 | T-unit | **SKIP to `SERVE-METRICS`** |
| `tests/v1/kv_connector/unit/test_offloading_connector.py` | 5 | T-e2e | **the binding e2e set** — `test_cpu_offloading:235`, `test_tiering_offloading:450`, **`test_fs_tiering_offloading:495`** (the disk gate), `test_mamba_align_cpu_offload:553` (hybrid; depends on `KV-MAMBA-ALIGN` W6) |
| `tests/v1/kv_connector/unit/test_kv_connector_lifecycle.py` | 1 | T-unit | W5 — metadata is cleared after every step |
| `tests/v1/kv_connector/unit/test_config.py` | 7 | T-unit | W5 `KVTransferConfig` validation |
| `tests/v1/kv_connector/unit/test_multi_connector.py` | 22 | T-unit | `test_multi_connector_overrides_all_base_methods:818` is portable NOW as an **interface-completeness oracle** over our abstract base; the rest SKIP to `KV-CONNECTORS` |
| `tests/v1/kv_connector/unit/test_example_connector.py` | 1 | T-unit | `test_shared_storage_connector_hashes:122` — the naming/hashing contract |
| `tests/v1/kv_connector/unit/test_kv_cache_layout.py` | 3 | T-unit | **directly binding on MLA** — `test_mla_common_backend_rejects_cross_layer_kv_cache:9` |
| `tests/v1/kv_connector/unit/test_error_propagation.py` | 2 | T-unit | W5 sync + async load failure |
| `tests/v1/kv_connector/unit/test_kv_load_failure_recovery.py` | 4 | T-unit | W5 `recompute` vs `fail` |
| `tests/v1/kv_connector/unit/test_invalid_blocks_correctness.py` | — | T-unit | W5 — the silent-wrong-output guard |
| `tests/v1/simple_kv_offload/test_scheduler.py` | 22 | T-unit | W2 alternative — the simpler CPU path |
| `tests/v1/simple_kv_offload/test_worker.py` | 3 | T-unit | W2 — incl. `test_store_orders_after_compute_write:106` and `test_build_params_src_access_order:171` (the §Risks R4 rule, tested) |
| `tests/v1/simple_kv_offload/test_integration.py` | 4 | T-e2e | W2 accuracy + latency |
| `tests/v1/kv_connector/unit/test_lmcache_connector.py` | 23 | — | **NOT PORTED in W1–W5** — SKIP-marked to `KV-EXTERNAL-CACHE` W6 |
| `tests/v1/kv_connector/unit/test_lmcache_integration.py` | 8 | — | **NOT PORTED** — it is an interface-stability contract for an external package; SKIP-marked to W6, retained as the enumeration of what LMCache demands |
| `tests/v1/kv_connector/unit/test_remote_prefill_lifecycle.py` / `test_remote_decode_lifecycle.py` | 9 / 4 | — | **NOT PORTED** — P/D disaggregation, `KV-CONNECTORS` |
| `tests/v1/kv_connector/unit/test_{nixl*,mooncake*,moriio*,hf3fs*,flexkv*,decode_bench,hidden_states,rixl*}.py` | — | — | **NOT PORTED** — external transports/stores |
| `tests/v1/core/test_deferred_block_free.py` | 11 | T-unit | already SKIP-marked to `KV-CONNECTORS` by the caching spike; W5 makes it live |
| `tests/v1/core/test_prefix_caching.py::test_cache_hit_local_and_external*` (`:3524,3645,3656,3682`) | 4 | T-unit | already SKIP-marked to `KV-EXTERNAL-CACHE`; W5 makes them live |

Per [test-porting.md](../test-porting.md):6 every case that cannot pass yet is
checked in SKIPPED with a tracked row reason, never dropped.

### Gates

1. **CPU (every work row):** clean warnings-as-errors build and full CTest; the
   ported cases green or explicitly SKIP-marked with a row reason. W1–W4 are
   developable and gateable entirely on the dev box using the host-resident KV
   path (`VT_DEVICE_KV_CACHE=0`, `src/vllm/v1/worker/gpu/runner.cpp:513-517`).
2. **Determinism (W1):** two independently launched processes over the same
   corpus emit **byte-identical block-hash lists**, asserted as a test, not
   observed. Without this no persisted-cache gate can pass, so W1 gates every
   row after it.
3. **Round-trip byte-exactness (W3):** a stored block reloaded into a fresh
   buffer `memcmp`s equal to the original, per spec kind — full attention, MLA
   (rank-3, no V), and a padded page — over the exact `page_size_bytes()` of
   each. A short read or a truncated file must FAIL loudly, never silently
   serve a partial block.
4. **Identity refusal (W3 — the safety gate):** a cache directory written under
   configuration A must be REFUSED, not silently used, when opened under a
   changed model, dtype, block size, quantization, rope/context config, KV-cache
   dtype, or spec kind. This is asserted per field, one negative test per field.
   A restore that proceeds against a mismatched identity is a **failing gate,
   never a warning**.
5. **Output invariance (W2, W3, W4 — DGX GB10):** token-exact on the affected
   model gates with offloading **ON**, **OFF**, and against the pinned vLLM
   oracle. An offload hit may change timing, never a token. This is a
   precondition that is never traded for speed.
6. **Cache-hit proof (W3, W4):** every benchmark arm reports queries/hits and a
   non-zero hit rate on a shared-prefix corpus, on BOTH engines. **This
   INHERITS the caching spike's W1 blocker** — we currently have no prefix-cache
   statistics at any level (`src/vllm/v1/core/kv_cache_manager.cpp:139` is a
   comment), so an arm without a hit counter is void and no offload benchmark
   can be run correctly until that lands.
7. **Every-axis performance (W4, W7 — DGX GB10):** match or beat vLLM on total
   and output throughput, req/s, TTFT, TPOT/ITL and peak memory at the
   large-concurrency operating point, per
   [benchmark-protocol.md](../benchmark-protocol.md), against vLLM launched with
   the equivalent `--kv-transfer-config`. An offload configuration that raises
   hit rate but regresses ANY axis is a failed change.
8. **No-regression when unused (every row):** offloading is opt-in and must be
   provably inert when off — a same-binary A/B with the feature disabled
   reproduces the prior binding numbers within run noise.
9. **Disk hygiene (W3, W4):** the dev box has ~16 GiB free and dgx ~184 GiB at
   95% utilization. Every disk-tier test must bound its `root_dir` and clean up;
   an unbounded tier filling the box presents as unrelated bogus test failures
   (the recorded ENOSPC lesson).
10. **Default-flip discipline:** per the standing parity-enabler rule, any lever
    a binding grid depends on has its default flipped BEFORE the binding run.

### Dependencies

- **Rows.** `KV-PREFIX-CACHE` W1 (prefix-cache statistics) is a **HARD BLOCKER**
  on gate 6 and therefore on W3's and W4's benchmark evidence — we cannot
  demonstrate a cache-hit win without it. `KV-BLOCK-POOL` owns
  `evict_blocks` (`src/vllm/v1/core/block_pool.cpp:139-143`) and the
  `block_size != hash_block_size` path (`:93-97`), both of which W4 needs.
  `KV-EVENTS` owns the event payload that `self_describing_kv_events` requires.
  `KV-MAMBA-ALIGN` gates the hybrid arm (`test_mamba_align_cpu_offload`).
  `SERVE-METRICS` owns the offload metric surface. `KV-MLA-SPEC` remains
  `INVENTORIED` and W3 must not assume it advances.
- **Ordering within this spike.** W1 (determinism) blocks W3 and everything
  after it — a content-addressed tier keyed on per-process-random hashes is
  worthless. W2 (CPU tier) blocks W3, because upstream's disk tier can only be
  reached through the CPU primary tier (`tiering/base.py:52-64`), and building
  the disk tier first would mean building a structure we then have to unpick.
- **Models.** W2/W3 need any dense model with APC on (Qwen3-dense, Qwen3-32B,
  Qwen3-Coder, OPT and DeepSeek-V2-Lite have all landed and default caching ON).
  W3's MLA round-trip case needs DeepSeek-V2-Lite. W4's hybrid arm needs 27B/35B
  and depends on `KV-MAMBA-ALIGN`.
- **Hardware.** W1, W3 (round-trip + identity), W5 and W6 are **CPU-only, no
  GPU** — pure behavioural ports gated by CTest on the dev box over the
  host-resident KV path. W2's transfer half, W4 and W7 need **DGX GB10
  (sm_121a)** under one `flock /tmp/gpu` per series on an idle box.
- **No new library dependency.** `O_DIRECT`, `pwrite`, `readv`, `rename`,
  `faccessat` and `std::thread` are all we need; JSON is already available via
  `nlohmann/json.hpp` (`include/vllm/transformers_utils/hf_config.h:14`) and
  SHA-256 is already ours (`src/vllm/v1/core/kv_cache_utils.cpp:97-160`). No
  io_uring, no external store client.
- **LMCache is NOT installed** in any environment on this project's boxes
  (`~/venvs` does not exist on the dev box; `lmcache` is an opt-in extras entry
  at `requirements/kv_connectors.txt:1` that `setup.py`/`pyproject.toml` never
  reference). W6 must budget for standing an LMCache server up before any
  interop claim is possible.

### Work breakdown

Ordered highest-value / lowest-risk first. Nothing below is claimed started;
every row is `open`.

| W | Deliverable | Gate | HW | State |
|---:|---|---|---|---|
| W1 | **Deterministic block hashes** — plumb the existing optional seed through `EnsureNoneHash` (`src/vllm/entrypoints/model_loader.cpp:140-146`), add the env/CLI escape hatch mirroring `PYTHONHASHSEED`, and the unset-warning we currently lack. **Corrects the caching spike's §B2, which claimed we were already deterministic** | gate 2 (two processes, byte-identical hash lists); CPU CTest | CPU | open |
| W2 | **CPU offload tier** — `CachePolicy` (+LRU, ARC), `BlockStatus` tri-state, `CPUOffloadingManager`, `LoadStoreSpec`/`OffloadKey`, and the GPU↔CPU batched side-stream transfer. Developable on the host-resident KV path | `test_manager.py` 16 + `test_factory.py` 12 + `simple_kv_offload` 25; then `test_gpu_worker.py` on DGX | CPU, then **DGX** | open |
| W3 | **Disk (`fs`) tier — the user's headline ask.** `store_block`/`load_block`, the dual-queue thread pool, `FileMapper` naming, and an EXTENDED verified identity header (§B2). Introduces the byte-writer the tree lacks | `test_fs_tier.py` 12 + `test_file_mapper.py` 8; gates 3 and **4 (identity refusal, per field)** | CPU | open |
| W4 | **Tiering manager + `OffloadingConnector` scheduler half** — promotion/cascade/reset ordering, `get_num_new_matched_tokens` incl. the `None` third state, `update_state_after_alloc`, `build_connector_meta`, `evict_blocks` (jointly with `KV-BLOCK-POOL`), one-step store deferral, `handle_preemptions` | `test_tiering_offloading.py` 25 + `offloading_connector/test_scheduler.py` 62; then `test_fs_tiering_offloading` e2e + gates 5/6/7 | CPU, then **DGX** | open |
| W5 | **The connector seam as a first-class C++ ABI** — abstract base with the 7 pure virtuals + defaulted hooks, compile-time registration, `KVTransferConfig`, `kv_load_failure_policy`, deferred block-free ownership, `SupportsHMA` multi-group finish. **Generalizes W4's one connector into the seam `KV-CONNECTORS` needs** | `test_multi_connector_overrides_all_base_methods` as a completeness oracle + `test_kv_connector_lifecycle` + `test_config` + the two failure-recovery suites; makes `test_deferred_block_free` and `test_cache_hit_local_and_external*` live | CPU | open |
| W6 | **LMCache interop STUDY, not a port** — stand an LMCache server up, read its `v1.multiprocess.{mq,protocol}` wire format, and produce a go/no-go with a cost estimate. Explicitly gated behind the two blockers we own: byte-compatible hashes (R1) and cross-process determinism (W1) | a written verdict with `file:line` on the external package; NO implementation claim | CPU | open |
| W7 | **B4 — imperative named per-sequence save/restore** (the one genuine beyond-parity item). Opt-in, sequenced LAST because it has no upstream oracle | §Risks B4 measurement plan; gate 4 identity refusal is MANDATORY here, not optional; inert-when-off A/B | **DGX** | open |
| — | NIXL / Mooncake / MoRI-IO / HF3FS / FlexKV / P/D disaggregation | **NOT SCHEDULED** — external transport dependencies absent from our boxes (`KV-CONNECTORS`) | — | dispositioned |
| — | Layerwise `save_kv_layer` pipelining | **NOT SCHEDULED in W1–W5** — forces PIECEWISE cudagraphs (§Risks R8) | — | dispositioned |

### Risks/decisions

**R1 — our block hashes are not byte-compatible with a stock vLLM, and this is
the gating constraint on ALL cross-engine interop.** We ship `sha256_cbor` while
upstream's DEFAULT is `sha256` over a pickle opcode stream, which is impractical
in C++ (recorded at [porting-inventory.md](../porting-inventory.md):76 and
[prefix-prompt-caching-parity.md](prefix-prompt-caching-parity.md) §R2). A
content-addressed tier shared with a stock vLLM would score **0% hits** unless
that vLLM is launched with `--prefix-caching-hash-algo sha256_cbor`. Decision:
this constraint is stated in the user-facing docs wherever interop is mentioned;
it does not block a vllm.cpp-only disk tier (our files, our keys), and it is a
precondition of W6. It also argues for writing `inference_engine: "vllm.cpp"`
into the identity digest (mirroring `file_mapper.py:57`) so our directories can
never be mistaken for vLLM's.

**R2 — LMCache has no portable contract, and the vendored code is a trap.**
Seeing 2396 lines of `lmcache_integration/` in-tree invites the conclusion that
LMCache is portable. It is not: every one of those files imports the external
package at module scope, and the actual storage engine, the paged-memory
connectors, the config schema, the ZMQ message queue and the CUDA-IPC handoff
are all outside the tree
(`lmcache_integration/vllm_v1_adapter.py:11-35`,
`multi_process_adapter.py:11-18`). There is no upstream test we could port that
does not itself `import lmcache`. Decision: LMCache is scoped as an interop
target over a ported seam (W6 = study, go/no-go), never as a from-scratch
client, and `ROAD-V1-D4` may not claim LMCache interop without also stating R1.

**R3 — upstream's identity check is weaker than it looks, and copying it
verbatim would be a silent-wrong-output hazard.** The `fs` tier's only identity
mechanism is a 12-hex path digest over
`{model_name, hash_block_size, gpu_blocks_per_file, tp/pp/pcp/dcp, dtype,
kv_cache_groups, inference_engine}` (`file_mapper.py:128-139`). `config.json` is
**written and never read** (`tiering/fs/manager.py:131-137`; no reader exists
anywhere). That digest does NOT cover: the checkpoint's actual content (two
different finetunes at the same HF path collide), the WEIGHT quantization scheme,
the rope/context configuration, `sliding_window`, or the KV-cache dtype
independently of `cache_dtype`. On our side the exposure is sharper still,
because our KV dtype is an env-var A/B (`include/vllm/v1/kv_cache_dtype.h:25-29`)
and `HfConfig` carries no quantization field at all. Loading a block written
under a different model or dtype produces plausible tokens that are simply
wrong — no crash, no warning. **Decision: we EXCEED upstream here (§B2). The
identity block is a VERIFIED header read on every open, not a path convention,
and a mismatch REFUSES the directory rather than warning.** Gate 4 makes this a
per-field negative test.

**R4 — `is_src_access_order_any` is a correctness requirement wearing the
costume of a tuning knob.** GPU→CPU copies must keep STREAM source-access
ordering because the compute stream is still writing the source buffer;
CPU→GPU may use `ANY` (`cpu/gpu_worker.py:388-394`,
`csrc/libtorch_stable/cache_kernels.cu:139-142`). Getting this backwards
produces torn blocks under load and nothing under a light test.
`tests/v1/simple_kv_offload/test_worker.py:171` tests it, and that case is
mandatory in W2.

**R5 — four upstream signatures that a competent port gets wrong.** Each is a
plausible-looking mistake: (a) `ref_cnt == -1` is a "not ready to read"
sentinel OVERLOADED onto the refcount field
(`cpu/policies/base.py:20-25`) — a port with a separate `bool ready` will
desynchronize; (b) `CachePolicy.evict` is ATOMIC, returning `None` and mutating
nothing when `n` evictions are impossible (`:70-73`) — a partial eviction is
wrong; (c) `prepare_store` returning `None` is a normal control path meaning
"skip the store", not an error (`cpu/manager.py:192-194`); (d)
`get_num_new_matched_tokens` has a THIRD state — `None` means "deschedule and
ask me again next step", handled at `scheduler.py:745-751` — and a port that
treats `None` as zero will spin or serve a partial prefix.

**R6 — `request_finished` transfers block-freeing OWNERSHIP.** Returning True
means the connector, not the scheduler, owns those blocks until `get_finished()`
reports the request id (`base.py:542-562`). Our scheduler frees blocks
immediately on preemption (`src/vllm/v1/core/sched/scheduler.cpp:102-113`) and
has no concept of deferred ownership. Combined with the absence of
`handle_preemptions` (#12), a naive W4 would let a preempted request's blocks be
overwritten while an async store is still reading them — a torn-block hazard
that presents as wrong tokens, not a crash. Decision: `handle_preemptions` and
deferred free land in the SAME row as the first async store (W4), never after
it, and the default `kv_load_failure_policy` is `fail`, mirroring upstream
(`vllm/config/kv_transfer.py:70-74`) — silent recompute hides exactly this class
of bug.

**R7 — non-HMA connectors assert a single KV cache group, and our gate models
are hybrid.** `scheduler.py:2371` routes to `request_finished_all_groups` only
for `SupportsHMA` connectors; the plain path asserts exactly one group and
passes `block_ids[0]`. Our 27B/35B are full-attention + GDN, i.e. two groups.
Decision: our connector base implements the multi-group finish from the start
rather than adding it later — the single-group form is a trap we would have to
unpick, and upstream's own `test_two_groups_full_and_sliding_window:413` /
`test_two_groups_different_block_sizes:519` are the spec.

**R8 — layerwise pipelining would forfeit a measured win.** The
`wait_for_layer_load`/`save_kv_layer` hooks are installed by a decorator on
attention layers (`kv_transfer_utils.py:15-57`) and force PIECEWISE cudagraph
mode (`lmcache_connector.py:74-81`, `vllm/config/vllm.py:1305`). We have a
measured decode-graph win on every gate model. Decision: W1–W5 use the
non-layerwise path (`start_load_kv` / `wait_for_save` at step granularity),
which is what `OffloadingConnector` itself uses; layerwise is dispositioned as
NOT SCHEDULED and revisited only if a measurement demands it.

**R9 — a disk format must not assume the K/V-pair shape.** MLA stores ONE 576-wide
latent per token with `num_kv_heads == 1` and **no V tensor**, viewed rank-3
(`src/vllm/model_executor/models/deepseek_v2.cpp:570-571`, asserted `:549-552`;
spec `src/vllm/v1/kv_cache_interface.cpp:63-74`), while full attention is rank-4
with an interleaved `[K|V]` page and a factor of 2 in the block stride
(`src/vllm/model_executor/models/qwen3_5.cpp:2408-2426`). Upstream independently
confirms MLA is the exception by excluding it from `parallel_agnostic` sharing
(`file_mapper.py:88-96`). **Decision: the on-disk block is OPAQUE BYTES of
exactly `page_size_bytes()`, and the spec kind plus every shape parameter is
recorded in the verified header (R3/§B2).** No layout interpretation happens at
the IO layer. This also covers `MambaSpec`, whose size is a sum over SSM and
conv state and is not a paged formula at all
(`src/vllm/v1/kv_cache_interface.cpp:102-123`) — recurrent state is explicitly
out of scope for W2–W5, since GDN state is addressed by a remapped compact slot,
not a block id (`src/vllm/v1/worker/gpu/runner.cpp:602-653`).

**R10 — sliding window is untested for persistence.** `SlidingWindowSpec` is
registered and unit-tested but has **no production construction site** in our
tree (`src/vllm/v1/kv_cache_spec_registry.cpp:77`; no model builds one). Upstream's
offload path has real sliding-window-specific behaviour — suffix-group lookup
(`offloading/scheduler.py:464`), `test_swa_alignment_skip:1452`,
`test_stale_sliding_window_block_after_prepare_store_failure:1573`, and
sliding-window groups keeping the placeholder event payload
(`test_events.py:246`). Decision: those cases are ported SKIP-marked to
`KV-SLIDING-WINDOW-SPEC`, and no W3/W4 claim covers sliding window until a model
constructs the spec.

**R11 — quantized KV cannot be persisted at all today.** Every spec's
`real_page_size_bytes()` throws when `kv_quant_mode != kNone`
(`src/vllm/v1/kv_cache_interface.cpp:18-20,25,49,64,77`). A disk tier therefore
covers unquantized KV only, and the header must record `kv_quant_mode` so that a
future FP8/NVFP4 KV cache cannot silently read bf16 files. Cross-referenced to
`KV-FP8` and `KV-NVFP4-TURBO`, both `INVENTORIED`.

**R12 — the disk tier has no eviction, upstream included.** `FileSystemTierManager`
implements neither capacity accounting nor `evict`; files accumulate until
something external reclaims them. With ~16 GiB free on the dev box and dgx at
95% utilization, a faithful port would fill the box and present as unrelated
bogus test failures. Decision: gate 9 bounds every test's `root_dir`, and §B3
adds the bounded-capacity behaviour upstream lacks.

#### The "and better" axes — concrete, with how each is measured

**B1 — no-GIL I/O, which is a structural advantage rather than a hypothesis.**
Upstream needs a dedicated C extension purely to release the GIL across a batch
of `faccessat` calls (`csrc/fs_io.cpp:23,40`) and tunes `DualQueueThreadPool`
around Python thread behaviour. A C++ port has no GIL, so batch lookup is an
ordinary parallel loop. *Measurement:* lookup throughput in blocks/s at fixed
directory fan-out, ours vs vLLM's C-extension path and vs its Python fallback,
on the same filesystem. Honest position: this is a real structural difference,
but lookup is unlikely to be the bottleneck against `O_DIRECT` reads — **if the
lookup share of restore wall-time is below ~2% the lever closes as NEUTRAL and
is recorded as such.**

**B2 — a VERIFIED identity header, where upstream has an unread `config.json`.**
Upstream writes `config.json` once and never reads it
(`file_mapper.py:122-126`, `tiering/fs/manager.py:131-137`), so its only
protection against loading a block written by a different model is a 12-hex path
digest over a field set that omits checkpoint content, weight quantization,
rope/context config and `sliding_window` (R3). We record a header that is READ
and CHECKED on every open, covering: `inference_engine` and format version;
`model_type` and `architectures`; a digest of `HfConfig::raw`
(`include/vllm/transformers_utils/hf_config.h:111`) so untyped fields are
covered too; `num_hidden_layers`, `num_kv_heads`, `head_size`, `head_size_v`;
the KV-cache spec KIND and `page_size_bytes()`; `block_size` and
`hash_block_size`; KV dtype (from `spec->dtype`, never the env var) and
`kv_quant_mode`; `sliding_window` and the full `RopeParameters`; the hash
algorithm name and the `NONE_HASH` value itself. *Measurement:* one negative
test per field asserting REFUSAL, plus a positive test asserting an unchanged
configuration loads. *Rule, non-negotiable:* a mismatch refuses; it never warns
and proceeds. **This is the safety property that makes the whole feature
shippable, and it is where "better than vLLM" is a correctness claim rather than
a speed claim.**

**B3 — a bounded, evicting disk tier.** Upstream's `fs` tier grows without
limit (R12). We add capacity accounting and an eviction policy over the same
`CachePolicy` seam the CPU tier uses (`cpu/policies/base.py:36-92`), configured
by a byte budget. Eviction is **semantically transparent** — dropping a cached
block loses a hit, never changes a token — which is what makes it safe to add
where upstream has nothing. *Measurement:* a fill-past-capacity test asserting
the budget is respected and hit rate degrades gracefully rather than the
filesystem filling; then the every-axis grid unchanged.

**B4 — imperative named per-sequence save/restore: the one thing neither vLLM
nor we have.** Per the caching spike, vLLM's reuse is implicit and
content-addressed; the only handle on a prefix is a derived hash that cannot be
named, pinned or exported, and the only imperative control is bulk
`reset_prefix_cache`. llama.cpp can be told "serialize slot 3 to `foo.bin`" and
later "load `foo.bin`". Proposal: an opt-in surface — a library call on the C++
engine (`include/vllm/v1/engine/llm_engine.h`), a C ABI pair behind a version
bump (`include/vllm.h:44`), and a dev-mode HTTP route added at
`src/vllm/entrypoints/openai/api_server.cpp:311-327` — that exports a request's
block list to a named file and re-imports it, giving snapshot-a-conversation and
session migration with APC-strength sharing rather than llama.cpp's
whole-sequence copy. *On-disk format:* the SAME verified header as B2, followed
by the ordered `(block_hash, page_bytes)` sequence — so a named snapshot and a
disk-tier directory validate through one code path and cannot disagree about
identity. *Safety rules, stated explicitly because this is the feature most
likely to produce corrupt output:* (i) the header is verified on every restore
and a mismatch refuses; (ii) the token id sequence is stored and re-hashed on
restore, and the recomputed chain must equal the stored hashes — a file whose
content does not hash to its own keys is rejected, which makes the format
self-validating independently of the header; (iii) restore populates the block
pool through the normal `cached_block_hash_to_block` path
(`include/vllm/v1/core/block_pool.h:213`) so refcounting and eviction stay
correct and a restored block is indistinguishable from a computed one; (iv)
restore is REFUSED, not silently partial, if any block fails to validate; (v)
the feature is opt-in and inert when off. *Measurement:* (a) restore-vs-recompute
TTFT at 8k+ prompt tokens, target >=5x faster than full prefill; (b) bytes/token
and save/restore milliseconds against llama.cpp's published floor (~8.2 KB/token,
49.9 ms save / 42.9 ms restore for 1745 tokens,
`tools/server/README.md:1069-1108`); (c) a token-exactness gate — a restored
continuation must equal the uninterrupted continuation; (d) an inert-when-off
A/B reproducing the binding grid. *Scope risk:* genuinely beyond vLLM, so there
is no oracle to mirror and no upstream test to port; it is therefore sequenced
LAST (W7), after the parity seam is closed and after the verified header (B2) it
depends on exists.

**B5 — deterministic cross-process hashes as a DEFAULT, not a documented
footgun.** Upstream requires operators to set `PYTHONHASHSEED` identically on
every instance or cross-process sharing silently produces zero hits, and says so
in its own documentation (`docs/features/kv_offloading_usage.md:115-121`). Once
W1 lands we can default to a deterministic seed and make the random behaviour
the opt-in, since our threat model (a local library and server) does not have
the hash-DoS concern Python's random seeding exists for. *Measurement:* two
independently launched processes over the same corpus score the full shared
prefix on a shared tier with ZERO configuration, while an unconfigured vLLM pair
scores 0%. **This is a correctness/usability win reported as a hit-rate number,
not a throughput claim — and note it is currently a DEFICIT, not an advantage
(§Our baseline Correction 1); B5 is the plan to turn it around.**
