# Scale-out / distributed execution — scope spike

Records-only W0 scope spike (NO build, NO GPU, NO download). Base `main` — see
the closing ledger/state entries for the exact SHA. Pinned vLLM oracle
`555967922` (0.26.0.dev0); vLLM paths below are `file:line` in that tree. Our
paths are `file:line` at the spike base. Owner claim: `CLAIM-SCALE-OUT-SPIKE`.

The engine is **single-GPU today**. This spike scopes the NEW capability
dimension of distributed / scale-out execution across three legs that share ONE
abstraction:

- **Leg 1 — multi-GPU** (tensor + pipeline parallel, intra-node NVLink/NCCL) —
  row `BACKEND-DISTRIBUTED-TP`, `BACKEND-DISTRIBUTED-PP`.
- **Leg 2 — multiple DGX Sparks** over the ConnectX-7 200GbE RoCE/RDMA cable —
  row `BACKEND-DISTRIBUTED-MULTINODE-SPARK`.
- **Leg 3 — MLX multi-node** over Thunderbolt (Apple-Silicon cluster) —
  row `BACKEND-DISTRIBUTED-MLX-RING`.

All three express TP/PP ONCE against a single `vt::` collective / process-group
abstraction, `BACKEND-DISTRIBUTED-COMM`, with backend-specific transports —
mirroring vLLM's `device_communicators` design.

## Scope

**In.** The 5 stable rows above. For each: the vLLM mechanism to mirror
(`file:line`), our exact seam (`file:line`), a reuse-vs-new estimate, and a
row-sized W-plan. The unifying `vt::` collective design (a `Communicator` bound
to a `Device`, sibling of `vt::Queue`, dispatched through `OpProvider`).

**Out.** Data parallel (DP) and expert-parallel-load-balancer (EPLB) beyond the
EP shard note (tracked separately in `feature-matrix.md` §3); KV-transfer
disaggregated prefill (`kv_transfer/`); elastic-EP. No implementation this pass.

**Supported modes / dispatch (target).** `world_size == 1` stays byte-identical
to today (every collective a no-op, exactly as vLLM's `GroupCoordinator` bypasses
when `world_size == 1`, `parallel_state.py:638` guard, and `PyNcclCommunicator`
self-disables at `pynccl.py:93`). A collective op is dispatched by `OpProvider`
keyed on `(OpId, DeviceType)`: NCCL provider on `kCUDA`, MLX-distributed provider
on `kMETAL`, a TCP/RDMA provider for the multi-Spark case.

## Upstream chain

### Leg 1 — tensor parallel (`BACKEND-DISTRIBUTED-TP`)

- **ColumnParallelLinear** `linear.py:418` — shards weight along output dim 0
  (`output_size_per_partition = divide(output_size, tp_size)` `linear.py:478`;
  param `"output_dim": 0` `linear.py:524`; loader narrows dim 0
  `linear.py:569-572`). Optional output all-gather `linear.py:599-601`
  (`tensor_model_parallel_all_gather`, only if `gather_output`).
- **MergedColumnParallelLinear** `linear.py:660` — same dim-0 sharding, per-shard
  offset/size divided by tp `linear.py:832-833`.
- **QKVParallelLinear** `linear.py:1021` — head sharding:
  `num_heads = divide(total_num_heads, tp_size)` `linear.py:1074`; KV heads
  `linear.py:1076-1079` (replicate to 1 when `tp_size > total_num_kv_heads`);
  `gather_output=False` `linear.py:1093` (Q/K/V stay sharded).
- **RowParallelLinear** `linear.py:1612` — shards input dim 1
  (`input_size_per_partition` `linear.py:1665`; param `"input_dim": 1`
  `linear.py:1708`; loader narrows `linear.py:1725-1728`). Partial-product
  **all-reduce** `linear.py:1765-1766` (`tensor_model_parallel_all_reduce`, if
  `reduce_results`).
- **VocabParallelEmbedding** `vocab_parallel_embedding.py:198` — vocab rows split
  (`num_embeddings_per_partition` `:302`; masked embed `:475-489`; combine
  `tensor_model_parallel_all_reduce` `:491`). **ParallelLMHead** `:505`; logits
  gathered in `logits_processor.py::_gather_logits:85` (all-gather `:93` or gather
  `:96`), trimmed to `org_vocab_size` `:152`.
- **Attention head split** — `llama.py:140-158`
  (`num_heads = total_num_heads // tp_size` `:143`,
  `num_kv_heads = max(1, total_num_kv_heads // tp_size)` `:153`).
- **MoE expert sharding** — `fused_moe/expert_map_manager.py::determine_expert_map:22`
  (`local_num_experts` split over `ep_size` `:67-69`, `-1` for non-local `:72-87`);
  TP-vs-EP decision `fused_moe/config.py::FusedMoEParallelConfig:1032`, `use_ep`
  `:1204-1205`; `RoutedExperts` (formerly `FusedMoE`) `routed_experts.py:44`,
  `local_num_experts` `:234`. All-to-all dispatch/combine on the device
  communicator (`base_device_communicator.py` `dispatch:91/379`, `combine:140/398`);
  trailing TP all-reduce gated by `skip_final_all_reduce`
  (`config.py:1308`, `layer.py:229-230`).

### Leg 1 — pipeline parallel (`BACKEND-DISTRIBUTED-PP`)

- **`PPMissingLayer(torch.nn.Identity)`** `models/utils.py:785` (pass-through
  forward `:793-795`); **`make_layers`** `:798` builds
  `[PPMissingLayer]*start + real[start:end] + [PPMissingLayer]*(N-end)` `:822-828`
  from **`get_pp_indices`** `distributed/utils.py:127` (env override
  `VLLM_PP_LAYER_PARTITION` `:143-154`).
- **IntermediateTensors flow** — first rank embeds, non-first reads
  `intermediate_tensors["hidden_states"|"residual"]` `llama.py:414-417`; owned
  layers only `islice(..., start_layer, end_layer)` `:420-425`; non-last returns
  `IntermediateTensors` `:430-433`.
- **Inter-stage send/recv** — `GroupCoordinator.send_tensor_dict:957` →
  `device_communicator.send_tensor_dict` `:1010-1013`; `recv_tensor_dict:1052` →
  `:1108-1111`; single-tensor `send:1185`/`recv:1192`. Reached via
  `get_pp_group()`.

### The comm abstraction to mirror (`BACKEND-DISTRIBUTED-COMM`)

- **`GroupCoordinator`** `parallel_state.py:358` — holds
  `device_communicator: DeviceCommunicatorBase | None` `:384`, created only when
  `world_size > 1` `:479-489`. Every collective bypasses at `world_size == 1`,
  else forwards to the device communicator: `all_reduce:638`→`:665`,
  `all_gather:667`→`:686`, `send:1185`→`:1190`, `recv:1192`→`:1199`,
  `dispatch:1249`/`combine:1263`. Module-level custom-op wrappers `all_reduce:130`
  / `all_gather:160` make it Dynamo-traceable.
- **`DeviceCommunicatorBase`** `device_communicators/base_device_communicator.py:147`
  — the backend-plural interface: `all_reduce:215`, `all_gather:219`,
  `reduce_scatter:252`, `gather:290`, `send:321`, `recv:328`, `broadcast:340`,
  `dispatch:379`/`combine:398`. **This is the port template for `vt::Communicator`.**
- Concrete backends prove the plurality: `cuda_communicator.py:29`,
  `cpu_communicator.py:20`, `xpu_communicator.py:16`.
- **CUDA transport** — `CudaCommunicator.all_reduce:273-339` dispatch order
  (custom-AR `:311-319` before pynccl `:325-331`); `PyNcclCommunicator`
  `pynccl.py:60` (NCCL comm from a broadcast `unique_id`
  `ncclCommInitRank:136-139`; `all_reduce:166`, `send:322`, `recv:349`);
  intra-node NVLink `CustomAllreduce` `custom_all_reduce.py:51` (disabled across
  nodes `:90-96`, world sizes `[2,4,6,8]` `:52`).

### Leg 2 — multi-node executor (`BACKEND-DISTRIBUTED-MULTINODE-SPARK`)

- **Executor factory** — `v1/executor/abstract.py::Executor:37`, `get_class:47-92`
  (`"ray"`/`"mp"`/`"uni"`/`"external_launcher"`); the worker interface is
  `collective_rpc:152-202`.
- **MultiprocExecutor** `multiproc_executor.py:103` (`supports_pp=True`) — one proc
  per local GPU (`for local_rank in range(local_world_size)` `:176`,
  `global_rank = global_start_rank + local_rank` `:177`); shared-memory
  `MessageQueue` RPC broadcast `:135-157,388`; `world_size == tp*pp*pcp` `:117-123`.
  For CUDA `nnodes > 1` vLLM defaults to `"mp"` per node
  (`config/parallel.py:912-913`), coordinated over a TCP store.
- **RayDistributedExecutor** `ray_executor.py:64` — cross-**node** placement:
  bundle→worker `:181-187`, `PlacementGroupSchedulingStrategy` per bundle
  `:192-196`, node IP gather + re-rank `:217-259`, multi-node IP uniqueness check
  (`VLLM_HOST_IP`) `:291-303`, multi-node distributed init `driver_ip + open_port`
  `:339-341`; PP-across-nodes compiled DAG `:527-620`.
  `initialize_ray_cluster` `ray_utils.py:528`.
- **Cross-host rendezvous** — `init_distributed_environment`
  `parallel_state.py:1560`; `nnodes > 1` branch uses
  `master_addr`/`master_port` `:1597-1600`, then
  `torch.distributed.init_process_group(backend="nccl", init_method=tcp://...)`
  `:1653-1659`. `get_distributed_init_method` → `tcp://ip:port`
  `utils/network_utils.py:130-138`; `VLLM_HOST_IP` honored in `get_ip:33-72`.
  Config `config/parallel.py`: `master_addr:270`, `master_port:273`,
  `node_rank:276`.
- **RDMA NIC selection** — `v1/executor/vllm_net_devices.py` (GPU→NIC map for
  UCX/NVSHMEM/NCCL, `NCCL_IB_HCA`/`UCX_NET_DEVICES` `:12`, PCI-BDF→`mlx5_*` `:88`),
  invoked at worker startup (`uniproc_executor.py:60`, `multiproc_executor.py:854`);
  env `VLLM_GPU_NIC_PCIE_MAPPING` `envs.py:2016-2021`. NCCL picks IB/RoCE itself.

### Leg 3 — MLX distributed (`BACKEND-DISTRIBUTED-MLX-RING`)

Grounded in the MLX docs (mlx 0.30/0.32) + mlx-lm, NOT in the vLLM tree:

- **`mlx.core.distributed.init(backend=...)`** — backends
  `{'any','ring','jaccl','mpi','nccl'}`; `Group` with `rank()`/`size()`.
- Collectives: `all_sum` (= all-reduce), `all_gather`, `send`, `recv`,
  `recv_like`, `sum_scatter`, `all_min`/`all_max`.
- **Ring backend** = TCP sockets, with a Thunderbolt-RDMA mode for low-latency
  Mac-to-Mac; **JACCL** = Apple's Thunderbolt-5 RDMA collective library
  (fully-connected mesh). Measured (Apple, 2 GB all-reduce): 1.3 s Ethernet →
  0.45 s MPI → 0.27 s ring. Setup via `mlx.distributed_config` (hostfile from
  `en0` IPs for Ethernet; per-link isolated nets for Thunderbolt).
- **mlx-lm sharding** — `--num-shards` (mlx-lm ≥ 0.21): default TP across attention
  heads + FFN for dense, expert-parallel for MoE — the SAME sharding split as
  vLLM's TP/EP, so the Leg-1 loader/forward seams are reused for the MLX path.

## Our baseline (honest gaps)

**There is NO collective / NCCL / process-group / communicator code in the tree.**
Whole-tree grep for `nccl|all_reduce|all_gather|process_group|collective_rpc`
returns only comments and unrelated intra-warp/bias-broadcast hits. What exists:

- **The executor is a direct single-worker call** — `v1/executor/executor.cpp:7-34`
  (every method comment names the `collective_rpc(...)` it collapsed);
  `include/vllm/v1/executor/executor.h:5-28` states the whole
  `collective_rpc`/`WorkerWrapperBase`/multiproc/Ray/DP machinery collapses at T0
  to a direct virtual call. **This is the multi-worker fan-out seam.**
- **Latent multi-rank scaffolding, all inert (fixed to 1/0)** —
  `dcp_world_size`/`pcp_world_size` threaded through
  `v1/core/kv_cache_coordinator.cpp:81-92,300-304` and
  `v1/core/single_type_kv_cache_manager.cpp`, and
  `v1/worker/gpu/block_table.cpp:30-32`
  (`total_cp_world_size_(1), total_cp_rank_(0)` — comment: "No dcp/pcp process
  groups at T0 -> world size 1, rank 0"). `scheduler.cpp:86-87` passes
  `dcp_world_size=1, pcp_world_size=1`. KV-offload cache keys already carry
  `rank`/`tp_size`/`world_size` for namespacing
  (`v1/kv_offload/cache_identity.cpp:57-192`,
  `lmcache/cache_engine_key.cpp:47-72`) — persistence keys, not live collectives.
- **`src/vllm/distributed/` holds ONE file** — `kv_events.cpp` (KV event publishing,
  not a collective).

## Port map (upstream file -> local file) + the reuse-vs-new seam map

The `vt::` op signature is **already collective-ready**: every op is
`void(Queue&, ...)` (`include/vt/ops.h:561`) and `Queue` carries a `Device`
(`include/vt/device.h:50-54`). The two structural blockers are both
single-device-by-construction.

| Seam | vLLM anchor | Our seam (`file:line`) | Reuse vs new |
|---|---|---|---|
| Collective abstraction (`vt::Communicator`/process-group) | `base_device_communicator.py:147`; `parallel_state.py:358` | NEW sibling of `vt::Queue` (`include/vt/device.h:131`); new `OpId::kAllReduce/kAllGather/kSend/kRecv` via `OpProvider` (`include/vt/op_provider.h:108`); stream-order hooks EXIST (`include/vt/backend.h:87-104` `RecordEvent`/`QueueWaitEvent`) | ~90% new; the dispatch table + stream-event ordering are reused |
| Multi-device backend registry | executor spawns 1 worker/GPU (`multiproc_executor.py:176`) | `src/vt/backend.cpp:42-77` is ONE `Backend*` per DeviceType; CUDA registrar hardcodes device 0 (`cuda_backend.cu:297-299`, no `cudaSetDevice`). `DeviceResourceOps` free-fns (`backend.h:140-155`) are the index-aware replacement, but `Backend::{Alloc,CreateQueue}` are "index-0 migration shims" (`backend.cpp:83,101`) | ~60% new (per-index backends); the index-aware free-fn layer is half-built |
| TP-sharded linears + all-reduce | `linear.py:418/1612/1766` | dense forward already tags Column/Row parallel: QKV `dense_attn_block.h:342-378`, o_proj RowParallel `:513-530` (all-reduce goes right after `:530`), MLP gate_up/down `qwen3.cpp:83-90` (all-reduce after `:90`) | ~70% new logic, but insertion points are UNAMBIGUOUS (structure preserved) |
| TP-sharded weight load | `linear.py:569-572` (col), `:1725-1728` (row) | SINGLE chokepoint `include/vllm/model_executor/models/dense_weight_loaders.h:131-138` (the merged-weight memcpy); MoE per-expert loop `qwen3_moe_weights.cpp:35-38` | ~50% new; one header slices row/col ranges, one loop selects the EP expert subset |
| Attention head split | `llama.py:143-153` | `dense_attn_block.h:317-320,383-384` (`Hq`/`Hkv`/`Dh` from config) | ~40% new (divide by tp_size) |
| MoE expert shard + combine | `expert_map_manager.py:22`; `combine` `base_device_communicator.py:398` | router `qwen3_5.cpp:4440`, grouped GEMM `:4513`, `vt::MoeCombine`/`OpId::kMoeCombine` (`include/vt/ops.h:104`) becomes reduce-scatter/all-reduce | ~60% new |
| Sharded KV | (heads/tp) | `runner.cpp:403 initialize_kv_cache`, per-layer alloc `:437-461`, `Hkv`/`Dh` from spec `:526-527`, backing `vt::Alloc(device_)` `:385`; CP fields pre-wired `block_table.cpp:30-32` | ~40% new (divide `Hkv` by tp in the spec) |
| PP stage split + send/recv | `utils.py:785,798`; `parallel_state.py:957` | NEW: `PPMissingLayer` analogue in `make_layers`-style construction; the model forward already returns per-layer state; send/recv are `vt::Communicator` point-to-point | ~80% new |
| Multi-worker executor | `multiproc_executor.py:103`; `ray_executor.py:64` | `v1/executor/executor.cpp:7-34` direct call → fan-out to N rank workers; header seam `executor.h:5-28` | ~85% new |
| CUDA/NCCL transport | `pynccl.py:60`; `custom_all_reduce.py:51` | NEW provider TU on `kCUDA`, comm per-`CudaBackend`, collectives on `AsStream(q)` (`cuda_backend.cu:52`) ordered via events `:125-163` | ~95% new (thin ctypes-free NCCL wrapper) |
| RDMA/TCP transport (Sparks) | `init_distributed_environment` `parallel_state.py:1560`; `vllm_net_devices.py` | REUSES the CUDA/NCCL provider (NCCL runs over IB/RoCE on ConnectX-7); NEW = the cross-host rendezvous (TCP store on `master_addr:port`) + NIC pinning | ~30% new on top of Leg 1 (transport swap, same collectives) |
| MLX-ring transport (Thunderbolt) | `mlx.core.distributed` | NEW provider TU on `kMETAL` beside `metal_mlx_provider.mm:162` (`mx::default_stream(gpu)`); reuse the zero-copy `MTLBuffer↔mx::array` bridge (`metal_mlx_provider.mm:24-46,101-118`) for comm buffers | ~50% new (MLX ships the collectives; we wrap + register) |

## The unifying `vt::` collective design

ONE abstraction, three transports — mirroring vLLM's `device_communicators`:

- **`vt::Communicator`** (a.k.a. process-group): bound to a `vt::Device`, a sibling
  of `vt::Queue` (`include/vt/device.h:131`). Carries `rank()`/`world_size()`
  (mirror `GroupCoordinator`/`DeviceCommunicatorBase`). `world_size == 1` ⇒ every
  method is a no-op (byte-identical single-GPU path).
- **Collective ops** as new `OpId`s (`kAllReduce`, `kAllGather`, `kReduceScatter`,
  `kSend`, `kRecv`, `kBroadcast`), dispatched by `OpProvider`
  (`include/vt/op_provider.h:108`) keyed on `(OpId, DeviceType)`. Stream ordering
  against compute uses the existing `Backend::RecordEvent`/`QueueWaitEvent`
  (`include/vt/backend.h:87-104`).
- **Transports (providers), one per backend:** NCCL on `kCUDA` (intra-node NVLink
  custom-AR + pynccl-equivalent; also carries multi-Spark over IB/RoCE),
  TCP/RDMA rendezvous for the multi-node case, MLX-ring/JACCL on `kMETAL`
  (Thunderbolt).
- The model forwards + weight loader express **TP/PP ONCE** against
  `vt::Communicator`; each backend only supplies the transport provider. This is
  the PR-#4 additive story for a new interconnect: a new transport is one provider
  TU, not a model-forward edit.

## Tests to port

| Upstream test | Local tier | Note |
|---|---|---|
| `tests/distributed/test_comm_ops.py` (all-reduce/all-gather correctness) | T2 unit (`tests/vt/test_collective_*.cpp`) | needs a ≥2-rank harness; CPU gloo-style or a 2-proc loopback first |
| `tests/distributed/test_pynccl.py` | T2 | NCCL comm-init + all-reduce; GPU-gated, ≥2 GPUs |
| TP-sharded-linear equivalence (sharded ≡ single-GPU) | T2 parity | reuse the existing dense golden; assert sharded forward == unsharded token-exact |
| PP two-stage equivalence | T2 parity | assert 2-stage output == 1-stage |
| Multi-node smoke (2× Spark) | T3 e2e | HW-gated; requires the interconnect |
| MLX ring all-reduce | T3 | HW-gated; ≥2 Macs over Thunderbolt |

## Gates

- **W1 (blocking, HW-gated):** a 2-GPU box is required to gate TP — GB10 is
  single-GPU, so Leg 1 correctness gating waits on a multi-GPU host (per the
  existing feature-matrix §3 caveat). Correctness bar: sharded forward
  **token-exact** vs the single-GPU golden.
- **Leg 2 payoff gate:** run **`nvidia/DeepSeek-V4-Flash` fp8 (~167 GiB)** across
  **2× DGX Spark** (2 × 119 GiB unified = 238 GiB) — it does NOT fit one Spark's
  119 GiB pool but fits two. Confirmed-vs-assumed facts below.
- **Leg 3 gate:** MLX-ring all-reduce over Thunderbolt on a 2-Mac cluster, MLX-LM
  `--num-shards` as the competitor arm.
- No speed claim is owed at spike (`benchmark_binding=false`).

## Leg 2 — GB10 / DGX Spark interconnect facts (confirmed vs assumed)

**Confirmed (NVIDIA + independent measurement):**
- DGX Spark = GB10, **128 GB LPDDR5X unified**, dual-port **ConnectX-7** NIC,
  **200GbE** per QSFP port, sold as a 2-unit bundle with a QSFP cable for
  2-node clustering.
- The link is **RoCE (RDMA over Converged Ethernet)**; NCCL runs over it as
  `NET/IB` (not TCP fallback). Independently measured on 2 Sparks: raw RDMA
  `ib_write_bw` **~24.6 GB/s**; **NCCL all-reduce ~10.2 GB/s**, sendrecv
  **~9 GB/s** (GPUDirect-RDMA off in that test — a tuning ceiling, not the floor).
- Each 200Gb port is two PCIe-5.0 x4 links presented as a pair of Linux net
  interfaces with matching RoCE devices.

**Assumed / project-internal (NOT independently re-verified here):**
- Usable unified pool ≈ **119 GiB** per Spark (project memory note; the 128 GB is
  shared host+GPU, so `gpu_memory_utilization` reserves host RAM — the OOM-reboot
  hazard is on record).
- **The payoff:** DeepSeek-V4-Flash **native fp8 ≈ 167 GiB** exceeds one 119 GiB
  Spark but fits **2× 119 = 238 GiB** across the cable — this is the "DSpark"
  build's intended target. (The NVFP4 W4 build ≈ 83 GiB already fits ONE Spark;
  see `specs/deepseek-v4-flash.md`. So the 2-Spark path is specifically what
  unlocks the fp8 tier and larger models.) The exact DeepSeek-V4 fp8 footprint and
  the TP-vs-PP split that fits 238 GiB are **assumed pending the W1 oracle run**.
- That vLLM's Ray/mp multi-node path runs **as-is** on the Spark pair is
  **assumed** (it is the standard NCCL-over-IB path; nothing Spark-specific blocks
  it, but it has not been run here).

## Dependencies

- A **≥2-GPU host** (Leg 1) — absent locally (GB10 single-GPU); the cluster nodes
  (Thor sm_110, Orin sm_87, DGX sm_121) are single-GPU each. TP gating is
  HW-blocked until a multi-GPU box or the 2-Spark cable exists.
- A **NCCL-equivalent** collective lib for CUDA (or a from-scratch thin wrapper —
  MIRROR policy prefers wrapping the same NCCL vLLM uses).
- **MLX with `mlx.core.distributed`** for Leg 3 (ships in MLX ≥ 0.30).
- Two DGX Sparks + the QSFP cable (Leg 2 e2e), or two Thunderbolt-linked Macs
  (Leg 3 e2e).

## Non-overlapping work breakdown (W-plan)

- **W0 (this spike):** scope + seam map + rows + records. DONE.
- **W1 — `vt::Communicator` skeleton (`BACKEND-DISTRIBUTED-COMM`): DONE
  (2026-07-28, CPU exact-gate, `CLAIM-SCALE-OUT-W1`).** Landed the process-group
  interface (`include/vt/communicator.h`: `rank()`/`world_size()` +
  `AllReduce(sum/max/min/prod)`/`AllGather`/`Send`/`Recv`, each stream-ordered on a
  `Queue&`, ported 1:1 from `DeviceCommunicatorBase base_device_communicator.py:147`
  and the `world_size==1` bypass `parallel_state.py:638`) and a CPU **in-process
  multi-rank** transport (`src/vt/communicator.cpp`: N ranks = N host threads over
  ONE generation-barrier + staging slots + a Send/Recv rendezvous mailbox — a REAL
  cross-rank reduction, no IPC). Chose in-process multi-rank over a 2-proc loopback
  (cleanest correctness gate, needs no IPC, and sidesteps the W2 per-device
  registry blocker entirely). **Direct `Communicator` methods, NOT `OpId`
  dispatch** — routing collectives through `OpProvider`/new `OpId`s
  (`kAllReduce`/…) is deferred to W2 (a direct method is the cleaner W1 gate, and
  the enum stays unpolluted by unregistered ops). Gate `tests/vt/test_communicator.cpp`:
  **8 cases / 50 assertions PASS** — 2- & 4-rank AllReduce-sum exact on every rank
  (f32/i32/i64/bf16), max/min, 2/4-rank AllGather concat, Send/Recv rendezvous, and
  the `world_size==1` **byte-identical** no-op (identity for AllReduce, memcpy for
  AllGather). **RED-first proven**: a non-reducing AllReduce stub failed 4 cases /
  37 assertions; the real impl passes. Clean full-library CPU `-Werror` build, 0
  warnings; the full vt/engine suite stays green (the abstraction is additive,
  unused by the single-GPU path). **W2+ transport residuals (HW/registry-gated,
  NOT this pass):** collective `OpId`/`OpProvider` routing; the NCCL (kCUDA),
  multi-Spark RoCE, and MLX-ring (kMETAL) transports; the one-`Backend*`-per-
  DeviceType registry (`backend.cpp:42`) + device-0 hardcoding
  (`cuda_backend.cu:297-299`); and real TP/PP in the model forwards.
- **W2 — TP forward + loader (`BACKEND-DISTRIBUTED-TP`): DONE (2026-07-28,
  CPU-gated + NCCL derive-and-ship, `CLAIM-SCALE-OUT-W2`).** See **§W2 result**
  below.
- **W3 — NCCL transport (`BACKEND-DISTRIBUTED-COMM`/CUDA):** provider TU on
  `kCUDA`, comm per-`CudaBackend`, multi-device registry (unblock
  `cuda_backend.cu:297-299` + the index-0 shims). Gate: 2-GPU all-reduce.
  **W2 already landed the multi-device registry + the NCCL provider TU
  (derive-and-ship, build-verify pending on a NCCL host)** — W3 residual is the
  per-op `cudaSetDevice` device-affinity + the real 2-GPU run.
- **W4 — PP (`BACKEND-DISTRIBUTED-PP`):** stage split + point-to-point send/recv +
  multi-worker executor fan-out (`executor.cpp`). Gate: 2-stage ≡ 1-stage.
- **W5 — multi-Spark (`BACKEND-DISTRIBUTED-MULTINODE-SPARK`):** cross-host
  rendezvous (TCP store on `master_addr:port`) + NIC pinning; REUSE W3's NCCL over
  RoCE. Gate: the DeepSeek-V4 fp8-across-2-Sparks payoff run.
- **W6 — MLX ring (`BACKEND-DISTRIBUTED-MLX-RING`):** provider TU on `kMETAL`
  beside `metal_mlx_provider.mm`, `mlx.core.distributed` ring/JACCL over
  Thunderbolt. Gate: 2-Mac all-reduce + MLX-LM `--num-shards` competitor arm.

## W2 result (2026-07-28, `CLAIM-SCALE-OUT-W2`, CPU-gated + NCCL derive-and-ship)

Base `main` `308c312a`. Additive; `world_size==1`/`tp_size==1` byte-identical
(proven). NOT pushed. What landed:

1. **Multi-device backend registry** (`src/vt/backend.cpp:42`, `include/vt/backend.h`).
   The one-`Backend*`-per-`DeviceType` table became a per-`Device{type,index}`
   grid (`kMaxDevicesPerType=16`); new `RegisterBackend(Device,…)` /
   `GetBackend(Device)` / `RegisterDeviceResourceOps(Device,…)` overloads make
   device 0..N-1 addressable. **Byte-neutral for device 0**: `Device{type,0}` is
   the SAME slot the type-level API reads/writes, so `GetBackend(type) ==
   GetBackend(Device{type,0})` (identical `Backend*`) — proven by
   `tests/vt/test_backend_multidevice.cpp` (14 assertions). The CUDA registrar
   (`cuda_backend.cu:284`) now registers every GPU at its own slot (device-0
   hardcode gone). RESIDUAL: per-op `cudaSetDevice` affinity so a device-i backend
   allocates/launches on GPU i (single-GPU index-0 is correct as-is; HW-gated).
2. **Collective `OpId` routing** (deferred from W1). Added `OpId::kAllReduce/
   kAllGather/kSend/kRecv` (`include/vt/ops.h`); the `vt::Communicator` methods now
   dispatch through `GetOp(OpId, queue.device.type)` so a backend SUPPLIES the
   transport (`CommAllReduceFn` etc., `include/vt/communicator.h`). The CPU
   in-process reduce is registered on kCPU; `test_communicator` still passes 50/50
   THROUGH the OpId path.
3. **NCCL transport (kCUDA), derive-and-ship** (`src/vt/cuda/nccl_communicator.cu`,
   CMake `VLLM_CPP_NCCL` OFF by default). `NcclCommunicator` mirrors
   `PyNcclCommunicator` (`pynccl.py`): `ncclCommInitRank` (`:129-139`),
   `ncclAllReduce` (`:166-188`), `ncclAllGather` (`:196-215`), `ncclSend`
   (`:305-323`), `ncclRecv` (`:332-350`), stream-ordered on `q.handle`; registered
   as the kCUDA collective providers. **Honest signal: NOT built here** (no CUDA
   toolchain / no ≥2-GPU box) — it is a CUDA TU compiled only under
   `-DVLLM_CPP_NCCL=ON` on a NCCL host; without the flag the TU is a named stub and
   `GetOp(kAllReduce,kCUDA)` stays unresolved (a CUDA TP run fails LOUDLY, not
   silently). Build-verify + run = W3 residual.
4. **TP in the Qwen3-dense forward** (`include/vllm/model_executor/models/
   tensor_parallel.h`). `TpAllReduceSum` after o_proj (`dense_attn_block.h`, both
   the bf16 and NVFP4 paths) and after MLP-down (`qwen3.cpp` `MlpBlock`);
   `TpShard` column-shards the merged BF16 weight at the loader chokepoint
   (`dense_weight_loaders.h:131`). All via an optional `const TensorParallel* tp =
   nullptr`, so every existing caller (and every SACRED text model) is unchanged —
   `tp==nullptr`/`tp_size==1` enqueues ZERO extra vt:: ops and takes the whole-
   tensor shard, i.e. byte-identical (structurally guaranteed + full CPU suite
   green + `test_tp_forward` tp=1 inertness case).

**The real gate (no GPU): `tests/vt/test_tp_forward.cpp` — 60/60 PASS.** A TP-2
MLP (`y = relu(x @ W1ᵀ) @ W2ᵀ`) split ColumnParallel(W1)→no-comm,
RowParallel(W2)→all-reduce over the W1 CPU communicator, asserted **equal to the
unsharded tp=1 forward** on every rank (+ the ColumnParallel shard-concat sanity).
**RED-verified**: removing the all-reduce fails 24/60 assertions (the partial
product ≠ the full result). This certifies the TP wiring algebra WITHOUT a GPU.

### Same-host multi-GPU recipe (2× RTX 6000 Ada / A6000 in one box)

The reachable Leg-1 target (discrete GPUs over PCIe/NVLink). To run the real TP-2
once the transport is built and per-device affinity lands (W3):

1. **Build:** `cmake -DVLLM_CPP_CUDA=ON -DVLLM_CPP_NCCL=ON …` on a host with
   `nccl.h` + `libnccl` (the same NCCL vLLM links). This compiles
   `nccl_communicator.cu` and registers the kCUDA collective providers.
2. **Launch:** one worker PROCESS (or thread) per GPU — ranks 0 and 1 bound to
   `Device{kCUDA,0}` / `Device{kCUDA,1}` (both now in the registry). Mirrors vLLM's
   `MultiprocExecutor` one-proc-per-local-GPU (`multiproc_executor.py:176`).
3. **NCCL init:** rank 0 calls `ncclGetUniqueId`, broadcast it to rank 1 out-of-band
   (a tiny TCP/pipe handshake, exactly `pynccl.py` broadcasting the id on the CPU
   group), then each rank `ncclCommInitRank(comm, /*world=*/2, id, rank)` →
   construct an `NcclCommunicator(comm, rank, 2)`.
4. **tp-size:** hand each rank a `TensorParallel{&nccl_comm}` (`tp_size()==2`); the
   Qwen3-dense forward then column-shards gate_up/qkv, row-shards o_proj/down, and
   all-reduces after o_proj + MLP-down — the SAME code the CPU gate exercises, only
   the transport swapped (CPU in-process → NCCL). Correctness bar: sharded forward
   **token-exact** vs the single-GPU golden (strict where vLLM is deterministic,
   near-tie distributional otherwise).

**HW-run residual (honest):** no ≥2-GPU box exists here, so the NCCL build-verify
and the TP-2 token-exact run are UNEXECUTED; the CPU 2-rank `tp==tp1` equality is
the correctness evidence that stands today. Also residual: QKV head-aware KV
replication (`linear.py:1076-1079`), vocab/LM-head + MoE-EP sharding, and the
per-op `cudaSetDevice` affinity.

## Risks / decisions

- **HW-blocked correctness (Leg 1).** No 2-GPU box here; W1's CPU-loopback
  provider is the decision that lets the abstraction + TP-forward land and gate
  WITHOUT a GPU, deferring only the transport to a multi-GPU host.
- **Single-device registry is the real structural cost.** `backend.cpp:42` (one
  `Backend*` per type) + `cuda_backend.cu:297-299` (device 0 hardcoded) must
  become per-index; the `DeviceResourceOps` free-fn layer is the half-built path
  and the index-0 `VT_CHECK`s (`backend.cpp:83,101`) are the exact lines to lift.
- **MIRROR policy:** wrap the SAME NCCL vLLM uses (don't reinvent the collective);
  for MLX, use `mlx.core.distributed` directly (Apple ships JACCL/ring).
- **Numerics.** All-reduce ordering can perturb bf16 near-ties; the sharded ≡
  unsharded gate must use the near-tie distributional methodology where the model
  is non-deterministic, strict where deterministic (per `near-tie-distributional-gate`).
- **Leg 2 payoff is assumed until W1/W5 run** — the DeepSeek-V4 fp8 238 GiB fit and
  the vLLM-multi-node-as-is claim are explicitly flagged unconfirmed above.
