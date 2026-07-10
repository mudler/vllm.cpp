# Spec: Tensor Parallelism (multi-GPU) — TP=2 first

*Task #50 · written 2026-07-10 · upstream pin vLLM `e24d1b24` at
`/home/mudler/_git/vllm` · status: **spec written, not implemented***

## ⚠ OPEN HARDWARE QUESTION (blocks all GPU phases)

**`dgx.casa` (GB10) is a SINGLE-GPU box — no phase of this spec past Phase 0
can run there.** The bring-up needs a single node with **2 CUDA GPUs**:

- **Correctness bring-up (Phases 1–3): any 2× sm_80+ node** — 2×3090/4090 or
  a 2×A100 cloud rental. The bf16 path is already sm_80-portable and NVFP4
  runs as Marlin W4A16 on sm_80+ (expansion-map-2026-07-10.md: "bf16
  correctness path is already sm_80+ … nvfp4→Marlin"). Cheapest honest ask:
  an on-demand 2×A100 rental for the gate runs.
- **Full-fidelity gate (Phase 4 perf vs vLLM TP=2): 2× sm_120** (2× RTX 5090
  or RTX PRO 6000) for the native NVFP4 W4A4 kernel family closest to
  GB10/sm_121. Without it, the perf gate is honest only on the
  Marlin/bf16 config (measured vs vLLM on the *same* 2-GPU box, same config —
  benchmark-protocol.md applies unchanged).
- **2× DGX Spark over ConnectX-7 is NOT a substitute**: that is *multi-node*
  TP (NCCL over RoCE); vLLM disables custom-allreduce there
  (`vllm/config/parallel.py:991-995`) and it drags in multi-node bootstrap.
  Explicitly out of scope for this spec (goes with the future multi-node/DP
  spec).

The vLLM denominator for every A/B in this spec is **vLLM TP=2 on the same
2-GPU box** — never a GB10 number.

## 0. Scope

**In scope: tensor parallelism, single node, N=2 first (then 4/8), both gate
models (Qwen3.6-35B-A3B MoE+GDN hybrid, Qwen3.6-27B dense hybrid), NVFP4 +
bf16, safetensors loading.** MoE runs in **TP mode** (intermediate-dim expert
sharding — vLLM's default when `--enable-expert-parallel` is off).

**Out of scope → their own future specs** (feature-matrix.md rows):
- **PP** (`specs/pipeline-parallel.md`): stage partitioning
  (`vllm/distributed/utils.py:127-172`), NCCL send/recv of intermediate
  tensors with TP-slice optimization (`vllm/v1/worker/gpu_worker.py:994-1041`,
  `parallel_state.py:941-1158`).
- **EP + EPLB** (`specs/expert-parallel.md`): whole-expert partitioning
  (`fused_moe/config.py:1219-1234`), all2all dispatch/combine backends
  (`fused_moe/prepare_finalize/*`, default `allgather_reducescatter` —
  `parallel.py:185`), EPLB (`parallel.py:56-113`).
- **DP** (`specs/data-parallel.md`): engine scale-out, DPCoordinator wave
  sync (`vllm/v1/engine/coordinator.py:23-158`), per-step token-count
  all-reduce (`vllm/v1/worker/dp_utils.py:36-98`).
- **DCP/PCP** (decode/prefill context parallel, `parallel.py:339-342,124`)
  and **sequence parallelism** (an Inductor pass rewriting
  `all_reduce→RMSNorm` into `reduce_scatter→RMSNorm→all_gather`,
  `vllm/compilation/passes/fusion/sequence_parallelism.py:133-215`) —
  note SP as a *portable fusion pattern* candidate for the surpass track,
  but not TP-v1.
- GGUF×TP: Phase gates run on safetensors; GGUF per-rank narrowing of quant
  blocks is a follow-up flagged in Phase 2.

## 1. Upstream anatomy (vLLM @ e24d1b24)

### 1.1 Config & composition

`ParallelConfig` (`vllm/config/parallel.py`): `tensor_parallel_size` (:122),
`pipeline_parallel_size` (:120), `data_parallel_size` (:126),
`enable_expert_parallel` (:162), `decode_context_parallel_size` (:339,
reuses TP GPUs, must divide tp), `prefill_context_parallel_size` (:124).
**`world_size = PP × TP × PCP`** (:791-797); DP is *outside* world_size
(`world_size_across_dp`, :516-520). EP is not a size knob: the EP group spans
DP×PCP×TP ranks (`parallel_state.py:1870-1896`). Executor backend
auto-selects `"mp"` when world>1 and fits on the node, `"uni"` when ==1
(:871-916). Validation: heads divisible (`vllm/config/model.py:1170-1177`),
custom-AR forced off multi-node (:991-995).

### 1.2 What `-tp N` does end-to-end (multiproc executor)

`vllm/v1/executor/multiproc_executor.py` — one OS process per local rank:
`_init_executor` loops `WorkerProc.make_worker_process` per rank (:176-192),
spawn context (:1047-1052), daemon `context.Process` (:691-696). Worker
lifecycle: `worker_main` → init_device → load_model → READY over pipe →
`worker_busy_loop` (:865-889); parent `wait_for_ready` (:732-768);
death-pipe + monitor thread for fate-sharing (:781-804, :268-299).

**RPC**: a single shared-memory ring `MessageQueue` broadcasts
`(method, args, kwargs, output_rank)` to all workers
(`collective_rpc`, :340-374; queue created :151-157; attach
:566-568; ring = 24 MiB × 10 chunks with zmq overflow,
`shm_broadcast.py:373-374,746-770`). Per step: one `SchedulerOutput`
broadcast down (`execute_model`, :307-317) + one `ModelRunnerOutput` up from
**one** rank: `output_rank = world_size − TP×PCP` = TP-rank-0 of the last PP
stage (:495-509); each worker has a 1-reader response MQ (:571).

### 1.3 Process-group init

Worker entry: `init_worker_distributed_environment`
(`vllm/v1/worker/gpu_worker.py:1269-1291`) → `init_distributed_environment`
(`parallel_state.py:1536-1691`; TCP-store rendezvous via
`init_method="tcp://master:port"`) → `initialize_model_parallel`
(:1694-1935): rank layout `arange(world).reshape(-1, DP, PP, PCP, TP)`
(:1760-1775); groups created in order TP (:1778-1792, with in-node 4 MiB
MessageQueue broadcaster :484-487), DCP, PCP, PP, DP, EP, EPLB. Each
`GroupCoordinator` holds a device (NCCL) PG + a CPU gloo PG (:424-439).
`StatelessProcessGroup` (`vllm/distributed/utils.py:198-513`) is a
TCPStore-only metadata group (bootstrap without torch's global PG state).

### 1.4 Device communicators & allreduce dispatch

`cuda_communicator.py` builds up to 5 backends (:75-118) and dispatches
`all_reduce` in order: NCCL symm-mem → QuickReduce (ROCm) → FlashInfer →
**CustomAllreduce** (`should_custom_ar`) → SymmMem → **PyNCCL** → torch
fallback (:255-312). Size gates in `all_reduce_utils.py:31-96` (e.g. SM90
custom-AR max: world 2 → 2 MiB).

**CustomAllreduce** (`custom_all_reduce.py` + `csrc/custom_all_reduce.cuh`):
worlds {2,4,6,8} (:52); single-node only (:90-96); world>2 needs full NVLink
mesh (:149-156) — **world==2 is exempt** (PCIe P2P suffices, `_can_p2p`
:31-45,161-167). IPC-shared buffer+meta (:173-194,293-312). One-shot kernel
(`.cuh:298-314`) vs two-shot reduce-scatter+all-gather (`.cuh:321-366`);
world 2 always one-shot, size thresholds `.cuh:589-597`. Enable gate: 16-byte
multiple, size < max (:230-243). Graph capture: `register_graph_buffers`
records IPC addresses used inside the graph (:196-228).

### 1.5 The NCCL C-API surface vLLM actually uses

`pynccl_wrapper.py` ctypes-loads `libnccl.so` and binds exactly:
`ncclGetErrorString, ncclGetVersion, ncclGetUniqueId, ncclCommInitRank,
ncclAllReduce, ncclReduce, ncclAllGather, ncclReduceScatter, ncclSend,
ncclRecv, ncclBroadcast, ncclCommDestroy, ncclCommAbort, ncclGroupStart,
ncclGroupEnd, ncclCommWindowRegister/Deregister` (:147-320). **Recorded
rationale** (:4-23): torch.distributed's allreduce issues extra CUDA API
calls that are illegal during CUDA-graph capture; a thin C-API binding is
capture-safe and lets the `.so` be swapped at runtime. `PyNcclCommunicator`:
rank 0 broadcasts `ncclUniqueId` out-of-band, `ncclCommInitRank`, 1-element
warmup allreduce (`pynccl.py:110-146`); ops pass raw pointers + the current
stream (:188-197); `ncclCommAbort` deferred to a watchdog thread to avoid
deadlock vs live graphs (:148-164). **A C++ port starts from the same
position by construction — we call the same C API.**

### 1.6 CUDA graphs × collectives

`GroupCoordinator.graph_capture` (`parallel_state.py:578-620,1410-1427`)
wraps capture in the custom-AR context; during capture custom-AR takes the
`registered=True` path (pre-registered IPC pointers,
`custom_all_reduce.py:264-280`) and on exit registers graph buffers. NCCL
collectives are capturable when invoked as bare C calls on the capture
stream (the pynccl design point, §1.5) — NCCL ≥2.9 supports capture; keep
one comm per rank and never call CUDA APIs mid-capture.

## 2. TP sharding of OUR exact layers

### 2.1 Machinery (`vllm/model_executor/layers/linear.py`)

- **ColumnParallelLinear** (:397): weight split on **output** dim;
  per-rank `divide(output_size, tp)` (:441); loader narrows checkpoint at
  `tp_rank*shard_size` on `output_dim` (:520-541). No forward collective
  (gather_output default off, :462).
- **RowParallelLinear** (:1541): split on **input** dim (:1594, loader
  :1645-1657); forward does local GEMM then
  `tensor_model_parallel_all_reduce` (:1694-1695); bias rank-0 only (:1691).
- **MergedColumnParallelLinear** (:580): per-segment shard
  offsets/sizes ÷ tp (:750-753), packed-dim & block-scale adjust
  (:719-767).
- **QKVParallelLinear** (:942): `num_heads = heads/tp` (:995); **kv
  replication** when `tp ≥ total_kv_heads`: `num_kv_heads=1`,
  `replicas = tp/total_kv_heads` (:996-1001); k/v checkpoint read offset
  uses `tp_rank // num_kv_head_replicas` so replicas load the same slice
  (:1303-1310).
- NVFP4 uses `weight_loader_v2` parameter classes (:48-64,473-475).

### 2.2 Full attention (`qwen3_next.py:225-392`, reused by qwen3_5.py:137)

Heads: asserts `heads % tp == 0` (:239); kv: divisible or replicate
(:245-250). `qkv_proj = QKVParallelLinear` with q doubled by
`attn_output_gate` (:260-268 — both gate models have the gate);
`o_proj = RowParallelLinear` → **the attention all-reduce** (:270-276).
Per-head q/k RMSNorm on head_dim slices — rank-local (:311-312,344-377).
KV cache per rank holds only its kv heads (:294-309).

### 2.3 GDN — gated delta net
(`vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`, class
`QwenGatedDeltaNetAttention` :420; Qwen3.5 layout `gqa_interleaved_layout=False`,
qwen3_5.py:134)

Everything splits on the **head/channel dim; state is fully rank-local; the
ONLY collective is out_proj's all-reduce** (no all_reduce/all_gather anywhere
else in the file):

- `in_proj_qkvz` = MergedColumnParallel `[key_dim, key_dim, value_dim,
  value_dim]` (4 independent column shards; :566-590). Checkpoint mapper
  fuses split `in_proj_qkv`+`in_proj_z` → `qkvz` (qwen3_5.py:203-210).
- `in_proj_ba` = MergedColumnParallel `[num_v_heads, num_v_heads]`
  (:592-614). The Marlin `maybe_disable_tp` replication hack (:616-642) is
  AWQ/GPTQ/INC-only — **does NOT trigger for NVFP4**; ba stays sharded.
- `conv1d` = ColumnParallel over `conv_dim = 2·key_dim + value_dim` (:466-473)
  with `mamba_v2_sharded_weight_loader([qk, qk, v], tp, rank)` (:502-510) —
  shards each of the K/K/V channel blocks by `dim//tp` so per-rank conv
  channels stay contiguous (`mamba_mixer2.py:171-228`).
- `A_log`, `dt_bias`: per-v-head params `divide(num_v_heads, tp)`,
  `sharded_weight_loader(0)` (:516-527).
- `RMSNormGated(head_v_dim)` — per-head, rank-local (:536-543).
- `out_proj = RowParallelLinear(value_dim → hidden)` → **the GDN
  all-reduce** (:545-552).
- **State shapes** (`mamba_utils.py:213-234`): conv state
  `(conv_dim/tp, kernel−1+num_spec)`; recurrent state
  `(num_v_heads/tp, head_v_dim, head_k_dim)` — the delta-net state splits on
  v-heads, head dims never split.
- **Constraints**: `divide()` asserts exact division of `num_v_heads`,
  `conv_dim`; `fix_query_key_value_ordering` needs `num_k_heads/tp` integral
  (:653,663). No group-replication fallback for GDN (mamba2's
  `extra_groups_for_head_shards` is NOT on this path) ⇒ **max TP = num_k_heads
  = 16** for both gate models.

### 2.4 MoE under TP (`qwen3_next.py:101-222` + fused_moe)

`use_ep = (dp·pcp·tp>1) && enable_expert_parallel`
(`fused_moe/config.py:1190-1193`). **TP mode (our v1)**: every rank holds
ALL 256 experts, each sharded on the intermediate dim —
`intermediate_size_per_partition = 512/tp` (config.py:1310-1315); w13
column-shard + w2 row-shard per expert (`routed_experts.py:448-540`).
Router gate + shared-expert gate = ReplicatedLinear (qwen3_next.py:138-152).
`shared_expert` has `reduce_results=False` (:165-174); its output folds into
the single MoE combine all-reduce
(`moe_runner.py:416-460` `_maybe_reduce_final_output`) — **one all-reduce
per MoE block, not two**. Constraints: `tp ≤ num_experts` (:118-122),
`intermediate % tp == 0` (config.py:1314).

### 2.5 Embedding / LM head / logits

`VocabParallelEmbedding` (`vocab_parallel_embedding.py:198`): vocab-dim
shard (:302-310), masked lookup + zero + **all-reduce** (:472-491).
`ParallelLMHead` (:505) same sharding; both models have
`tie_word_embeddings=false` (separate lm_head). Logits:
`LogitsProcessor._gather_logits` — **all-gather** (or gather-to-rank-0) of
vocab-sharded logits (`logits_processor.py:75-87`), trim to org_vocab_size
(:102-103).

### 2.6 NVFP4 (ModelOpt W4A4/W4A16) sharding constraints
(`vllm/model_executor/layers/quantization/modelopt.py`)

- Packed weight: `uint8 [out_pp, in_pp/2]`, `input_dim=1, output_dim=0`
  (:1158-1168) — the ÷2 packing is baked into the allocated shape, so the
  standard narrow-by-`shape[dim]` loaders shard it with no extra math
  (`parameter.py:148-154,220-230`).
- Block scale: fp8 `[out_pp, in_pp/16]`, same dims (:1186-1197; group 16,
  :1020,1083-1084) — narrows in lockstep with the weight, so groups stay
  aligned **provided each rank's K-shard is a multiple of 16**: enforced by
  `create_weights` raising when `input_size_per_partition % 16 != 0`
  (:1147-1150). This is the **hard alignment rule for every RowParallel fp4
  shard** (o_proj, GDN out_proj, w2).
- `weight_scale_2`, `input_scale` = PerTensorScale, **replicated**
  (:1172-1183; `parameter.py:279-289`); collapsed to `alpha =
  input_global·weight_global` post-load (:1215-1229).
- Swizzled-scale layout is produced per-rank post-shard by the kernel's
  `process_weights_after_loading` (:1231-1232) — shard FIRST in the linear
  (row-major) scale layout, THEN swizzle per rank. Same order in our port.
- W4A16 Marlin variant (:1243-1277) shards identically (sm_80 fallback path).

### 2.7 Divisibility check — our exact models (all values from
qwen36-forward-notes.md §1 real config.json)

| quantity | 35B-A3B | 27B | TP=2 | TP=4 | TP=8 |
|---|---|---|---|---|---|
| attn q-heads (×2 gate) | 16 | 24 | 8/12 ✓ | 4/6 ✓ | 2/3 ✓ |
| attn kv-heads | 2 | 4 | 1/2 ✓ | rep×2 / 1 ✓ | rep×4 / rep×2 ✓ |
| GDN Hk (groups) | 16 | 16 | 8 ✓ | 4 ✓ | 2 ✓ |
| GDN Hv | 32 | 48 | 16/24 ✓ | 8/12 ✓ | 4/6 ✓ |
| GDN conv_dim | 8192 | 10240 | ✓ | ✓ | ✓ |
| MoE/MLP intermediate | 512 | 17408 | 256/8704 ✓ | 128/4352 ✓ | 64/2176 ✓ |
| fp4 K%16 (o_proj in 4096/6144; out_proj in 4096/6144; w2 in 512) | | | ✓ | ✓ | ✓ |
| vocab 248320 | | | 124160 ✓ | 62080 ✓ | 31040 ✓ |

Both models are cleanly TP=2/4/8-able; GDN caps TP at 16.

## 3. Hot-path collectives at TP=2 (the complete inventory)

Per step, bf16 hidden activations (`[T, hidden]`):

| where | op | count/step (35B / 27B) | payload at decode T=B |
|---|---|---|---|
| embedding | all-reduce | 1 | B×2048 / B×5120 ×2B |
| GDN out_proj | all-reduce | 30 / 48 | same |
| attn o_proj | all-reduce | 10 / 16 | same |
| MoE combine / MLP down | all-reduce | 40 / 64 | same |
| logits | all-gather | 1 | B×(vocab/2)×dtype |

**≈ 82 (35B) / 129 (27B) all-reduces + 1 all-gather per step.** At decode
these are tiny (a few hundred KB at B=64) → **latency-bound**, which is why
vLLM's one-shot custom-allreduce and graph capture of collectives matter;
NCCL-only will pass correctness but bleed decode latency (Phase 4 target).
No reduce-scatter/all-gather pairs on the TP hot path unless SP (out of
scope) or EP (out of scope) is enabled.

## 4. C++ mirror design (no torch.distributed)

### 4.1 NCCL directly

New `src/vllm/distributed/` mirroring upstream file-for-file where it makes
sense (`parallel_state.{h,cpp}`, `communication_op.{h,cpp}`) + a
`vt::` communicator (`src/vt/cuda/cuda_nccl.cu` + `include/vt/comm.h`).
Bind the **same 17-symbol C-API surface as pynccl_wrapper** (§1.5), via
`dlopen("libnccl.so.2")` mirroring vLLM's runtime-swap rationale (build
falls back to `find_package`-linked NCCL; keep the dlopen seam so Vulkan/
Metal backends can no-op it). All ops take `(ptr, count, dtype, op,
comm, stream)` — capture-safe by construction.

### 4.2 Process model — one process, one thread per rank (recorded deviation)

vLLM runs one *process* per rank because of Python (GIL, CUDA context per
proc) and pays for it with the shm `MessageQueue` + pickle + TCP-store
machinery (§1.2-1.3). In C++ **Phase 1 runs TP ranks as threads of one
process**:

- `ncclCommInitAll` (or ncclCommInitRank per thread group-wrapped) — no
  ncclUniqueId exchange, no TCP store, no shm ring: the whole §1.2/§1.3
  bootstrap collapses.
- `collective_rpc` collapses to an in-process fan-out: the executor calls N
  rank-runners on N threads and passes `SchedulerOutput` **by const ref**
  (zero serialization); output taken from rank 0 (mirrors
  `output_rank`, §1.2). This is exactly the collapse our
  `include/vllm/v1/executor/executor.h:5-25` comment already documents for
  world=1 — we extend it to world=N.
- Custom-allreduce (Phase 4) gets *simpler*: peer pointers via
  `cudaDeviceEnablePeerAccess`, no IPC handles (upstream's
  `register_buffer`/graph-IPC dance exists only because of process
  isolation; keep the registration *interface* shape so the port stays
  mechanical).
- **Deviation record**: multiproc WorkerProc/MessageQueue/TCP-store are NOT
  ported in TP-v1. Record in porting-inventory §9 + parity-ledger. They
  return with the multi-node/DP spec (where process isolation is forced).
  The `Executor` seam keeps vLLM's method surface (`execute_model`,
  `sample_tokens`, `collective_rpc(method, output_rank)`) so a
  `MultiprocExecutor` can drop in later without touching EngineCore.

### 4.3 Parallel state

Port `parallel_state` minimally: `init_distributed_environment(world, rank)`
+ `initialize_model_parallel(tp)` with only the TP group at v1 (groups
PP/DP/EP stubbed with world=1 semantics, mirroring upstream's rank-layout
math :1760-1775 so adding groups later is mechanical). Global accessors
`get_tp_group()/get_tensor_model_parallel_rank()/world_size()` mirrored —
per-rank *thread-local* in our thread model.

### 4.4 Collectives dispatch

`communication_op` port: `tensor_model_parallel_all_reduce/_all_gather`
routing to the vt communicator. v1 = NCCL always; Phase 4 adds the
custom-allreduce fast path behind vLLM's `should_custom_ar` gates (§1.4)
ported 1:1 (16-byte multiple, size ceiling, world∈{2,4,6,8}), kernel ported
from `csrc/custom_all_reduce.cuh` (one-shot :298-314 first — world 2 always
one-shot; two-shot :321-366 with 4+ GPUs).

### 4.5 CUDA graphs

Our decode graphs (`Qwen3_5DecodeGraph`, `src/vt/cuda/cuda_backend.cu:76-110`
capture API) must capture `ncclAllReduce` calls on the capture stream —
legal since the vt communicator is bare C calls (§1.5 rationale). Rules:
comm init + warmup allreduce BEFORE first capture (mirror pynccl warmup,
`pynccl.py:110-146`); NCCL buffers from the pre-warmed DevicePool (no
allocation mid-capture); all ranks capture/replay in lockstep (the executor
fan-out already synchronizes steps). Custom-AR in-graph: port
`register_graph_buffers` equivalent (pointer table finalized at capture
end, §1.4).

## 5. Our-code touch points (survey of this repo)

Zero distributed code exists today (no nccl/collective hits anywhere); the
seams were deliberately kept:

1. **Executor seam**: `include/vllm/v1/executor/executor.h:42` — the
   collapsed UniProcExecutor holding one `ModelRunnerBase&`; becomes the
   N-rank fan-out (§4.2). `EngineCore::step` (`v1/engine/core.h:65`)
   unchanged.
2. **Per-rank runner**: `include/vllm/v1/worker/gpu/runner.h:117`
   (`GPUModelRunner`) — instantiate one per rank (own device, Queue,
   DevicePool, KV cache, InputBatch); sampler runs on rank 0 (logits
   all-gather lands before sampling).
3. **Weight sharding hook**: `qwen3_5_weights.h:234` `TensorResolver` +
   `LoadQwen3_5Moe[Layer]` (`qwen3_5_weights.cpp`), `qwen3_5_dense_weights.cpp`
   — add a `ShardSpec{dim, tp_rank, tp_size, segments, kv_replicas}` applied
   at resolve time (mirrors upstream loader narrowing §2.1). Weights are
   already stored UNFUSED (weights.h:16-18 notes fusion is "a TP-sharding
   naming convenience") — per-tensor narrowing is straightforward; NVFP4
   `Nvfp4Weight` (weights.h:72) shards weight/scale in lockstep with the
   K%16 rule (§2.6), scale2/alpha replicated.
4. **Forward collectives** in `src/vllm/model_executor/models/qwen3_5.cpp`:
   all-reduce after attn `o_proj` (:1964-1966, :2169-2171), GDN `out_proj`
   (:1533-1537, :1826-1830), MoE combine incl. shared expert (:2311, :2488,
   :2728-2731, :2241-2253), dense `down_proj` (:2882-2894); vocab-parallel
   embedding (:2996-3009); logits all-gather (:3014, :3069-3094). Per-rank
   head/expert dims come from dividing `HfConfig`-derived sizes
   (`hf_config.h:22`) — introduce a per-rank "sharded dims" view, don't
   mutate the parsed config.
5. **Config plumbing**: `tensor_parallel_size` into `EngineParams`
   (`model_loader.h:36`), a `ParallelConfig` struct under
   `src/vllm/config/` mirroring upstream fields we implement; CLI
   `--tensor-parallel-size`/`-tp` in `examples/cli/main.cpp:66` +
   `examples/server/main.cpp`; multi-rank stack constructed in
   `LoadedEngine` (`model_loader.h:66`).
6. **Build**: NCCL dep next to `find_package(CUDAToolkit)`
   (CMakeLists.txt:230-232); new sources appended to the explicit lists
   (:127-202, cuda block :219-232).

## 6. Phased plan with gates

**Phase 0 — infra, runs on ANY box incl. CPU/dgx** (~3-5 d):
ParallelConfig + CLI plumbing; `vt::Comm` interface with a *host mock*
backend (ranks = threads, collectives = std::barrier+sum) so
parallel_state, sharding math, and executor fan-out get CPU unit tests;
TensorResolver ShardSpec + per-rank dims view with **TP=1 no-op gate: the
existing token-exact suite stays green with the sharding code live**.
GATE-0: divisibility table (§2.7) asserted in unit tests; TP=2 CPU-mock
forward of a toy model matches TP=1 bit-for-bit on the mock reducer.

**Phase 1 — 2-GPU bring-up, NCCL only** (~1 wk, needs 2×GPU box):
dlopen NCCL binding + comm init (threads, §4.2); collectives in eager
(non-graph) path; per-rank weight narrowing for every layer of §2; 27B
first (dense = fewer moving parts), then 35B.
GATE-1: **ours-TP=2 token-exact (16/16 greedy) vs vLLM-TP=2** on the same
box/config — the same-sharding oracle comparison makes numerics
deterministic-comparable; ours TP=2 vs ours TP=1 recorded as a secondary
check (reduction-order drift allowed, greedy tokens expected stable).

**Phase 2 — CUDA-graph decode + full features** (~1 wk):
NCCL inside decode graph capture (§4.5); MoE TP path incl. shared-expert
fold (§2.4); NVFP4 W4A4 (if sm_120 box) or Marlin W4A16 path; GGUF×TP
flagged/deferred explicitly if quant-block narrowing is nontrivial.
GATE-2: GATE-1 correctness re-held with graphs on, both models.

**Phase 3 — perf to parity** (~1-2 wk):
nsys BOTH sides (parity-lever-protocol) on the 2-GPU box; expected levers:
custom-allreduce port (one-shot, world 2 — §4.4; decode is latency-bound at
~82-129 tiny all-reduces/step, §3), overlap/fusion of allreduce with
adjacent norm (mirror-as-floor; SP-style fusion = surpass-track candidate).
GATE-3 (the honest gate): **≥ vLLM TP=2 on every axis** (throughput, TTFT,
TPOT, memory), both models, same box, per benchmark-protocol.md; plus
scaling sanity TP=2 ≥ ~1.6× TP=1 throughput at the memory-bound operating
point (record; vLLM same-box scaling is the denominator).

**Phase 4 — breadth** (follow-ups, own mini-gates): TP=4/8 (kv-replication
path §2.1 exercised), GGUF×TP, multiproc executor + multi-node (own spec
with DP).

## 7. Tests to port (per .agents/test-porting.md)

Upstream `tests/` @ pin → our tiers:

| upstream | cases | tier → ours |
|---|---|---|
| `tests/distributed/test_pynccl.py` | `test_pynccl`, `_multiple_allreduce`, `_all_gather`, `_reduce_scatter`, **`_with_cudagraph`** (:79-416) | T-unit (2-GPU nightly): doctest `tests/vt/test_comm_nccl.cpp` — allreduce/allgather numeric asserts, graph-capture replay twice |
| `tests/distributed/test_comm_ops.py` | `test_multi_process_tensor_parallel` (all_reduce/all_gather/broadcast workers :342-376) | T-unit: `tests/vllm/distributed/test_comm_ops.cpp` over `tensor_model_parallel_*` wrappers (mock backend on CPU CI, NCCL on nightly) |
| `tests/distributed/test_custom_all_reduce.py` | `test_custom_allreduce` (:123; sizes sweep, graph + eager) | T-unit + T-parity vs NCCL result (Phase 3, gate for the custom-AR port) |
| `tests/distributed/test_multiproc_executor.py` | `_initialization_tensor_parallel` (:90), `_collective_rpc` (:114), `_failure_callback` (:138) | T-unit: executor fan-out semantics re-expressed for the thread executor (init, rpc fan-out+output_rank, rank-failure propagation); multiproc-specific cases SKIPPED-tracked until the multiproc spec |
| `tests/distributed/test_shm_broadcast.py` | MessageQueue ring cases (:152-351) | SKIPPED-tracked (deviation §4.2 — no shm MQ in TP-v1); ports with multi-node spec |
| `tests/basic_correctness/test_basic_correctness.py::test_models_distributed` | TP=2 generate vs TP=1/HF (:204-244) | T-parity: our TP=2 vs vLLM-TP=2 token-exact goldens (GATE-1), `tools/parity/` recipe extended with `--tp` |
| `tests/distributed/test_utils.py` (StatelessProcessGroup parts) | | SKIPPED-tracked → multi-node spec |
| `tests/v1/distributed/*`, `test_pipeline_parallel.py`, `test_expert_parallel.py`, `test_symm_mem_allreduce.py`, `test_nccl_symm_mem.py` | | out of scope → listed in the PP/EP/DP specs |

T-e2e: server conformance suite re-run with `-tp 2` on the nightly 2-GPU
job (mirrors upstream running entrypoints tests under distributed marks).
Traceability: case names derived 1:1; record the mapping in
porting-inventory.md with the code mapping.

## 8. Effort ranking (grounded in §1-§5)

| # | slice | effort | risk |
|---|---|---|---|
| 1 | Phase 0 infra + mock (CPU-testable now) | 3-5 d | low |
| 2 | NCCL binding + thread executor | 2-3 d | low (17 C symbols) |
| 3 | Weight sharding all layers (incl. NVFP4 lockstep, GDN segmented conv loader) | 4-6 d | medium — most of the correctness surface |
| 4 | Forward collectives + per-rank dims | 2-3 d | low |
| 5 | Graph-captured collectives | 2-4 d | medium (capture discipline) |
| 6 | custom-allreduce port + perf gate | 1-2 wk | medium — decode latency parity lives here |

Critical path to GATE-1 ≈ 2-2.5 wk of agent time **after hardware lands**;
Phase 0 is dispatchable immediately.
