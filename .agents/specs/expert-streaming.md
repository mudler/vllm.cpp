# Spike: expert streaming from disk (`ENG-EXPERT-STREAM`)

Run MoE models with less resident GPU/unified memory by keeping routed-expert
weights on fast storage (NVMe) and paging them in on demand, keyed on router
output, under a byte-budgeted resident cache. User-directed spike
(2026-07-10); design grounded in a full source scan of **antirez/ds4**
(DwarfStar, github.com/antirez/ds4) plus measured dgx.casa NVMe numbers and
the actual 35B-A3B checkpoint shapes. Related mirror-floor inventory row:
`ENG-WEIGHT-OFFLOAD` (vLLM UVA `cpu_offload_gb`, stays a separate future
mirror port).

Verdict up front (from the bandwidth math in §3, all inputs measured):

- **Low concurrency (c1-c4): VIABLE as a capacity feature, conditional on the
  bank-only loader and fixed-slot Marlin design below.** At a 50%
  expert-resident fraction the 35B-A3B needs ≤ 270 MiB of expert reads per
  token worst-case; the dgx NVMe sustains 5.0-5.3 GB/s with a small read
  pool, an I/O-only bound of ≥ 17.7 tok/s before locality. It can free ~8.4 GiB
  (f=0.5) to ~12.7 GiB (f=0.25) only if streaming mode never materializes the
  current per-expert host `OwnedTensor` copies or the full device Marlin
  resident; keeping either would erase the capacity win on unified-memory GB10.
- **High concurrency (the MVP gate operating point): NOT SERVED.** At
  conc≥32 every step touches ~64-98% of all 256 experts per layer
  (1-(1-8/256)^B), so per-step I/O approaches (1-f) x 16.9 GiB regardless of
  reordering — orders of magnitude below the gated ~2.8k tok/s. The feature
  must refuse/warn at high concurrency, never silently degrade the gate.
- **tmpfs//tmp is NOT a disk tier on GB10**: unified memory means tmpfs pages
  ARE the same LPDDR5X the GPU uses (and dgx's `/tmp` is ext4 on the same
  NVMe anyway — verified `df -hT /tmp` 2026-07-10). On discrete-GPU hosts a
  host-RAM tier IS meaningful (PCIe ~25-60 GB/s, ~5-10x NVMe) — kept as a
  pluggable backing-store leaf (W7), the expert-granular analogue of vLLM's
  blanket `cpu_offload_gb`.

## Scope

| Field | Content |
|---|---|
| Row IDs | `ENG-EXPERT-STREAM` (this spike; work-breakdown leaves W0-W8 below). Mirror-floor context row: `ENG-WEIGHT-OFFLOAD` (INVENTORIED, not claimed here) |
| In | Routed-MoE expert weights only (gate/up/down per expert per layer) for MoE models, first target Qwen3.6-35B-A3B NVFP4 (safetensors): versioned NVMe Marlin-layout expert bank, bank-only loader, fixed-capacity contiguous Marlin slot arrays, logical-expert→slot remap after router D2H, async O_DIRECT pread/copy pool, hotness-decayed-LFU + LRU-tiebreak eviction with in-flight/selected protection, chunked slot sweeps for long prefill + decode-cache seeding, optional hotlist preload, `--simulate-used-memory`-style honest measurement mode |
| Out (this row) | Dense weights, shared experts, router, norms, KV cache (all stay resident); GGUF expert streaming (a follow-up leaf, but NOT for the reason first recorded — see the GGUF reconciliation note below: the slicer LANDED 2026-07-22 and only the residency policy remains); host-RAM tier for discrete GPUs (W7, after gate); vLLM UVA `cpu_offload_gb` mirror (own row `ENG-WEIGHT-OFFLOAD`); expert-parallel EPLB (`PAR-EP-EPLB`) |
| Supported modes | `off` (default, unchanged engine); `nvme` (expert bank file + device cache). Budget accepted as expert count or `NGB` (ds4 CLI semantics `--ssd-streaming-cache-experts 32GB`, ds4_ssd.c:46-70); auto budget = fraction of free device memory minus non-streamed needs (ds4_ssd.c:80-106) |
| Dispatch behavior | Streaming engages only when enabled AND the model is MoE. It branches before `BuildMoeMarlinResident`, never builds the full `[E,...]` resident, copies router IDs device→host and synchronizes once per MoE layer in phase 1, maps logical IDs to fixed cache slots, then runs the unchanged dense-stride Marlin kernel over slot IDs. Engine WARNS and refuses (or auto-disables per config) when max concurrency exceeds the regime bound (§3, default warn at conc>4, hard cap configurable); dense models reject the flag |
| Regimes served | c1-c4 capacity/single-user; models larger than device memory. Explicitly NOT the high-concurrency throughput gate |

## ds4 anatomy (what we are porting from)

**What ds4 is.** DwarfStar: antirez's self-contained C inference engine (no
ggml link; kernels/quants adapted from llama.cpp under MIT, README.md:46-57)
for exactly two models: DeepSeek V4 Flash (43 layers, 256 routed experts
top-6 + 1 shared, 3 hash-routed layers, ds4.c:180-196) and DeepSeek V4 PRO
(61 layers, 384 experts top-6, ds4.c:214-232), custom GGUFs only, backends
Metal (primary), CUDA (incl. DGX Spark/GB10) and ROCm/Strix Halo
(README.md:14-17). SSD streaming is its capacity mode: "non-routed model
weights stay resident, while routed MoE experts are kept in an in-memory
cache and loaded from the GGUF file on cache misses" (README.md:180-216).
README calls it Metal-only but the code enables CUDA/Linux and ROCm too
(ds4.c:79-90 `ds4_backend_supports_ssd_streaming`; commit bbd069d "Add CUDA
and ROCm SSD streaming").

**Design, precisely (file:line in the ds4 tree @ 80ebbc3):**

- **Weight layout / reads**: single GGUF, experts live in per-layer 3D
  tensors; a per-expert slice is `tensor_bytes / n_expert` at
  `abs_offset + expert_id * per_expert_bytes` (ds4.c:14048-14060) —
  per-expert bytes must be uniform within a layer
  (`weights_streaming_layer_experts_uniform`, ds4.c:3338-3356; mixed
  per-LAYER quants allowed, e.g. Q4 "boosted" last layers among IQ2, commit
  81f35e7). Reads are plain `pread(2)` against the model fd
  (ds4_metal.m:7776-7820); macOS gets `F_RDADVISE` page-cache readahead
  hints (ds4_metal.m:7661-7686); the CUDA path has an O_DIRECT variant with
  aligned staging (ds4_cuda.cu:1055-1104) then `cudaMemcpyAsync` on an
  upload stream (ds4_cuda.cu:1262).
- **Async I/O**: a persistent pthread pool, default 9 threads, cap 18
  (`DS4_METAL_STREAMING_EXPERT_PREAD_THREADS`, ds4_metal.m:7754-7773,
  7836-7848); one task per missing tensor — up to 6 experts x 3 tensors = 18
  tasks per layer (ds4_metal.m:7699-7723 `ds4_gpu_stream_expert_pending_load`).
- **When selection is known vs when weights are needed (the prefetch
  window)**: selection is only known after the layer's router. ds4 splits
  the GPU command buffer at each MoE layer, reads back the 6 selected ids
  (or computes the router ON CPU from the read-back pre-FFN norm
  activations, `metal_graph_decode_cpu_router`, ds4.c:14170-14247), then
  `ds4_gpu_stream_expert_cache_begin_selected_load` starts pool preads for
  the missing experts (ds4_metal.m:10618-10800) while the GPU continues; a
  masked address-table kernel computes the RESIDENT experts' contribution
  first and a deferred pass computes the MISSING ones after
  `pending_load_finish` waits on the pool (split path,
  ds4_metal.m:23392-23433; timing counters resident_submit/missing_wait,
  ds4.c:2608). Exception: DeepSeek's 3 hash layers route by token id alone
  (`ffn_gate_tid2eid`, ds4.c:7307-7320, 14020-14076), so their loads can
  begin before any compute — a true lookahead ds4 exploits; Qwen3.6 has no
  hash layers, so our window is strictly within-layer.
- **Caching/eviction**: per-(layer,expert) entry table `[61][384]` backed by
  slab-allocated buffers (256 slabs, ds4_metal.m:370-458); mlocked with
  graceful budget-capping when mlock fails (ds4_metal.m:8108-8205, commit
  7a77a28; README.md:246-256). Eviction is NOT plain LRU: victim = lowest
  **route-hotness** with LRU tiebreak, where hotness += 1 for every
  *selected* expert (hit or miss) and is halved every 16 decode tokens
  (`DS4_METAL_STREAM_EXPERT_HOTNESS_DECAY_TOKENS`, ds4_metal.m:376,
  8678-8776, 9666-9830). Recorded rationale: hit-count LFU "penalizes
  experts that are repeatedly selected but evicted before a second hit"
  (ds4_metal.m:9679-9683). Entries referenced by in-flight GPU command
  buffers (sequence-tracked, ds4_metal.m:834-905) or currently selected are
  eviction-protected. CUDA backend uses simple LRU ticks + device slabs,
  default budget 512 experts (ds4_cuda.cu:38, 156-198, 1967, 2011-2098).
- **Budget**: `--ssd-streaming-cache-experts N|NGB` (GiB converted via
  per-expert bytes, ds4_ssd.c:46-78); auto plan = 80% of the recommended
  working set minus non-routed bytes (ds4_ssd.c:80-106).
- **Prefill is a different regime and ds4 knows it**: long prefill touches
  ~all experts, so streaming prefill switches to **full-layer sequential
  streaming** above a per-quant token cutoff with a lookahead thread
  pipelining the next layer's expert tensors (ds4.c:11696-11760, commit
  57b8a4c), and afterwards **seeds the decode cache from the last ≤64
  prefill tokens' router selections** (`DS4_STREAMING_PREFILL_CACHE_SEED_MAX_TOKENS`,
  ds4.c:10310, 13812-13824, 19657).
- **Hot experts**: an expert-locality profiler records per-layer selection
  histograms + weights, simulates latest-N-unique caches for caps 1..384 and
  measures adjacent-token overlap/Jaccard (ds4.c:750-830); its output ships
  in-source as default hotlists (13,320 (layer,expert) pairs across PRO +
  Flash, ds4_streaming_hotlist.inc:1-3) used to pre-warm the cache at
  startup, capped at 4096 experts in auto mode (ds4.c:13949-14016);
  `--ssd-streaming-cold` disables preload for measurement.
- **Honest measurement tooling**: `--simulate-used-memory` mlocks N GiB
  before load so page-cache effects can't fake streaming results
  (ds4_ssd.c:108-181).
- **Measured numbers ds4 documents**: sparse. README.md:246-249 (M5 Max
  128 GB, PRO q2): the automatic budget (~59 GB expert cache) beat manual
  64-75 GB settings on a short streaming decode bench; no absolute tok/s or
  hit rates are published — the profiler exists to measure them per
  deployment. Non-streamed reference points: PRO q2 9.56 tok/s decode @32k
  (M3 Ultra 512 GB), Flash q2 13.75 tok/s decode on DGX Spark GB10
  (README.md:162-175). Hit/miss/pread-bytes/pread-ms are runtime log counters
  (ds4.c:2738-2828), not published results.

**Portable to us**: the resident/streamed split (non-routed resident, routed
experts tiered); byte-or-count budget + auto plan; hotness-decayed-LFU
eviction with in-flight protection; the pread pool; a synchronous phase-1
router-readback/load boundary; chunked prefill streaming + last-K decode seed;
locality profiler + hotlist preload; the simulate-used-memory honesty tool;
CLI semantics. ds4's resident-first/missing-deferred overlap is a later
candidate, not a mechanical port: our selected GB10 Marlin kernel uses dense
expert strides and has no address table or pair-mask input.
**ds4-specific, not portable**: Metal no-copy mmap buffer wrapping +
`F_RDADVISE`; hash-layer lookahead (DeepSeek `tid2eid`; Qwen3.6 has none);
top-6 bitmasks (we are top-8). GGUF 3D-tensor slicing arithmetic was listed
here as non-portable on the grounds that safetensors, where each expert is a
separate tensor, is simpler for us. That is now WRONG in both directions and is
corrected below.

## Upstream chain

Pinned vLLM `e24d1b24` (verified `git log -1` 2026-07-10). The three senses
of "streaming", settled with citations — one-line verdicts:

1. **LOAD-TIME weight streaming (fast startup, full residency after): PRESENT
   in-pin — NOT this feature.** `LoadFormats` registry incl. `runai_streamer`
   (`vllm/model_executor/model_loader/__init__.py:33-49`, loader map :50-66);
   `RunaiModelStreamerLoader`
   (`vllm/model_executor/model_loader/runai_streamer_loader.py:21`;
   iterator `weight_utils.py:987`); safetensors mmap/lazy strategy
   (`vllm/config/load.py:63-84`), `sharded_state_loader.py`, fastsafetensors
   (`weight_utils.py:1024`, `default_loader.py:267-268`). Mirror obligation
   already tracked by `LOAD-SAFETENSORS`/`LOAD-LONGTAIL`.
2. **INFERENCE-TIME CPU weight offload (`cpu_offload_gb`): PRESENT in-pin,
   blanket per-parameter, v1-supported, NOT expert-router-aware.**
   `UVAOffloadConfig.cpu_offload_gb` (`vllm/config/offload.py:23`); UVA
   offloader `_maybe_offload_to_cpu` per-parameter pinned+zero-copy
   (`vllm/model_executor/offloader/uva.py:64,80-108`); creation
   (`offloader/base.py:126-162`), installed in the v1 worker
   (`vllm/v1/worker/gpu_model_runner.py:445,913` — no v1 guard/raise).
   `cpu_offload_params` allows opt-in name-segment targeting of expert
   tensors (`config/offload.py:34-44`) but offload is static — no router
   keying, no per-token paging decisions. A second backend, layer-group
   `PrefetchOffloader` (`offloader/prefetch.py`; config
   `config/offload.py:47-76`), is CPU-only (`prefetch.py:557-560`). Mirror
   obligation recorded as `ENG-WEIGHT-OFFLOAD` (INVENTORIED; own future
   spike `specs/weight-offload-uva.md`).
3. **INFERENCE-TIME disk/SSD-tier weight paging (expert/layer granular):
   ABSENT in-pin.** Searched `vllm/model_executor/layers/fused_moe/`,
   `vllm/model_executor/offloader/`, `vllm/config/offload.py`,
   `model_loader/` for offload/nvme/io_uring/disk/ktransformers/expert-cache:
   nothing pages weights from disk during decode. `eplb` is expert
   REBALANCING across EP ranks, not offload
   (`vllm/distributed/eplb/eplb_state.py:3-27`); `ep_weight_filter.py:3-8`
   skips non-local experts at LOAD time only; the only disk tier in-pin is
   for KV cache (`vllm/v1/kv_offload/tiering/fs/manager.py:76`), covered by
   `KV-OFFLOAD`.

**Consequence**: `ENG-EXPERT-STREAM` is a **surpass-track** feature (roadmap
"mirror as the floor, surpass beyond it"). Upstream-sync safety: the feature
is (a) strictly additive and default-off, (b) implemented behind our loader +
MoE dispatch seams (no upstream-mirrored file is restructured), (c) its
config namespace is ours (`expert_streaming.*`), kept disjoint from vLLM's
`offload_config` so a future mirror port of `ENG-WEIGHT-OFFLOAD`/sense-2
lands without collision, and (d) recorded here + in the ledger at
implementation time as a surpass divergence, so the recurring sync cycle
(`upstream-sync.md`) diffs cleanly against the pin. If upstream later grows
expert-granular offload, that becomes the mirror target and this row gets
re-based on it (same rule as GGUF vs llama.cpp).

Design-reference chain (non-vLLM, cited per the ground-every-impl rule):
antirez/ds4 @ 80ebbc3 — files/lines inventoried in §ds4-anatomy above. Ports
from ds4 carry `// Ported from: antirez/ds4 <file>:<line> @80ebbc3` headers
exactly like vLLM-derived files.

**Runtime trace plan (required because dispatch is dynamic).** Before W1, run
`nsys profile` on the non-streamed 35B c1 workload and record the actual router,
`moe_align`, Marlin w13/w2, repack, allocation, and synchronization kernels plus
their wall time/launch order. W0 also runs an instrumented router-only probe that
copies `dtid` after `MoeRouterTopK` and proves the device→host event boundary.
At W3, trace streaming-off and phase-1 streaming-on in one same-binary series:
the OFF trace must remain structurally identical; the ON trace must show one
intentional D2H/event wait per MoE layer, bank reads/copies before slot use, and
no full-resident repack/allocation. At W4, trace each prefill slot sweep and
verify every routed pair is computed exactly once. Kernel names and steady-state
times, not source inference, are the acceptance evidence.

## Our baseline

- **Checkpoint (measured from the real safetensors header, dgx 2026-07-10)**:
  `nvidia/Qwen3.6-35B-A3B-NVFP4`, 40 MoE layers x 256 experts, top-8 + 1
  shared (`config.json`: num_experts=256, num_experts_per_tok=8,
  moe_intermediate_size=512, hidden 2048). Per expert per layer: gate
  U8[512,1024] 524,288 B + FP8 scales [512,128] 65,536 B + f32 scale2;
  up identical; down U8[2048,256] 524,288 B + scales 65,536 B + f32 scale2
  → **1,769,484 B ≈ 1.688 MiB**. Routed-expert total is **16.88 GiB of the
  ~22 GiB checkpoint (~77%)**. Each projection is a separate safetensors
  entry, but our reader exposes only an mmap pointer/length through `StTensor`
  (`safetensors_reader.h:13-47`); it does not expose the owning shard path, fd,
  data-section base, or absolute offset for later pread. The bank builder must
  consume those spans while shards are alive or extend the reader metadata.
- **Our MoE execution path (the seams)**: `LoadNvfp4Raw` copies every packed
  weight and scale span into host `OwnedTensor` storage
  (`qwen3_5_weights.cpp:196-223,276-298`), and the model retains those copies.
  CUDA then uploads per-expert originals, repacks ALL E experts once into one
  contiguous Marlin layout — fused w13 `[E, H/16, 4I]`, w2
  `[E, I/16, 2H]`, contiguous scale arrays and `[E]` global scales —
  synchronizes, and frees only the temporary DEVICE originals
  (`qwen3_5.cpp:2392-2549`). Decode runs grouped
  `moe_wna16_marlin_gemm` over `moe_align` outputs
  (router `MoeRouterTopKKernel` `src/vt/cuda/cuda_moe.cu:47-127`;
  `MoeAlignKernel`/`MarlinMoeAlignBlockSize`
  `src/vt/cuda/cuda_marlin_repack.cu:203-293`). The Marlin kernel reads
  `expert_ids_ptr[block]`, indexes `global_scale[expert_id]`, and computes
  `B_expert_off = expert_id * prob_n * prob_k / (pack_factor*4)`
  (`marlin_template.h:543-550`): runtime storage is dense-stride, not a pointer
  table. Router IDs stay on device (`cuda_moe.cu:54-128`).
- **Honest gaps**: (1) nothing in the engine can drop or re-load expert
  weights after load — residency is all-or-nothing; (2) retaining the current
  host `OwnedTensor` expert vectors would consume the bytes streaming is meant
  to free on GB10 even if the device cache were smaller; (3) a naive checkpoint
  pread would still require the Marlin repack/scale-processing kernels per miss,
  including layer-wide combined scale factors; (4) `MoeMarlinResident` is a
  process-static full-E allocation keyed by `MoeBlockWeights*` and must be
  bypassed, not partially populated; (5) router IDs require an explicit D2H
  synchronization before host cache decisions; (6) memory accounting
  (`src/vllm/entrypoints/model_loader.cpp:117-129`) assumes static weights;
  (7) we have no expert-locality data for Qwen3.6 routing; (8) c1 decode tok/s
  on dgx is not yet recorded in the ledger
  (the online-serving campaign `SERVE-GATE-ONLINE` is in flight) — the W6
  campaign must measure the non-streamed c1 baseline first.

### The honest bandwidth/locality math (decides viability)

Measured storage (dgx.casa `/dev/nvme0n1`, ext4 root, 2026-07-10, this
spike): sequential O_DIRECT 16 MiB blocks **5.4 GB/s**; random expert-sized
(1,769,472 B) O_DIRECT preads: **2.76 GB/s @1 thread (0.64 ms/read), 5.02
GB/s @4, 5.03 GB/s @9, 5.29 GB/s @16** — a small pool saturates the disk at
~5 GB/s; commands recorded in state.md. (`/tmp` is ext4 on this same NVMe —
tmpfs would be unified LPDDR5X, i.e. the memory we are trying to save.)

Per-token worst case (batch 1, zero hits): 8 experts x 40 layers x 1.688 MiB
= **540 MiB/token** → I/O-only ceiling 5.0 GB/s / 540 MiB = **8.8 tok/s**.
With hit rate h the bound scales as 1/(1-h):

| resident fraction f (bytes) | uniform-routing h≈f | I/O-only tok/s bound | memory freed vs full residency |
|---|---|---|---|
| 0.25 (4.2 GiB) | 0.25 | 11.8 | 12.7 GiB |
| 0.50 (8.4 GiB) | 0.50 | 17.7 | 8.4 GiB |
| 0.75 (12.7 GiB) | 0.75 | 35.4 | 4.2 GiB |
| 0.90 (15.2 GiB) | 0.90 | 88.5 | 1.7 GiB |

Uniform routing is the conservative cache-hit model; routing skew + hotness
caching + consecutive-token locality may push h above f, but W0/W5 must measure
that rather than assume it. The table is an I/O-only upper bound: c64 TPOT does
not identify c1 compute time, and phase 1 deliberately adds 40 router-readback
waits per token. The **≥12 tok/s at f=0.5** target remains a gate derived from
the 17.7 tok/s I/O ceiling, not a predicted result; W0 records the non-streamed
c1 floor before implementation and no overlap credit is assumed until traced.

High-batch regime (why the gate is out of scope): B independent top-8 draws
touch ~256x(1-(31/32)^B) experts per layer per step — B=16: ~102 (40%),
B=32: ~163 (64%), B=64: ~225 (88%), B=128: ~250 (98%). Per-step miss bytes
approach (1-f) x 16.88 GiB; at f=0.5, B=64 that is ~6.3 GiB/step ≈ 1.26 s
of I/O per step ≈ 51 tok/s aggregate — vs ~2.8k tok/s gated non-streamed
(state.md:1275). No reordering changes the bytes. Streaming therefore serves
capacity, not throughput; the engine must say so at configuration time.

The 27B gate model is dense — this row does not apply to it. The larger
strategic reach: MoE checkpoints BIGGER than GB10's 119 GB unified memory
(DeepSeek-class, GLM-4.x-class at 2-4 bit) become runnable at all — pinned
vLLM has no answer on this hardware (sense-3 absent), which is the
surpass-track headline.

## Port map

New code goes under our seams; no upstream-mirrored file is restructured.
"Ported from" = design source to cite in headers (ds4 @ 80ebbc3).

| Source (design) | Local target | Notes |
|---|---|---|
| ds4_ssd.c:46-106 (budget parse + auto plan), ds4_ssd.c:108-181 (simulate-used-memory) | `src/vllm/model_executor/expert_stream/budget.{h,cpp}`; probe tool `tools/expert_stream/simulate_used_memory.cpp` | count-or-`NGB` parsing, auto plan from free device memory minus non-streamed needs; honesty tool for W6 |
| ds4_metal.m:370-458, 8678-8776, 9653-9830 (entry table, slabs, hotness decay, victim scan, protection); ds4_cuda.cu:156-198, 1967, 2011-2098 (device slabs, LRU) | `src/vllm/model_executor/expert_stream/expert_cache.{h,cpp}` + `src/vt/cuda/cuda_expert_cache.cu` | CPU metadata maps `(layer, logical_expert) -> slot`; CUDA owns fixed contiguous arrays for C Marlin slots (`w13`, `w2`, both processed scales, both global scales), because the kernel addresses `base + slot*stride`. Never allocate unrelated per-entry device pointers. Hotness-decayed LFU (decay every 16 decode tokens), LRU tiebreak, in-flight + selected protection |
| ds4_metal.m:7699-7995 (pread task/pool), ds4_cuda.cu:1055-1104 (O_DIRECT staging), 1262 (async upload) | `src/vllm/model_executor/expert_stream/pread_pool.{h,cpp}` | persistent pool (default 9, cap 18 threads), one fixed-size bank-entry task per missing expert; O_DIRECT into aligned pinned staging, then explicit copy-stream upload into the chosen contiguous slot. GB10 is unified memory but current cache arrays are `cudaMalloc`; copy cost/overlap is measured, never assumed away |
| ds4_metal.m:10618-10800 (begin_selected_load), 23392-23433 (resident/missing split) | streaming branch beside `MoeBlockFusedMarlinCuda` in `src/vllm/model_executor/models/qwen3_5.cpp`, before the current full-E `MoeMarlinResidentFor` build | Phase 1: run router; async D2H `dtid` + event synchronize; update hotness/dedupe logical IDs; load/evict until all selected experts are resident; rewrite a device copy of top-k IDs from logical IDs to slot IDs; run `MarlinMoeAlignBlockSize(..., num_experts=C)` and unchanged Marlin GEMMs over C-slot tensors. OFF path is byte- and trace-identical. Resident-first/missing-deferred overlap requires new pair partition/accumulation kernels and is out until phase-1 profiling justifies a separately spiked leaf |
| ds4.c:11696-11760 (full-layer streaming prefill + lookahead thread), 13812-13824 + 19657 (decode cache seed from last ≤64 prefill tokens) | same streaming branch, prefill mode | full-layer residency is impossible when C<E. Sweep the bank in chunks of at most C logical experts: filter routed pairs for the chunk, load contiguous slots, run aligned Marlin, scatter/accumulate each pair exactly once, then advance; pipeline next chunk reads only after correctness. Seed decode cache from the last-K prompt selections |
| ds4.c:750-1207 (expert locality profiler + hotlist write), ds4.c:13949-14016 (preload), ds4_streaming_hotlist.inc | `tools/expert_stream/expert_profiler` (+ optional hotlist preload in expert_cache) | measures per-layer histograms, latest-N cache simulation, adjacency; produces hotlist for preload and the h(f) curve for gates |
| (ours, no ds4 counterpart) | `tools/expert_stream/build_expert_bank.cpp`; source-metadata extension in `safetensors_reader.{h,cpp}`; streaming loader branch near `qwen3_5_weights.cpp:196-223,276-298,326-339` | one-time offline/first-load repack of all experts to a **Marlin-layout bank**. Builder consumes mmap spans while each owning shard lives (or explicit path/absolute-offset metadata), computes layer-wide w13/w2 combined scale factors, and writes fixed-size per-expert records plus manifest. Runtime streaming mode loads only manifest/non-expert weights: it must not populate `expert_{gate,up,down}_fp4` host bytes or call `BuildMoeMarlinResident`. Bank key covers checkpoint content, architecture, shapes, fused-vs-split w13 mode, Marlin layout ABI and target arch; derived bank is never redistributed |
| vLLM config style (`vllm/config/offload.py:15-44` as the shape reference) | `include/vllm/config/expert_stream.h` + plumb through `src/vllm/entrypoints/model_loader.cpp:117-129` memory accounting | our namespace `expert_streaming.*` (see upstream-sync safety); memory accounting learns "streamed expert bytes not resident" |

## Tests to port

Upstream vLLM has NO tests for inference-time disk expert paging (feature
absent in-pin — §Upstream chain sense 3); per test-porting.md the executable
spec here is (a) our own token-exactness/e2e gates in the established
harness patterns and (b) upstream tests for the ADJACENT mirror row recorded
for its own future spike, not this one.

| Test | Source / pattern | Local tier / target | Status plan |
|---|---|---|---|
| Streaming-vs-resident token exactness (same checkpoint, same seed, 16/16 greedy; f ∈ {0.25, 0.5, 1.0-ε}) | pattern: `tests/parity/test_qwen36_paged_engine.cpp:140` gate | T-parity, new `tests/parity/test_qwen36_expert_stream.cpp` (checkpoint-gated, dgx) | with W3 |
| Expert cache unit semantics: budget parse (count/`NGB`), auto plan subtraction, hotness decay halving, victim = min-hotness/LRU-tiebreak, in-flight + selected protection, logical→slot mapping, slot reuse, C<E bounds | ds4 semantics at ds4_ssd.c:46-106 and ds4_metal.m:8678-8776, 9653-9830 (cited in test header) | T-unit doctest, new `tests/vllm/test_expert_cache.cpp` (CPU-only, mock storage) | with W1 |
| pread pool: task fan-out, O_DIRECT alignment, failure propagation, shutdown | ds4_metal.m:7776-7995 | T-unit, `tests/vllm/test_expert_pread_pool.cpp` (tmp files, CPU-only CI) | with W2 |
| Expert bank builder: shard ownership/offset bounds, manifest-key rejection, layer-wide scale factors, and per-expert byte identity vs the existing full-resident Marlin repack | ours, current reference `qwen3_5.cpp:2392-2549` | T-unit + real-checkpoint golden compare, `tests/vllm/test_expert_bank.cpp` | with W2 |
| Slot-remap kernel: random logical top-k IDs + mappings become valid `[0,C)` IDs; `moe_align` + w13/w2 outputs equal an E-resident reference; unmapped ID hard-fails before GEMM | current Marlin dense-stride contract `marlin_template.h:543-550` | T-unit CPU reference + CUDA op parity | with W3 |
| Prefill chunk sweeps + decode seed: every routed pair appears exactly once across chunks; accumulated output equals full-resident path; cache state after prefill equals last-K selections | ds4.c:11696-11760, 13812 adapted to dense-slot Marlin | T-unit with mock router + checkpoint-gated e2e assert | with W4 |
| Regime guard: conc>bound warns/refuses; dense model rejects flag | ours (product rule from §3) | T-unit config + `tests/vllm/entrypoints/openai/test_conformance.cpp` addition | with W3 |
| `ENG-WEIGHT-OFFLOAD` mirror tests — `tests/basic_correctness/test_cpu_offload.py:11` (UVA/pin-memory matrix), `tests/quantization/test_cpu_offload.py:18,32,48,64` | upstream | NOT this row — inventoried for `specs/weight-offload-uva.md` | recorded only |

## Gates

All GPU work under the coordination GPU-lock rules; benchmark arms on an
idle box, ≥2-3 reps, exact commands into the ledger.

| Gate | Requirement | Exact command sketch |
|---|---|---|
| G1 token exactness (precondition, never traded) | Streaming on (f=0.5 and f=0.25) is 16/16 greedy token-exact vs the SAME build with streaming off, 35B gate corpus; and the streaming-off build stays 16/16 vs the vLLM oracle as today | `flock /tmp/gpu -c './tests/parity/test_qwen36_expert_stream --resident-frac 0.5'` (dgx) |
| G2 measured memory reduction | At f=0.5: peak whole-system used memory AND CUDA allocation high-water are ≥7.5 GiB below non-streamed. Prove the streaming loader retained no routed-expert host `OwnedTensor` bytes and allocated no full-E `MoeMarlinResident`. On GB10, `nvidia-smi` alone is insufficient: record `/proc`/`free`, process RSS/PSS, `cudaMemGetInfo`, and page-cache baseline; use O_DIRECT plus `simulate_used_memory` so file cache cannot fake the win | A/B same binary, fresh process per arm: `--expert-streaming off` vs `cache=8.4GB` |
| G3 tok/s floor at stated resident fraction | c1 decode (1024-token prefill, 128 decode, greedy) on dgx NVMe at f=0.5: **≥ 12 tok/s**, with the full measured curve published for f ∈ {0.25, 0.5, 0.75, 1.0-ε} incl. hit rates (I/O math §3 supports 17.7 uniform; 12 leaves compute+sync margin — a miss on 12 means the overlap machinery, not the concept, failed) | bench harness c1 arm x3 reps + cache hit-rate counters |
| G4 prefill regression bound | 8k-token prefill with streaming ≤1.5x non-streamed at f=0.5. Trace chunked slot sweeps: each routed pair exactly once, no full-layer resident allocation, and next-chunk overlap only if separately measured correct | same harness, prefill-heavy arm + nsys |
| G5 regime/off-path honesty | conc>4 warns/refuses; streaming mode disables CUDA graphs explicitly in phase 1; with feature off, 35B and 27B large-concurrency throughput/TTFT/TPOT/peak memory and kernel trace are unchanged within reproduced noise against fresh vLLM denominators | conformance + same-binary OFF-vs-parent A/B, both gate models, 3 reps |
| G6 record closure | Matrix row anchors + ledger row + README capacity-mode note + state.md, per DoD | `python3 scripts/check-agent-record.py` |

## Dependencies

| Dependency | Why | State |
|---|---|---|
| dgx.casa NVMe + 35B NVFP4 checkpoint | the only MoE gate model; bank build needs ~17 GiB free disk (346 GB free, headroom rule OK) | available (measured this spike) |
| Marlin fused-w13 MoE path (`VT_MOE_FUSED_W13`, `qwen3_5.cpp:2392-2652`) | bank records and fixed slots must match its fused layout, layer-wide scale factors, dense expert stride and global-scale indexing | merged, gated; exact contract now cited |
| Safetensors source metadata + bank-only loader representation | current `StTensor` lacks public owning path/fd/absolute offset, and `LoadNvfp4Raw` always copies bytes | W2 owns the narrow metadata/representation change; hard dependency before paging |
| Router D2H + logical-to-slot op | current `dtid` is device-only and Marlin consumes dense expert IDs | W3; phase 1 intentionally non-graphed |
| `SERVE-GATE-ONLINE` c1 baseline numbers | G3 needs the honest non-streamed c1 denominator | in flight (`CLAIM-SERVE-GATE-1`); W6 can measure its own baseline if still open |
| `QUANT-GGUF-KEEPQ-LOADER` (`specs/gguf-keep-quant-loader.md`) | **SATISFIED, not pending.** L2+L3 landed `429e19d6a` 2026-07-22 and shipped the per-expert row slicer this spec was waiting for; still not needed for safetensors/W1-W6 | L1/L2/L3/L5/L6 landed |
| No new third-party deps | plain pread/O_DIRECT + pthreads + CUDA runtime; io_uring explicitly NOT required (measured 5 GB/s with 4 threads) | - |
| Licenses | ds4 is MIT with llama.cpp/GGML attribution retained; we port design + cite, keeping our header discipline | OK |

## Work breakdown

Claim-sized, non-overlapping leaves. W0 is evidence before implementation;
W1 is CPU-only; W2 has CPU manifest tests plus one GPU golden; W3-W7 are the
critical runtime/gating path.

| Leaf | Scope (files above) | Depends on | Gate slice |
|---|---|---|---|
| W0 trace + c1 baseline | nsys current 35B router/align/Marlin/repack; router D2H boundary probe; fresh non-streamed c1 throughput/latency/memory; bank-layout byte accounting | - | committed evidence; no code state claim |
| W1 cache policy + budget | CPU `expert_cache` metadata and `budget`: logical→slot mapping, LFU/LRU/protection, no device allocation; unit tests | W0 | CPU tests green |
| W2 bank format/builder + reader + pread pool | safetensors source metadata, layer-wide scale-factor/full-resident byte golden, versioned bank manifest, bank-only weight representation, O_DIRECT pool and aligned staging. Does not change live dispatch | W0 | CPU tests + real-bank golden; build/startup/RSS checkpoint vs parent |
| W3 phase-1 decode dispatch | fixed contiguous C-slot CUDA arrays; router D2H/event wait; ensure-resident; logical→slot rewrite; align with C; unchanged Marlin GEMMs; config/memory accounting/regime guard; bypass full-E resident | W1,W2 | G1 + G2 + G5; nsys contract; own performance checkpoint |
| W4 prefill chunk sweeps + decode seed | routed-pair filter/scatter accumulation over ≤C experts per pass; last-K seed; only then measured next-chunk I/O overlap | W3 | G4 + exact pair coverage |
| W5 locality profiler + hotlist preload | h(f)/adjacency report for 35B corpus, optional preload | W3 (parallel with W4) | published curve; no gate rebasing |
| W6 optional resident/missing overlap spike | only if W3 trace shows wait dominance: inventory pair partition/accumulation kernel and graph implications before code | W3 evidence | separate accepted spike required |
| W7 per-regime campaign + closure | c1 curve, prefill, whole-system memory, OFF-path both-model vLLM A/B, ledger/README/matrix/state | W3-W5 | G3 + G6, all benchmark-protocol axes |
| W8 post-gate host-RAM tier | bank-file vs pinned-host backing store on discrete GPU; distinct from vLLM `cpu_offload_gb` mirror | W3 | separate hardware mini-gate |

## GGUF reconciliation (2026-08-14, issue #824)

**This spec was written 2026-07-10 and put GGUF expert streaming out of scope
because it "needs per-expert slicing of 3D tensors" and had to wait for
`QUANT-GGUF-KEEPQ-LOADER`. That slicer landed 2026-07-22, twelve days later,
under exactly that row, and is now the production decode path.** The record was
never reconciled, so until this note the spec directed a reader to wait for
something that already existed.

What exists, with anchors re-derived at HEAD:

- `OwnGgufQuantBlocks(tensor, n, k, row_offset, ...)`
  (`src/vllm/model_executor/models/qwen3_5_gguf_weights.cpp:57`) slices a stacked
  GGUF quant tensor by ROW OFFSET, deriving the byte offset from
  `vt::RowSizeBytes(dt, k)` and bounds-checking it against `tensor.nbytes`. Landed
  `429e19d6a` (L2+L3), refined `f6be46eda` (L6) and `d967733d2` (NVFP4).
- `GemmRowSlice(be, w, x, T, N, K, row_off)`
  (`src/vllm/model_executor/models/deepseek_v4.cpp:468`) runs a keep-quant GEMM
  against a row-slice of a stacked `[E*out, K]` weight, on DEVICE when the weight
  is block-quant.
- It is already called PER EXPERT at runtime: `deepseek_v4.cpp:1004,1005,2487,2489,2493`
  with `row_off = e * mi`, and `laguna.cpp:1225` with `id * moe_I`.
- The layout contract is stated on the role itself: `gguf_keep_quant.cpp:27`,
  `kStackedExpertWeight` is `[E, out, in]` and "each expert slice is whole rows of
  the same K".

**The consequence is a scheduling inversion, not just a tidy-up.** This spec
designs a safetensors -> Marlin BANK (versioned bank file, manifest, repack)
because that path needs one: safetensors experts are separate tensors in a
non-Marlin layout. On the GGUF path the mmap'd file ALREADY IS the bank — block
quantized, per-expert rows contiguous, offsets computable, and the GEMM already
takes a base pointer plus a row offset. No bank build, no manifest, no repack,
and no second on-disk copy of the expert bytes (the ~17 GiB cost this spec
accepts under Risks for the safetensors lane).

So the GGUF lane is plausibly the CHEAPER of the two, and it is the one that
matters commercially: every Qwen3.8-class checkpoint that exists today is GGUF
(`unsloth/Qwen3.8-2.4T-A95B-GGUF`), so nothing in that lane runs at all until
GGUF is served.

**What genuinely remains for GGUF is the residency policy, not the slicer.**
Today every caller passes `row_offset=0` and makes the WHOLE stacked
`[E*out, K]` tensor resident; streaming needs a bounded cache that materializes
only the touched expert slices and evicts under a byte budget. That is W1's
cache policy applied to a different backing store, which is what the original
"same cache, different slicer" note was reaching for — it was right about the
cache and wrong about needing a new slicer.

This note does NOT re-scope W1-W7, claim the GGUF leaf, or promise a date. It
corrects the record so the next person to cost that leaf starts from what is
true. Re-verify these anchors before implementing: they were re-derived at HEAD
on 2026-08-14 and this spec has already been wrong about them once.

## Target checkpoint census and the quant encodings it needs (2026-08-15, issue #912)

The GGUF reconciliation note above establishes that the GGUF lane needs no bank
and only a residency policy. It did not ask a prior question: **can this tree
decode the bytes of the checkpoint the row exists to run?** Measuring that
changed the target.

Every published GGUF of `unsloth/Qwen3.8-2.4T-A95B-GGUF` at revision
`567d3e6ac26c5474b18311e619c04350fb9a5556` was censused by parsing tensor
headers directly (HTTP range requests, no full download). Coverage is total, not
sampled: **1702 tensor records parsed against 1702 declared in
`split.tensors.count`.**

`UD-Q1_0` (370 GiB), the smallest quant and the one first chosen for this row,
is **not decodable by any admissible oracle** and was rejected:

- Its expert tensors carry ggml type **66**, holding ~97 % of all parameters.
- Type 66 exists in neither the pinned llama.cpp (`237ad9b96`, whose enum ends
  at `GGML_TYPE_Q1_0 = 41`) nor upstream `ggml-org/llama.cpp` master
  (`ad1de39e0`, 2026-08-15, ends at `GGML_TYPE_Q2_0 = 42`, `COUNT = 43`).
- It is defined only on the fork branch `unslothai/llama.cpp @ iq1-narrow` as
  `GGML_TYPE_IQ1_XXXS = 66, // 1.1875 bpw, 256-entry grid`. The bits-per-weight
  derived independently here from GGUF offset deltas is **1.1875**, matching that
  declaration exactly, which is what confirms the identification.

**That refusal was overruled by developer direction on 15 August 2026, and the
fork is now pinned.** The refusal rested on a policy premise, no upstream
definition and no pinned oracle, rather than on a technical one. The fork is
public and pinnable, so the developer directed anchoring the encoding to it, and
that resolves the objection at its root: the encoding now HAS a recorded
upstream and a fixed revision to cite.

`unslothai/llama.cpp` is therefore admitted to the oracle table as
`llama-cpp-unsloth`, pinned at `36fe8e1cc` on branch `iq1-narrow` and scoped to
the sub-IQ1_S encodings alone. It records `gateable = no`, because the port was
grounded in the fork's SOURCE rather than in a running comparison, and issue
#933 owes the build-and-run measurement. See
[`.agents/oracles/llama-cpp-unsloth.md`](../oracles/llama-cpp-unsloth.md).

Both checkpoints are therefore targets, and the order is UD-Q1_0 first per
developer direction, then UD-IQ1_S. The two are structurally IDENTICAL, censused
the same way over all shards, 1702 records against 1702 declared in each: same
276 expert tensors, same 96.92 %, same six other encodings with the same counts.
Only the expert encoding differs, ggml 66 against ggml 19. So the same streaming
lane serves both, and each needed exactly one new encoding.

The census is the scope statement for both targets, `UD-Q1_0` (370 GiB) and
`UD-IQ1_S` (473 GiB). Support status is read from the CODE rather than from
comments (`BlockDTypeFromGgmlTypeId` in `src/vt/dtype.cpp`, `BlockVecDot` in
`src/vt/cpu/cpu_quant_dot.cpp`, and the `DType` enum itself). That distinction
earned itself: a stale header comment listing six encodings led this section to
record Q2_K as missing, and reading the dispatch showed it has been served all
along.

Both censuses, side by side. The `UD-Q1_0` row set is identical apart from the
expert encoding:

| Encoding | ggml id | Tensors | Gparams | % params | Status |
|---|---|---|---|---|---|
| IQ1_XXXS (`UD-Q1_0` experts) | 66 | 276 | 2370.8 | **96.92** | added, fork-anchored |
| IQ1_S (`UD-IQ1_S` experts) | 19 | 276 | 2370.8 | **96.92** | added |
| Q5_K | 13 | 420 | 34.0 | 1.39 | served (`VecDotQ5_KQ8_K`) |
| Q2_K | 10 | 3 | 25.8 | 1.05 | served (`VecDotQ2_KQ8_K`) |
| Q6_K | 14 | 162 | 10.8 | 0.44 | served (`VecDotQ6_KQ8_K`) |
| Q4_K | 12 | 2 | 4.1 | 0.17 | served (`VecDotQ4_KQ8_K`) |
| Q8_0 | 8 | 1 | 0.1 | 0.01 | served |
| F32 | 0 | 838 | 0.5 | 0.02 | served (not keep-quant) |

**Each checkpoint was missing exactly ONE encoding, and it was 96.92 % of the
model.** In both, the expert tensors are precisely the 92 non-MTP layers times
three expert matrices (`ffn_down_exps`, `ffn_gate_exps`, `ffn_up_exps`), and the
Q2_K trio is block 92, the `nextn` MTP layer. So one encoding per checkpoint
makes a complete 92-layer forward pass loadable, and nothing else gates the
first benchmark.

IQ1_S is an ordinary upstream k-quant at the existing pin, and this tree already
carries the whole grid-table pattern it needs, built for DeepSeek-V4 under
`.agents/specs/gguf-iquant-dsv4.md`: `kIQ2_XXS`, `kIQ3_XXS`, `kIQ2_S` and
`kMXFP4` each have a keep-quant `vec_dot` against the Q8_K activation encoding,
CPU grids in `src/vt/cpu/cpu_quant_iq_tables.h` and device grids in
`src/vt/cuda/cuda_quant_iq_tables.cuh`. IQ1_S adds one more row on those rails.

Upstream anchors, re-derived at the pin `237ad9b96` (re-verify before
implementing, per the reconciliation note's own warning):

| Piece | Anchor |
|---|---|
| `block_iq1_s` (`ggml_half d; uint8_t qs[32]; uint16_t qh[8]`) | `ggml/src/ggml-common.h:414-419` |
| block size assert, 50 bytes per 256 elements = 1.5625 bpw | `ggml/src/ggml-common.h:420` |
| `ggml_vec_dot_iq1_s_q8_K_generic` | `ggml/src/ggml-cpu/quants.c:1099` |
| `iq1s_grid`, uint64, `NGRID_IQ1S` = 2048 | `ggml/src/ggml-common.h:1124` |
| `iq1s_grid_gpu`, uint32, 2048 | `ggml/src/ggml-common.h:1639` |

### The IQ1_XXXS decode was checked against REAL checkpoint bytes

Synthetic random blocks sweep bit patterns, but they cannot catch a
misunderstanding of the FILE: a wrong field order, a wrong stride, or a codebook
that turns real weights into legal-looking noise. So the decode was also run on
the downloaded `UD-Q1_0` shards, 15 August 2026.

Two independent things were compared on the SAME bytes: this tree's
`DequantIQ1_XXXS`, and a separate transcription of the fork's
`dequantize_row_iq1_xxxs` reading the fork's grid directly out of its own tree.

| Tensor | Layer | K | Weights | Result |
|---|---|---|---|---|
| `ffn_gate_exps` | 0 | 8192 | 524288 | bit-identical |
| `ffn_up_exps` | 0, 23 | 8192 | 262144 | bit-identical |
| `ffn_down_exps` | 0, 23 | 2048 | 65536 | bit-identical |

**1179648 real weights, max absolute difference 0.0.** The decoded values also
look like weights rather than noise: mean -4.2e-7, sd 8.6e-4, symmetric tails,
24 discrete levels, and no non-finite value. The file layout resolves exactly as
the port assumes, `row_bytes = K/256*38`, and the implied tensor size of
1275068416 bytes matches 512 experts times 2048x8192 at 1.1875 bpw.

Be precise about what this does and does not establish. It removes transcription
error from OUR C++, which is the failure this port was most exposed to, and it
proves the reader addresses the real file correctly. It does NOT make the fork
oracle gateable: both sides are transcriptions of the same source, so a defect in
the FORK would be reproduced identically by both. Only building and running the
fork closes that, which is what #933 owes.

Why this is in this spec rather than its own row: the encoding is the load
path's half of the same capability. A streamer that can address an expert slice
it cannot decode moves bytes for nothing, so the row's own gate cannot be met
without it. It is nonetheless useful beyond this row, since any low-bit GGUF
gains it, which is the standing k-quant obligation in `AGENTS.md` rather than a
detour for one model.

This note does not re-scope W1-W7 or claim a date. It adds the leaves below and
records why the target checkpoint changed.

| Leaf | Scope | Depends on | Gate slice |
|---|---|---|---|
| W9 IQ1_S decode | `DType::kIQ1_S`, geometry `{256, 50, 19}`, `BlockDTypeFromGgmlTypeId` row, CPU `iq1s_grid` + `VecDotIQ1_SQ8_K` traits row, dequant path | - | CPU tests green against upstream-derived vectors; `KeepQuantDType(19)` true |
| W10 IQ1_S device decode | `iq1s_grid_gpu` + CUDA `vec_dot`, so streamed slices dot on device. NOT optional polish: `cuda_quant_dot.cu:1531` maps an unknown weight dtype to `return false`, which is a SILENT CPU fallback. Tokens would still be correct, so no token gate can see it, and a 2.4 T model would simply run at CPU speed while looking healthy | W9 | CUDA tests; parity with the CPU arm; the fallback must be observable rather than inferred |
| W11 checkpoint load | UD-Q1_0 first, then UD-IQ1_S, loads end to end on dgx.casa, refusing any encoding it cannot honour by name. BLOCKED on W4: measured 16 August 2026, the load needs memory of the order of the whole file (4 GiB per 110 s, linear, no plateau) and cannot complete in 119 GiB | W9, W10, W12, W4 | model loads; token output captured |
| W4a allocation attribution | **DONE 16 August 2026.** The allocator is the GDN V-head reorder forcing `attn_qkv` and `ssm_out` to `kTransformedWeight`, which expands them to bf16: about 50 GiB across 93 layers. Experts were never the problem; they borrow with 0 copies | - | measured, see the census section |
| W4b transformed-weight residency | Stop the V-head reorder from costing ~50 GiB. Either apply the reorder at RUNTIME so the stored blocks stay borrowed, or confine it inside block boundaries so a block-typed weight survives it. Without this, streaming the experts still does not fit | W4a | the two projections stay block-resident, measured on the same load |
| W12 IQ1_XXXS decode | `DType::kIQ1_XXXS`, geometry `{256, 38, 66}`, the 256-entry fork codebook, `VecDotIQ1_XXXSQ8_K`, dequant, traits row and the reader's `case 66`. Grounded in the pinned fork oracle, cited per site | - | CPU tests green; `KeepQuantDType(66)` true; grid digest sealed |
| W13 IQ1_XXXS device decode | the CUDA arm, for the same silent-fallback reason as W10 | W12 | CUDA tests; parity with the CPU arm |

## Loading a 370 GiB split GGUF: the recipe, and what it cost to find

Recorded 16 August 2026 from two dry runs against the partially downloaded
`UD-Q1_0`, because each failure was cheap to hit and expensive to guess.

**The model path must be a `.gguf` FILE, not a directory.** `--model <dir>`
sends the server down the HuggingFace branch, which fatals on a missing
`config.json` before it ever looks for GGUFs (`model_loader.cpp:1287` takes the
GGUF branch only for `fs::is_regular_file(dir) && extension == ".gguf"`). For a
split model, point at shard 1; the reader finds its siblings by the
`-NNNNN-of-MMMMM.gguf` naming and cross-checks `split.count`.

**The container needs `--gpus all` even for `--help`.** `libcuda.so.1` is
injected by the NVIDIA Container Toolkit, so a CUDA-linked binary cannot load
without it. This is not about using the GPU; a build or a usage message fails
the same way.

**Paths on dgx.casa after the 14 August reimage:** the NAS is mounted at
`/usr/local/nas_share`, not `/mnt/nas_share`. There is no host CUDA toolkit, so
builds run inside `vllmcpp-build:gb10`, and `docker` needs `sudo`.

Working invocation:

```sh
docker run --gpus all -v /usr/local/nas_share:/nas:ro -e VT_GGUF_PREFAULT=0 \
  vllmcpp-build:gb10 ./build-srv/examples/vllm-server \
  --model /nas/checkpoints/<model>/<model>-00001-of-000NN.gguf \
  --device cpu --max-num-seqs 1 --max-model-len 512
```

`VT_GGUF_PREFAULT=0` is the load-bearing flag for a model larger than memory.
Keep-quant weights are BORROWED from the mmap rather than copied
(`p.mmap_residency = EnvOnOr("VT_GGUF_MMAP", p.keep_quant)`,
`gguf_keep_quant.cpp:208`, then the `OwnedBytes::Borrow` branch), so the tower
costs address space rather than anonymous memory. The load-time prefault would
undo that by touching every page, which is right for a model that FITS and wrong
for one that does not.

`--max-num-seqs 1` follows this spec's own regime bound. At high concurrency
every step touches most experts, so the working set stops being a working set.

**What the second dry run proved, before the checkpoint was complete.** Pointed
at shard 1 with 9 of 10 shards present, the loader merged the split, parsed the
metadata, resolved `qwen35moe` with no architecture refusal, did NOT refuse
IQ1_XXXS, and walked tensors by name to `blk.87.ffn_up_exps.weight` before
stopping on a span that exceeded the incomplete shard 9. Everything above the
byte read therefore works: architecture resolution, split merging, tensor naming
and the new encoding. What remains untested is what happens AFTER the last
tensor is read.

**One operational note that is not about this tree.** A long-lived HuggingFace
connection can decay to under 1 MiB/s while a fresh connection to the same file
runs at 20 MiB/s and the NAS writes at 97 MiB/s. Both endpoints measure healthy,
so a stalled fetch is indistinguishable from a slow one without a rate check.
Killing the curl lets the fetch script's `-C -` resume open a fresh connection.

## The 370 GiB load was MEASURED, and the page cache does not do this for free

16 August 2026, dgx.casa, 119 GiB of unified memory, `UD-Q1_0` complete and
sha256-verified, run from LOCAL NVMe so the network filesystem is not a variable.

**This section exists to record a refuted hypothesis, because it was mine and it
was load-bearing.** Reading the loader, the keep-quant path BORROWS weights from
the mmap (`p.mmap_residency = EnvOnOr("VT_GGUF_MMAP", p.keep_quant)`,
`gguf_keep_quant.cpp:208`, then the `OwnedBytes::Borrow` branch). From that it
seemed to follow that a model larger than memory would simply be demand-paged by
the kernel, and that the streaming lane was an optimisation rather than a
requirement. That is not what happens.

### What was measured

Routing is CORRECT. A probe that asks the policy directly, rather than inferring
it, reports `keep_quant = 1`, `mmap_residency = 1`, `quant_repack = 0`, and:

| Tensor | ggml | Routed to |
|---|---|---|
| `blk.0.ffn_gate_exps.weight` | 66 | `keep_quant` |
| `blk.0.ffn_down_exps.weight` | 66 | `keep_quant` |
| `blk.0.attn_qkv.weight` | 13 | `keep_quant` |
| `blk.0.ssm_out.weight` | 14 | `keep_quant` |
| `output.weight` | 12 | `keep_quant` |
| `token_embd.weight` | 12 | `expand_bf16` (2.03 G params, about 4 GiB) |

And yet the process accumulates ANONYMOUS memory linearly with load progress:

| t | anon | shared_clean | avail |
|---|---|---|---|
| 300 s | 21 GiB | 0 | 93 GiB |
| 600 s | 32 GiB | 0 | 82 GiB |
| 1360 s | 37 GiB | 0 | 77 GiB |
| 1800 s | 53 GiB | 0 | 60 GiB |

**4 GiB per 110 s, dead linear for 30 minutes, no inflection.** Extrapolated,
the load needs memory of the order of the whole 370 GiB file and cannot complete
in 119 GiB. The run was stopped at 53 GiB rather than allowed to reach the OOM
guard, because on GB10 an out-of-memory kill takes the MACHINE down and not just
the process.

`shared_clean` stayed at 0 for the entire run. No file pages were ever resident
in the process, which is what a borrow with `VT_GGUF_PREFAULT=0` should look
like, and is also why RSS alone cannot answer this question: RSS counts resident
file pages, so it conflates the two hypotheses. Anonymous versus file-backed is
the measurement that separates them.

### What this establishes, and what it does not

Established: **the model does not load in 119 GiB today, and expert streaming is
REQUIRED rather than an optimisation.** The row's original premise stands
unchanged; the "the kernel already does this" shortcut does not exist.

**Narrowed 16 August 2026, same run.** The anonymous memory is ONE merged VMA,
not many small ones: at t+300 s the process held a single 12.9 GiB anonymous
region, fully resident, alongside a 209 MiB heap and nothing else above 64 MiB.
Linux merges adjacent anonymous VMAs with identical flags, so that signature is
a run of large sequential allocations coalescing, which is what
`AllocAligned64` (`src/vt/cpu/cpu_backend.cpp:15`) produces one weight at a
time. It is NOT an arena (the CPU backend has none), and it is NOT the KV cache
(`num_blocks` auto-resolves to 256, about 4 GiB).

So the weights are being copied into CPU device memory even though
`qwen3_5.cpp:976` aliases them when `GetPlatform(d.q.device.type).is_cpu()`, and
`--device cpu` demonstrably reaches the loader (`server_main.cpp:1034` sets
`engine_params.device` before the `FromModelDir` at :1103). Those two facts still
disagree, and THAT is the remaining question: instrument the branch and find out
whether `is_cpu()` is false at that point, or whether a second copy happens
elsewhere.

**ANSWERED 16 August 2026 by instrumenting the load. The allocator is the
GatedDeltaNet V-head reorder, and it has nothing to do with the experts.**

Counters placed at each candidate, then run on both hosts, eliminated every
other explanation and found the real one:

| Candidate | Verdict |
|---|---|
| keep-quant copy branch (`OwnGgufQuantBlocks`) | **0 copies**, 281 borrows, 147.6 GiB borrowed |
| q8_0 repack branch | only accepts `kQ8_0`; this checkpoint has ONE such tensor |
| model materialisation (`qwen3_5.cpp:976`) | instrument NEVER fired during load |
| vt CPU and CUDA allocators | never reached the 2 GiB print threshold |
| `OwnedBytes` copy | shallow; a borrow shares its owner |
| **expand-to-bf16 (`DqBf16`)** | **69 calls by layer 9**, on `attn_qkv` and `ssm_out` |

`in_proj` (`attn_qkv`) and `out_proj` (`ssm_out`) become `kTransformedWeight`
whenever the V-head reorder is active (`num_v != num_k`, `qwen3_5_gguf_weights
.cpp:1020`). A transformed weight is LAYOUT-REWRITTEN at load, so it "can never
keep its blocks" and expands to bf16. For this model that is 8192x16384 plus
16384x8192 per layer, about 268 MiB each after expansion, so roughly 536 MiB per
layer across 93 layers, or **about 50 GiB** — which matches the measured 53 GiB
almost exactly.

The two hosts disagreed for a reason worth keeping: on x86 without the reorder
path taken the same load holds **anon = 0 GiB across 281 borrows**, and on GB10
it climbs. The reorder is what separates them, not CUDA, not the filesystem, and
not the expert encoding.

**The consequence changes the plan.** Expert streaming alone does NOT make this
model load. The experts already borrow correctly and cost no anonymous memory;
the blocker is ~50 GiB of DENSE weights inflating from about 5.5 bits to 16.
Streaming the experts and leaving this in place still exceeds the box. Whoever
takes this needs BOTH: the streaming lane for the ~330 GiB of experts, and a
transformed-weight path that does not expand, by doing the reorder at runtime or
by confining it inside block boundaries.

Instrument note, because it nearly cost the diagnosis: the first loader counter
printed on `(n % 300) == 1`, so it fired once and read as "1 borrow" when there
were 281. A counter that samples cannot answer "how many". The second flaw was
placing that counter AFTER the repack early-return, which would have hidden
every repack copy had there been any.

NOT established previously, now closed. The remaining unknown is not the
allocation site. The evidence bounds the OUTCOME
(linear anonymous growth proportional to load progress) without identifying
which call allocates. Routing says borrow, the process says anonymous, and those
two facts are not yet reconciled. The likely candidates, in order, are that the
borrowed `OwnedTensor` is copied again when the model materialises its own
weight buffers, or that a per-expert path re-materialises what the stacked path
would have borrowed. Whoever takes W4 should settle that FIRST, because if it is
the former then wiring `ExpertStreamer` under a loader that copies afterwards
changes nothing.

### What the load DID prove

Everything above the byte read works on the real checkpoint. The loader merged
all 10 split shards, parsed the metadata, resolved `qwen35moe` with no
architecture refusal, did NOT refuse IQ1_XXXS, and walked tensors by name deep
into the stack (`blk.87.ffn_up_exps.weight` in the incomplete-checkpoint dry
run). Architecture resolution, split merging, tensor naming and both new
encodings are not the blocker. Capacity is.

## First run: Qwen3.8-2.4T-A95B loads and generates on one DGX Spark

16 August 2026, dgx.casa, GB10, 119 GiB unified memory, `UD-Q1_0` (370 GiB, 10
shards, sha256-verified) on local NVMe, CPU device, `--max-num-seqs 1
--max-model-len 512`, `VT_GGUF_PREFAULT=0`, server entry point.

**It loads, it serves, and it is correct.**

```
prompt: "Q: What is the capital of France? A:"
output: " Paris. Q: What"
```

| Axis | Measured |
|---|---|
| load to serving | 13 min |
| resident anonymous memory | **62 GiB** of 119 GiB |
| TTFT (token 1) | **3318 s** |
| steady decode | **66.7 s/token** (66.5, 66.9, 66.8) |
| decode rate | **0.015 tok/s** |
| expert read bandwidth during decode | ~100 MB/s (measured 50 MiB/s at one sample, 914 % CPU) |

### Why a 370 GiB model fits in 119 GiB

Because the experts are never copied. They are BORROWED from the mmap and
demand-paged by the kernel, costing zero anonymous memory. What is resident is
the dense remainder, and it matches the prediction from the checkpoint's own
tensor table:

| Component | Predicted |
|---|---|
| `attn_qkv` expanded to bf16 | 21.56 GiB |
| `ssm_out` expanded to bf16 | 17.25 GiB |
| `token_embd` + F32 norms | 5.81 GiB |
| predicted total | 44.6 GiB |
| measured, with KV and runtime | **62 GiB** |

This closes the question the earlier measurement left open, and corrects the
conclusion drawn from it. The load was previously stopped at 53 GiB on the
extrapolation that it was heading for 370 GiB. It was not; it plateaus. The
extrapolation was wrong, the data was not, and the difference was letting it
finish.

### What the speed says, precisely

0.015 tok/s is not a streaming result. It is the result of having NO streaming
lane: every token needs roughly 6.7 GB of expert bytes (10 experts x 3 matrices
x 93 layers), and the kernel serves them as 4 KiB demand faults in router order.
That yields about 100 MB/s against an NVMe that sustains ~5 GB/s, so **the gap
is about 50x and it is entirely access pattern.**

The steady-state number is the useful one. 66.5, 66.9 and 66.8 seconds for three
consecutive tokens is not noise; it is the same working set being re-faulted
every step, which is exactly the behaviour a resident expert cache removes.

TTFT of 3318 s is prefill plus the cold set and should not be quoted as a decode
number.

### What this changes for the row

Streaming is NOT what makes this model fit. Borrowing already does. Streaming is
what makes it FAST, and the size of that prize is now measured rather than
assumed: 50x of pure I/O access pattern, before any cache-hit benefit. The c1
target of 3 to 6 tok/s needs roughly a 200 to 400x improvement, so it needs both
the explicit batched reads AND the hit rate the hotness cache is designed for.

Everything the lane needs is already in tree and unwired: `ExpertSlotCache`
(W1), `GgufExpertSpanOf` (W2) and `ExpertStreamer` (W3). W4 now has a measured
baseline to beat and a working end-to-end vehicle to beat it on.

## Risks/decisions

| Risk / decision | Call |
|---|---|
| Product call: is a below-gate-throughput capacity mode in scope? | YES per user direction (this spike was user-directed); it is surpass-track, default-off, and G5 protects the gate paths. Only genuine product call in here — everything behavioral follows ds4's proven design + our math |
| Per-layer sync adds ~40 waits/token | Phase 1 accepts the structural cost but does not assume margin; W0 measures c1 and W3 has its own checkpoint. Any resident/missing overlap is W6 and requires a fresh spike, not an in-place optimization promise |
| Uniform-routing h≈f may undershoot G3 | W5 measures h(f), but G3 stays fixed at f=0.5/12 tok/s. A miss is an open gap; changing the fraction is a new recorded gate, never silent rebasing |
| Expert bank = second copy of expert bytes on disk (~17 GiB) | Accepted: one-time build, keyed+versioned, evictable file; alternative (per-miss repack kernel) taxes every miss forever |
| GGUF checkpoints (APEX 35B, and every Qwen3.8-class checkpoint that exists) not covered by W1-W6 | Still out of scope for W1-W6, but the stated blocker is GONE: the slicer landed 2026-07-22. What remains is the residency policy, and the GGUF lane needs NO bank at all (see the reconciliation note) |
| Target checkpoint carries an encoding no UPSTREAM oracle defines (`UD-Q1_0`, ggml type 66) | First refused, then admitted on developer direction (15 August 2026) by pinning the fork as `llama-cpp-unsloth` at `36fe8e1cc`. The objection was policy, not feasibility, and a recorded pin answers it. The cost is honest and recorded rather than hidden: the pin is a BRANCH, which can be rebased under its own name, so the ported grid carries a digest seal, and `gateable = no` with issue #933 owing the build-and-run measurement. The oracle is scoped to the sub-IQ1_S encodings only and never outranks vLLM or upstream llama.cpp |
| IQ1_S at 1.5625 bpw may cost accuracy versus the 1.1875 bpw quant originally chosen | Accepted, and it moves the other way too: IQ1_S is the HIGHER-fidelity encoding of the two. The cost is 473 GiB rather than 370 GiB on disk, which the 3.0 TB free on dgx.casa absorbs. Quality is not asserted here; the row's gate is token output against an oracle, and any accuracy claim needs its own measurement |
| One encoding is 96.92 % of the target model | This is why W9 blocks W11 and why no partial-decode fallback is offered. A model that expands its experts to bf16 to avoid IQ1_S would need multiple TB of memory, so "unsupported encoding" here means refusal by name, never a silent widening |
| tmpfs "tier" temptation on GB10 | Rejected with reasons (§Scope verdict): tmpfs is the same unified memory; documented so it is not re-proposed |
| CUDA graphs vs data-dependent miss handling | Phase 1 explicitly disables graphs. Current Marlin has no address table; graph compatibility would require a separately spiked kernel/dispatch change after W3 profiling |
| Full-layer prefill with C<E | A single unmodified Marlin launch cannot address experts absent from slots. W4 uses exact chunk filters + scatter accumulation and proves every routed pair once; it may not allocate a hidden E-sized buffer |
| Host-copy trap on GB10 | Streaming loader must never materialize the 16.88 GiB expert `OwnedTensor` vectors. G2 inspects representation and whole-system memory, not device allocation alone |
