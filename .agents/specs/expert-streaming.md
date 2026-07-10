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
| Out (this row) | Dense weights, shared experts, router, norms, KV cache (all stay resident); GGUF expert streaming (needs per-expert slicing of 3D tensors — follow-up leaf after `QUANT-GGUF-KEEPQ-LOADER`); host-RAM tier for discrete GPUs (W7, after gate); vLLM UVA `cpu_offload_gb` mirror (own row `ENG-WEIGHT-OFFLOAD`); expert-parallel EPLB (`PAR-EP-EPLB`) |
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
top-6 bitmasks (we are top-8); GGUF 3D-tensor slicing arithmetic (our first
target is safetensors where each expert is already a separate tensor —
simpler for us).

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
| `QUANT-GGUF-KEEPQ-LOADER` (`specs/gguf-keep-quant-loader.md`) | only for the LATER GGUF-streaming leaf (3D-tensor slicing); not needed for safetensors/W1-W6 | READY, unclaimed |
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

## Risks/decisions

| Risk / decision | Call |
|---|---|
| Product call: is a below-gate-throughput capacity mode in scope? | YES per user direction (this spike was user-directed); it is surpass-track, default-off, and G5 protects the gate paths. Only genuine product call in here — everything behavioral follows ds4's proven design + our math |
| Per-layer sync adds ~40 waits/token | Phase 1 accepts the structural cost but does not assume margin; W0 measures c1 and W3 has its own checkpoint. Any resident/missing overlap is W6 and requires a fresh spike, not an in-place optimization promise |
| Uniform-routing h≈f may undershoot G3 | W5 measures h(f), but G3 stays fixed at f=0.5/12 tok/s. A miss is an open gap; changing the fraction is a new recorded gate, never silent rebasing |
| Expert bank = second copy of expert bytes on disk (~17 GiB) | Accepted: one-time build, keyed+versioned, evictable file; alternative (per-miss repack kernel) taxes every miss forever |
| GGUF checkpoints (APEX 35B) not covered by W1-W6 | Explicit out-of-scope; follow-up leaf after `QUANT-GGUF-KEEPQ-LOADER` lands (same cache, different slicer) |
| tmpfs "tier" temptation on GB10 | Rejected with reasons (§Scope verdict): tmpfs is the same unified memory; documented so it is not re-proposed |
| CUDA graphs vs data-dependent miss handling | Phase 1 explicitly disables graphs. Current Marlin has no address table; graph compatibility would require a separately spiked kernel/dispatch change after W3 profiling |
| Full-layer prefill with C<E | A single unmodified Marlin launch cannot address experts absent from slots. W4 uses exact chunk filters + scatter accumulation and proves every routed pair once; it may not allocate a hidden E-sized buffer |
| Host-copy trap on GB10 | Streaming loader must never materialize the 16.88 GiB expert `OwnedTensor` vectors. G2 inspects representation and whole-system memory, not device allocation alone |
