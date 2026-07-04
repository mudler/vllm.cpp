# State log (append; newest last)

- **2026-07-02** — Project kicked off. Repo created (`mudler/vllm.cpp`),
  pushed to GitHub. vLLM reference mapped (engine core + full feature
  inventory via subagents). Gates defined (Qwen3.6 NVFP4 parity on GB10 +
  GGUF + library + tools/grammars + streaming/non-streaming + e2e suites).
  `.agents/porting-inventory.md` (then docs/) written with T0–T3 tiers. Core design doc
  rev 2 (CUDA-first on GB10, hybrid GDN+MoE+NVFP4 in T0, MRV2, library-first).
  Upstream synced to `e24d1b24`. AGENTS.md restructured as index over
  `.agents/*.md`. NOT STARTED: any implementation. NEXT: user reviews
  inventory + design doc → writing-plans for milestone M0.
- **2026-07-02 (later)** — Project made agent-maintainable: inventory moved to
  `.agents/porting-inventory.md` as a *living parity record* (inline status
  markers + upkeep rules), `parity-ledger.md` created (append-only: every
  change → what it does vs vLLM, upstream PR refs, verification),
  `roadmap.md` created (M0–M3 broken into PR-sized units with DoD),
  `workflow.md` rewritten as the agent session protocol. NEXT unchanged:
  user review → writing-plans for M0.1.
- **2026-07-02 (later)** — Upstream sync formalized as a protocol
  (`upstream-sync.md`): repo-wide PARITY PIN (`e24d1b24`), per-file pins in
  headers, and a repeatable 7-step sync cycle (enumerate → classify
  PORT-NOW/INVENTORY/IGNORE → report under `.agents/sync/` → port → re-verify
  → advance pin). Sync tooling + recurring cycle added to roadmap.
- **2026-07-02 (later)** — M0.1 done: repo builds as `libvllm` with tests
  (doctest/ctest), optional CUDA verified on dgx.casa at sm_121 (nvcc
  13.0.88, cubin confirmed, smoke kernel ran on GB10, 2/2 tests), CI green
  on GitHub Actions (run 28625582067 for `411c072`). Vendored: doctest
  v2.5.2, nlohmann/json v3.12.0, cpp-httplib v0.49.0. CI access was
  restored by adding localai-bot as collaborator. Final whole-branch review:
  READY TO CLOSE; deferred minors triaged (BUILD_INTERFACE → M3.5;
  vllm_test_main warnings + CMake nits → opportunistic in M0.2; smoke.cu
  placeholder dies in M0.2). NEXT: M0.2 vt runtime core (Tensor/dtype/
  device, arena, op dispatch, CPU scalar ops).
- **2026-07-02 (later)** — Backend portability scoped (user): Metal (MLX vs
  native MSL exploration, ANE for encoder-class only), Vulkan, Intel XPU —
  strategy in new `backends.md`; portability confined to vLLM's own seams
  (platforms/ + attention backends + vt op tables); binding vt:: interface
  requirements added to M0.2 roadmap unit; implementations post-MVP (D10 in
  design doc; deviation §9.6 for Metal/Vulkan extensions). NVIDIA gate
  unchanged.
- **2026-07-03** — M0.2 done: vt runtime core landed (dtype/tensor/backend/
  arena/dispatch + 5 CPU scalar ref ops, all unit-tested; backends.md
  interface requirements honored: open device enum, unified-memory flag,
  per-op Queue, graph-capture hook stubs). NEXT: M0.3 parity harness
  (upstream golden dumps on dgx.casa + C++ replay). Parity-harness note:
  upstream RoPE cos/sin cache is f32 and drifts at long context (~1e-2 at
  pos 131k) — M0.3 goldens need position-scaled tolerance; rmsnorm/silu
  standard paths keep f32 precision vs upstream's pre-multiply dtype
  rounding (documented in ops.h).
- **2026-07-03 (later)** — M0.3 done (`f063890`): parity harness runs
  end-to-end — upstream dump scripts on dgx.casa → committed goldens →
  C++ replay in CI (12/12 ctest; 9 golden op cases: 4 rmsnorm + matmul +
  silu + embedding + 2 rope, all green). Oracle: pip vLLM 0.24.0 in
  `~/venvs/vllm-oracle` on dgx.casa; per-op drift checks vs pin `e24d1b24`
  clean (silu/rope byte-identical; matmul/embedding pure torch) —
  drift-check discipline documented in `tools/parity/README.md`. Note:
  M2.1's vLLM-install TODO is partially pre-staged (oracle venv exists;
  serving baseline still to be measured). NEXT: M0.4 loaders (safetensors
  + GGUF incl. NVFP4 extension types; remember the `Tensor::Numel`
  overflow guard is due here — untrusted file metadata).
- **2026-07-03 (later)** — M0.4 done (`4aeae62`, tooling `b8b4bd8`): container
  loaders landed — hf_config (upstream-loyal rotary defaults), safetensors
  mmap reader (offset-order-independent, ASan-fuzzed), GGUF v3 reader (std
  types + fork NVFP4 id 40 / Q1_0 id 41, DoS-capped) + `dump_container` CLI
  with python truth side (`tools/parity/verify_containers.py`). E2e gate on
  dgx.casa: 5/5 checkpoints byte-verified vs Python (per-tensor sha256 diff
  empty) — Qwen3-0.6B safetensors (311 tensors), unsloth 27B NVFP4 (2111),
  APEX-I-Mini GGUF (733), Coder-30B Q8_0 GGUF (579), nvidia 35B NVFP4
  safetensors 3 shards (51662+63484+9322 = 124468 tensors, index.json lists
  all 3 shards, 178s). Discoveries: the APEX GGUFs contain NO NVFP4 tensors
  (they're IQ2_S/IQ4_XS/k-quants); fork NVFP4 = GGUF type id 40 (documented
  in `.agents/gguf-nvfp4-notes.md`); the nvidia 35B snapshot was incomplete
  and had to be downloaded fresh. NEXT: finish M0.5 tokenizer (tasks 5-7:
  incremental detokenizer, corpus goldens vs HF tokenizers, records).
- **2026-07-03 (later)** — M0.5 done (`0baa46e`): tokenizer landed — unicode
  tables (exhaustive 1.1M-cp oracle-verified), pretokenize scanner, BPE core
  + tokenizer.json loader (548-string differential vs real Qwen3.6 tokenizer
  clean except recorded NFC deviation), GGUF vocab source (real-file parity
  33/33), incremental detokenizer (ported from pinned v1/engine/
  detokenizer.py, oracle exec'd verbatim, char-window stop parity), 64-entry
  corpus goldens (HF path in CI; GGUF path diff-empty on dgx.casa). 20/20
  ctest, CI green. Discoveries: the Qwen3.6 pretokenize regex is a \p{M}
  variant (\p{M} additions vs classic qwen2) — DISCOVERED and verified vs
  both gate checkpoints; classic "qwen2" fail-louds until a kQwen2Classic
  exists; the GGUF pre name is "qwen35"; NFC gap open (normalizer
  accepted-not-applied) → post-MVP hardening item; kLlama3 regex remains
  PROVISIONAL. NEXT: M0.6 CUDA baseline ops on dgx.casa (carried items:
  fn-type aliases → ops.h; embedding GPU bounds contract; widen out-dtype
  validation for bf16 outputs).
- **2026-07-03 (later)** — User scoped post-MVP goal: kernel-level drop-in
  compatibility with upstream vLLM (CUDA/AMD first) — signatures aligned so
  upstream kernels bind without rewrite. Recorded in backends.md + roadmap
  post-MVP queue; to be shaped after MVP with M0.6-M2 experience in hand.
- **2026-07-03 (later)** — M0.6 done (`3750954`): CUDA baseline ops landed —
  real CUDA backend (GB10, replaces M0.1 smoke placeholder; cudaMalloc/
  cudaMemcpyAsync-Default/streams, Synchronize/DestroyQueue virtuals),
  CUDA kernels for rmsnorm(+gemma,+fused-residual)/silu_and_mul/embedding
  (device-flag bounds check, sync-time throw)/rope_neox, cuBLASLt matmul
  (bf16/f32, f32 compute). Carried M0.2 items closed: fn-type aliases in
  ops.h with static_cast registration ties; out-dtype widened to bf16-or-f32
  (unsupported combos throw). Parity gate: test_op_parity runners factored
  onto the Backend interface — 9/9 CPU + 9/9 CUDA golden cases pass on
  dgx.casa at the SAME upstream-anchored tolerances (full ctest 21/21);
  CI (CPU-only) green. CUDA kernels carry upstream-counterpart comments per
  backends.md §drop-in. NEXT: M0.7 GDN layer (eager) — conv1d, chunked-scan
  prefill, recurrence decode; model-level goldens MUST dump from the PINNED
  checkout per tools/parity/README.md (pip oracle is fine for pure-op cases
  only); FLA chunk semantics → CUDA correctness-grade.
- **2026-07-03 (later)** — M0.7 done (`ead59d6`): GDN ops landed CPU+CUDA —
  causal conv1d fwd/update, l2norm, gated rmsnorm, gated-delta-rule
  prefill/decode. Correctness-grade scope call recorded in the ledger:
  prefill is a sequential recurrence validated against the pinned chunked
  reference (measured chunk-vs-sequential gap 2.4e-4 out / 2.3e-3 state,
  bf16); the chunked perf kernel lands M2.3. Verification: 15 pinned-oracle
  goldens, 24/24 CPU + 24/24 CUDA parity on GB10, racecheck clean; three-way
  check (kernels ↔ `.agents/gdn-semantics.md` ↔ pinned oracle) at f32 noise.
  `.agents/gdn-semantics.md` is the formula record (all math cited to pinned
  sources) and serves as M0.9's assembly reference. Deferred to M0.9: g/beta
  derivation (gdn-semantics §6 — the ops take g/beta as inputs) and
  metadata-integrity ownership (prefill/decode segmentation inputs are
  trusted; the gdn_attn.py-equivalent metadata builder owns validation).
  NEXT: M0.8 MoE layer (eager, bf16) — router top-k + shared expert;
  dequantized NVFP4→bf16 on load.
- **2026-07-03 (later)** — M0.8 done (`65788b3`): MoE layer landed CPU+CUDA —
  router top-k (f32 softmax over all experts, renormalize the k selected
  probs, lowest-index tie-break) + weighted combine, both CPU+CUDA; the MoE
  block is composed from existing Matmul + SiluAndMul ops; plus an NVFP4
  W4A16→bf16 host-side dequant utility (`DequantNvfp4ToBf16`). Verification:
  5 pinned-oracle MoE goldens, CPU 29/29 + CUDA 29/29 on GB10, racecheck +
  memcheck clean; NVFP4 dequant is bit-exact (max bf16 diff 0) on real 35B
  modelopt tensors; three-way check (kernels ↔ `.agents/moe-semantics.md` ↔
  pinned oracle). `.agents/moe-semantics.md` is the MoE formula record.
  M0.9 HANDOFF — everything M0.9 assembly needs is now ready: gdn-semantics
  §6 (g/beta prep) + moe-semantics + NVFP4 dequant (`DequantNvfp4ToBf16`,
  host-side) + hf_config + tokenizer + all vt ops (GDN, MoE); attention is
  pending M1. OPEN M0.9 DECISION (record in the M0.9 plan): MoeCombine keeps
  the routed sum in f32 with a single store-round, whereas upstream §6 does
  bf16(routed)+bf16(shared) (a double-round, ≤1 bf16 ulp) — M0.9 decides
  whether the bf16 model output must bit-match upstream. OPEN M0.9 SCOPING
  QUESTION (record in the M0.9 plan): M0.9 needs the full-attention path for
  the 1-in-4 non-GDN layers, which currently only lands in M1.6 — M0.9 may
  need a minimal correctness-grade full-attention forward (non-paged, since
  M0.9 is single-sequence) BEFORE M1, OR M0.9 scope covers just the GDN+MoE
  stack plus a simple dense attention. NEXT: M0.9 Qwen3.6 forward + registry
  (logits parity + greedy decode — the M0 exit criterion).
- **2026-07-03 (later)** — M0.9 Task 1 done (oracle de-risk): oracle = **pip
  vLLM 0.24.0** (`~/venvs/vllm-oracle`; the pinned-Python-over-pip-kernels
  overlay was BLOCKED — post-0.24.0-tag pinned Python expects newer
  vendored/compiled APIs than the pip wheel ships; every forward-math module is
  byte-identical to pin `e24d1b24`, only weight-loader plumbing + ROCm differ).
  Per-layer + full-model goldens dumped for BOTH gate checkpoints
  (`tests/parity/goldens/qwen36_*_{27b,35b}`) via `tools/parity/dump_qwen36.py`;
  recipe in `.agents/qwen36-forward-notes.md`. **KEY RE-SCOPE (plan assumption
  was wrong):** the 27B is **DENSE** (`Qwen3_5ForConditionalGeneration`, hidden
  5120, 64 layers, **compressed-tensors W4A4**, NO experts) — it does NOT
  exercise MoE. The **35B** (`Qwen3_5MoeForConditionalGeneration`, 256e/top-8,
  hidden 2048, 40 layers, **modelopt W4A16**) is the MoE gate. So M0.9
  correctness targets the **35B PRIMARILY** (its weight-only-4bit→bf16 matches
  our DequantNvfp4ToBf16 + bf16-matmul path → logit-atol + greedy bar); the
  **27B dense/W4A4 correctness is DEFERRED** to when compressed-tensors W4A4
  support lands (~M2.2), and meanwhile is a greedy-token-match-only secondary
  check. **mRoPE → kRopeNeox for text:** for text-only single-sequence all 3
  mRoPE position rows equal the token position, so mRoPE collapses exactly to
  partial NeoX RoPE on `positions[0]` (rotary_dim 64); Task 2 reuses
  `kRopeNeox`, true section-split mRoPE `[11,11,10]` is multimodal-only and
  deferred. NEXT: M0.9 Task 2 (dense causal attention op) → Task 3–5.
- **2026-07-03 (later)** — **M0 EXIT MET** (`25326fc`, with `861b1e4`
  full-model gate + `45d3c18` tolerance + `25326fc` guardrail 1.5). M0.9
  Tasks 2–5 done: dense causal attention op (Task 2), registry + qwen3_5
  model + weight mapping (Task 3), per-layer runners (Task 4:
  qwen36_{embed,gdn_layer,fullattn_layer,norm}), full-model logits + greedy
  gate (Task 5: qwen36_logits). **The assembled Qwen3.6-35B forward
  (embed→40 layers GDN/full-attn+MoE→norm→lm_head; NVFP4/FP8→bf16 weight load)
  greedy-decodes 16/16 tokens token-for-token vs the pinned oracle on the real
  35B on GB10** — independently re-verified by review. Prompt "The capital of
  France is Paris, and the" → "… capital of Germany is Berlin.\nThe capital of
  France is Paris, and the". Per-layer parity: embed/norm exact, GDN 1.3e-2,
  full-attn 1.9e-2; max top-1000 logit gap 0.994 (accepted compounding —
  doesn't flip greedy). ACCEPTED DEVIATIONS (greedy-matched): f32 residual
  stream vs upstream bf16; f32 single-round MoeCombine vs upstream bf16
  double-round. What M0 delivered (M0.1–M0.9): build skeleton + CI · vt runtime
  core · parity harness (pip-vLLM 0.24.0 oracle on dgx.casa) · container
  loaders (safetensors + GGUF) · tokenizer + detokenizer · CUDA baseline ops ·
  GDN layer · MoE layer · NVFP4 dequant · full Qwen3.6-35B forward + registry.
  The 35B (MoE, modelopt W4A16) is the M0-exit gate; 27B (dense,
  compressed-tensors W4A4) correctness DEFERRED (~M2.2, needs activation-quant
  compute; unregistered). REMAINING for the GGUF gate (#2) at model level
  (re-scoped as roadmap M0.10, T0/MVP): the APEX GGUFs (arch `qwen35moe`) are
  k-quant (Q4_K/Q5_K/Q6_K/Q8_0, IQ2_S/IQ4_XS), NOT NVFP4 — we read the GGUF
  container (M0.4, byte-verified) but have NO k-quant→bf16 dequant. Needs:
  k-quant dequant kernels matching ggml + GGUF tensor-name→param mapping for
  qwen35moe + greedy verify. Close-out asserts: `PendingRunnerOps()` empty
  (grep-confirmed, `tests/parity/test_op_parity.cpp:1035`); CI green (run
  28675136541 for the guardrail commit). NEXT (recommended): finish the GGUF
  model load (M0.10) next — it's a T0 gate and the model forward is fresh in
  hand — then begin M1.1 (engine core: Request/SamplingParams/outputs).
- **2026-07-03 (later)** — **M1.1 done** (`4d477eb` final port; ports `b888645`/`fabf48f`
  SamplingParams, `4320dae`/`a43eaf8` Request+RequestStatus, `cd13ec3` EngineCore
  I/O types, `4d477eb` RequestOutput). **M1 (the engine) has begun.** The shared
  engine vocabulary is now ported 1:1: SamplingParams (Verify/PostInit ==
  `__post_init__`, eos_token_id on the params object per upstream, NOT on
  Request), Request + RequestStatus (12-status ordering, IsFinished, the
  FinishReason map), EngineCore I/O (EngineCoreRequest/EngineCoreOutput(s),
  ModelRunnerOutput, SamplerOutput), and the public RequestOutput/CompletionOutput
  (FinishReason→string via FINISH_REASON_STRINGS). All four ports were reviewed
  PASS; CI green throughout; verification is behavioral CPU unit tests
  (validation throws, status/finish-reason maps, string mapping) — no goldens, a
  structural port. This close-out also adds `RequestOutput.prompt_logprobs`
  (opaque `std::optional<bool>` placeholder, default nullopt) for symmetry with
  CompletionOutput.logprobs (Task 4 review Important). CONTRACTS carried forward:
  (1) **PostInit contract** — SamplingParams::PostInit mirrors `__post_init__`
  and MUST be run by the constructing unit (M1.8 InputProcessor); the type does
  not self-invoke it. DEFERRED-FIELDS carryovers for later units:
  (2) **stop_reason → variant at M1.8 OutputProcessor** — currently
  `std::optional<std::string>` (T0 deviation); upstream `int|str|None` fidelity
  for the API is restored when the OutputProcessor lands.
  (3) **logprobs placeholders → sampler unit** — CompletionOutput.logprobs and
  RequestOutput.prompt_logprobs are opaque optionals (engaged/disengaged flag);
  the SampleLogprobs/PromptLogprobs payloads land with the sampler/logprobs unit.
  (4) **prompt_token_ids non-optional** — a plain vector (empty == upstream
  None/[]) until prompt-embeds arrive. NEXT: **M1.2 BlockPool + prefix caching**
  (the KV-cache foundation — hashing, refcount, LRU eviction; behavioral tests
  ported from upstream `tests/v1/core/`).
- **2026-07-03 (later)** — **M1.2 done** (`5ee2301`; ports `95be067`
  KVCacheBlock + intrusive LRU FreeKVCacheBlockQueue, `a0a8622` sha256_cbor
  block hashing, `2bba0ad` BlockPool, all re-ported/finalized at `5ee2301`).
  **The KV block store — the prefix-caching heart — is ported 1:1 from
  e24d1b24.** KVCacheBlock + FreeKVCacheBlockQueue (intrusive LRU free list),
  sha256_cbor block hashing (parent-chained, group-aware, byte-exact vs
  upstream cbor2+hashlib on a 2000-hash fuzz), BlockPool (get_new_blocks/
  cache_full_blocks/get_cached_block/touch/free_blocks LRU-split/evict/
  reset_prefix_cache) + Request.block_hashes/update_block_hashes. All reviewed
  PASS, CI green, ASan-clean; behavioral CPU tests ported from upstream
  tests/v1/core/{test_prefix_caching,test_kv_cache_utils}.py. **IMPORTANT
  LESSON:** verify the PINNED file's CURRENT api — the classic BlockPool port
  was caught DIVERGING and had to be re-ported to e24d1b24 (which carries the
  hybrid-relevant `hash_block_size`); always diff the port against the pin's
  live source, not a remembered/classic shape. **sha256_cbor-default
  DEVIATION:** we default the block-hash fn to sha256_cbor because upstream's
  default (sha256 over pickle) is not cross-language reproducible — the
  config-wiring task (M1.3+) MUST default to sha256_cbor. **DEFERRED behind
  1:1 stubs:** align path (hash_block_size≠block_size), cache_partial_block,
  evict_blocks, kv-cache events, BlockHashToBlockMap union, and
  generate_block_hash_extra_keys (mm/lora extra keys). Close-out asserts:
  `PendingRunnerOps()` empty (grep-confirmed — M1.2 added no goldens,
  `tests/parity/test_op_parity.cpp:1035` kPending = {}); CI green (run
  28679327251 for `5ee2301`). NEXT: **M1.3 KVCacheManager + hybrid
  coordinator** — the gate models need a GDN-state group alongside the
  full-attn block group; assert the literal manager-level test_evict order
  `[6,10,5,4,3,2,1]`.
- **2026-07-03 (later)** — **M1.3 done** (`75caf38`; ports `bae8f7a`
  KVCacheSpec/KVCacheConfig, `ec6f4be`+`9f30013` SingleTypeKVCacheManager,
  `5fdbb7b`+fix KVCacheCoordinator/HybridKVCacheCoordinator, `c708753`+
  `75caf38` KVCacheManager). **The hybrid KV allocator is ported 1:1 from
  e24d1b24** — the gate models' GDN recurrent-state group alongside the
  full-attn block groups. Pieces: **KVCacheSpec hierarchy + KVCacheConfig**
  (FullAttentionSpec/AttentionSpec/MambaSpec `page_size_bytes` byte-exact vs
  upstream); **SingleTypeKVCacheManager** (FullAttentionManager = left→right
  multi-block prefix; MambaManager = right→left single recurrent state,
  `[NULL..state]`, skip-all-but-tail — mode-none gate path + same-step deferral
  covered); **KVCacheCoordinator + HybridKVCacheCoordinator** (cross-group
  `find_longest_cache_hit` = fixed-point MIN/intersection, full-attn-first
  bounded shrink); **KVCacheManager** (allocate_slots line-by-line:
  accounting/watermark/OOM→nullopt/admission-cap; literal test_evict order;
  hybrid prefill). All reviewed PASS, CI green, ASan-clean; behavioral CPU tests
  ported from upstream `tests/v1/core/{test_prefix_caching,
  test_single_type_kv_cache_manager}.py` (allocate/OOM/watermark/hybrid/
  literal-evict-order). **LESSONS / DEVIATIONS:** (a) **VERIFY-CURRENT-PINNED-API
  is now a proven habit** — it caught classic-BlockPool in M1.2 and here caught
  `add_local_computed_blocks` / `get_num_common_prefix_blocks(running_request_id)`
  / `get_num_skipped_tokens`; always diff the port against the pin's live source.
  (b) **Python-floor vs C++-trunc division audit** — MambaManager
  `get_num_skipped_tokens` is the *sole* negative-operand source; spot-fixed
  (including the `block_size==1` SIZE_MAX crash), no blanket floor_div applied.
  (c) **sha256_cbor-default deviation still pending config-wiring** (carried from
  M1.2). (d) **DEFERRED behind 1:1 stubs:** sliding-window/MLA/chunked-local
  specs, mamba `align` mode, `enable_caching=false` NoPrefixCache, take_events/
  stats. Close-out asserts: `PendingRunnerOps()` empty (grep-confirmed — M1.3
  added no goldens, `tests/parity/test_op_parity.cpp:1035` kPending = {}); CI
  green. NEXT: **M1.4 Scheduler** — the unified token-budget scheduling
  (running-first then waiting, chunked prefill, FCFS preemption via
  allocate_slots→nullopt, SchedulerOutput new/cached diff); the biggest engine
  unit, ported from `vllm/v1/core/sched/scheduler.py` + `tests/v1/core/
  test_scheduler.py`.
- **2026-07-03 (later)** — **M1.5 done** (`62fdfca`; ports `f3bf0ac`
  BlockTable+MultiGroupBlockTable, `2d9f693` persistent InputBatch
  add/remove/condense, `62fdfca` update_states+prepare_inputs). **The persistent
  batch + step-input build is ported — the bridge from scheduler output to the
  forward pass.** Pieces: **BlockTable + MultiGroupBlockTable** (V1 host-array
  block-id storage, `slot_mapping = block_id*block_size + offset`, multi-group
  fanout); **persistent InputBatch** (add/remove/`condense()` field-by-field
  densification, MRV2 contract `token_ids = prefill_token_ids`, per-slot sampling
  arrays staged for M1.7); **update_states + prepare_inputs** (query_start_loc/
  seq_lens/positions/slot_mapping/logits_indices matched 1:1 vs upstream
  `_prepare_inputs`). All reviewed PASS, CI green, ASan-clean; behavioral CPU
  tests ported from `tests/v1/worker/{test_gpu_input_batch,test_gpu_model_runner}
  .py`. **ARCH DECISION (recorded in `.agents/vllm-v1-v2.md` at `2889abd`):**
  "MRV2" is TWO axes — the scheduler-output **CONTRACT** (T0, ported via the V1
  host-array algorithm from `block_table.py` + `gpu_input_batch.py` +
  `gpu_model_runner.py::_prepare_inputs`) vs the staged worker **STORAGE** (M2:
  MRV2 gpu/ staged tensors, StagedWriteTensor/UVA/fused-Triton). T0 delivers the
  contract via the V1 algorithm; staged device storage is deferred to M2.
  **KEY CARRIED DEPENDENCY — GDN-state zeroing:** SchedulerOutput omits
  `new_block_ids_to_zero`, so `update_states` cannot zero fresh GDN/mamba blocks.
  This is INERT at T0 (no batched-GDN forward yet) but MUST be wired when the
  batched-GDN forward lands (M1.6/M2) — add the field + a zero-block step, or
  have the block pool zero reused blocks — else stale recurrent state corrupts
  SSM compute. **DEFERRED:** staged device tensors, swap_states/reorder hook,
  CUDA-graph padding, LoRA/spec/mm/structured-output slot state, and the sampler
  bits (generators/allowed_token_ids/bad_words). Close-out asserts:
  `PendingRunnerOps()` empty (grep-confirmed — M1.5 added no goldens,
  `tests/parity/test_op_parity.cpp:1035` kPending = {}); CI green. NEXT: **M1.6
  Paged attention backend** — the KV-cache-aware attention (replaces M0.9's dense
  attention for the batched/cached path) + the `CommonAttentionMetadata` builder
  consuming these step inputs (query_start_loc/seq_lens/slot_mapping/block_table);
  note the GDN backend metadata (prefill/decode segmentation) for the hybrid gate
  models.
- **2026-07-03 (later)** — **M1.4 done** (`4f12158`; ports `2f0ea69`
  SchedulerConfig + FCFS RequestQueue [+ `fc81a43` un-defer fix: watermark +
  scheduler_reserve_full_isl], `c65e650` SchedulerOutput/NewRequestData/
  CachedRequestData, `a591a0d`→`f09509c` schedule() re-ported to MRV2,
  `4f12158` update_from_output + check_stop). **The Scheduler is ported — the
  schedule→execute→update loop that EngineCore drives is now in place.** Pieces:
  **SchedulerConfig + FCFS RequestQueue** (T0 field set; watermark +
  scheduler_reserve_full_isl un-deferred because T0 schedule reads them);
  **SchedulerOutput + NewRequestData/CachedRequestData** (the scheduler→runner
  diff value types); **schedule()** (unified token-budget: running-first then
  waiting, chunked prefill, FCFS preemption via allocate_slots→nullopt);
  **update_from_output + check_stop** (min_tokens→eos→stop_token_ids→length
  precedence, free-on-finish, EngineCoreOutput). All reviewed PASS, CI green;
  verification is behavioral CPU unit tests ported from upstream
  `tests/v1/core/test_scheduler.py` (FCFS / chunked-prefill / budget /
  preemption / stop-precedence / resumed-as-new). **ARCH LESSON (recorded):**
  the scheduler OUTPUT path is COUPLED to the model-runner version — schedule()'s
  output tail was RE-PORTED from MRV1 to the **MRV2 shape** (`prefill_token_ids`
  + resumed-as-new fold) at `f09509c` to match the project's MRV2 direction
  (`.agents/vllm-v1-v2.md`), so it is M1.5-ready. **DEFERRED behind 1:1 stubs:**
  priority scheduling, spec-decode scheduling hooks, structured-output grammar
  bitmask, encoder budget, KV-connector, async-scheduling, caching-OFF
  coordinator path, and EngineCoreOutputs.finished_requests frontend fan-out.
  **M1.5 WATCH-ITEMS:** (a) `new_block_ids_to_zero` for GDN/SSM state zeroing on
  fresh blocks; (b) the MRV2 InputBatch consumes `prefill_token_ids` on new reqs
  + the folded resumed-as-new reqs. Close-out asserts: `PendingRunnerOps()` empty
  (grep-confirmed — M1.4 added no goldens, `tests/parity/test_op_parity.cpp:1035`
  kPending = {}); full local ctest 40/40; CI green. NEXT: **M1.5 InputBatch/
  BlockTable (MRV2)** — the persistent batch (incremental add/diff/swap-remove
  from SchedulerOutput) + step-input build (query_start_loc/seq_lens/slot_mapping/
  positions), ported from `vllm/v1/worker/gpu/{input_batch,block_table}.py`.
- **2026-07-03 (later)** — **M1.6 done** (`370ddaf`; ports `bd47ce3`
  CommonAttentionMetadata + AttentionBackend/Impl/MetadataBuilder ABCs + flash
  NHD get_kv_cache_shape, `e231196`→`7de4f0c` vt::ReshapeAndCache stride-based
  NHD write [re-ported to index by tensor strides, not shape — the committed KV
  cache is ONE (num_blocks,2,block_size,H,D) allocation whose k/v are the rank-4
  STRIDED unbind(1) slices, block stride 2·bs·H·D; the first cut derived strides
  from shape + required whole-cache contiguity, which THROWS/corrupts on the real
  slice — mirrors pinned cache_kernels.cu::reshape_and_cache_flash reading
  key_cache.stride(0/1/2)], `c244592` vt::PagedAttention varlen causal GQA
  stride-based paged read, `370ddaf` GDNAttentionMetadata segmentation). **The
  KV-cache-aware attention that replaces M0.9's dense attention for the
  batched/cached path is in place.** Pieces: **CommonAttentionMetadata** (T0 field
  set query_start_loc(_cpu)/seq_lens(_cpu)/num_reqs/num_actual_tokens/max_query_len
  /max_seq_len/block_table_tensor/slot_mapping/causal) + the ABCs; **flash NHD
  layout** get_kv_cache_shape=(num_blocks,2,block_size,num_kv_heads,head_size)
  chosen over cpu_attn's HND — the internal cache layout is validated only via the
  attention OUTPUT (layout-agnostic), so Task2-write and Task3-read just have to
  agree, which they do (both index by NHD strides); **ReshapeAndCache**
  (stride-based write, slot→block=slot/bs,offset=slot%bs, -1 skip, CPU+CUDA);
  **PagedAttention** (per-token causal GQA softmax over paged K/V, GQA
  kv_head=h/(Hq/Hk), causal p=(seq_lens[r]-query_len)+local inclusive, CPU+CUDA)
  — **ANCHORED** to M0.9 dense vt::Attention on the single-seq case (genuine
  independent cross-check, reviewer hand-traced the causal right-alignment 3 ways)
  + composed-ref batched varlen; **GDNAttentionMetadata** (decode-first
  split_decodes_and_prefills segmentation, has_initial_state=context_lens>0 mask,
  prefill-slice rebasing prefill_query_start_loc/state_indices/has_initial_state).
  All 4 tasks reviewed PASS (Task 2 caught + fixed the stride defect via
  adversarial review before it reached Task 3); CPU ctest 47/47, warnings-as-errors
  clean. **CUDA-vs-CPU parity tests are build-guarded (this box is CPU-only) →
  dgx-pending** for ReshapeAndCache + PagedAttention (not faked).
  **GDN-STATE-ZEROING CARRY — RE-FRAMED (supersedes the M1.5 `new_block_ids_to_zero`
  framing above, which was the WRONG mechanism):** upstream does NOT pre-zero
  mamba blocks in the block pool. The GDN LAYER forward gathers the state rows and
  zeros the fresh ones keyed by the mask:
  `initial_state = ssm_state[prefill_state_indices];
   initial_state[~prefill_has_initial_state] = 0`
  (qwen_gdn_linear_attn.py:1512-1513 @ e24d1b24). Our vt::GdnPrefill/GdnDecode
  (src/vt/cpu/cpu_ops.cpp GdnPrefillKernel) read the `state` buffer
  UNCONDITIONALLY — no has_initial_state gate. So M1.6's GDNAttentionMetadata
  correctly DELIVERS `has_initial_state`/`prefill_has_initial_state`, but the
  actual zeroing is a **CALLER OBLIGATION for the batched GDN-layer assembly
  (M0.9/runner milestone)**: it MUST gather state[prefill_state_indices] then zero
  rows where prefill_has_initial_state==0 before calling vt::GdnPrefill, else a
  fresh request reads a stale mamba block → silent wrong output. Documented inline
  in include/vllm/v1/attention/backends/gdn_attn.h (prefill_has_initial_state
  doc-comment). **STILL AN OPEN CARRY** for the batched-GDN forward — the current
  M0.9 single-sequence forward zeros its own state, so this is INERT until a
  batched GDN step with mixed fresh+continuing requests runs.
  **mamba_cache_mode="align" DEFERRAL (recorded safe):** GDNAttentionMetadata sets
  state_indices = block_table column 0, which is correct for mamba_cache_mode ∈
  {"none","all"} (upstream mamba_get_block_table_tensor returns the table unchanged
  there). "align" (which gathers the last 1+num_spec blocks) is deferred. Default
  is "none" (cache.py:134); it only becomes "align" when enable_prefix_caching is
  ON *and* the hybrid model lacks supports_mamba_prefix_caching (config.py:551-565),
  which additionally requires chunked prefill — no T0 gate-model run selects it, so
  col-0 is correct for the T0 path. Wire the align col-gather when prefix caching
  over mamba layers is enabled (post-MVP). **DEFERRED (M2.4):** the FlashInfer-class
  perf paged-attention kernel + CUDA graphs (T0 is correctness-grade
  block-per-(query,head)); the CUDA reshape_and_cache/paged_attention parity
  confirmation on dgx. Close-out asserts: `PendingRunnerOps()` empty (M1.6 added no
  parity-manifest goldens — validation is in-test against the dense op + composed
  reference); CI green. NEXT: **M1.7 Sampler** — the sampling pipeline consuming
  the per-slot sampling arrays from M1.5's InputBatch + the logits from the forward
  (upstream pipeline order, seeded RNG, logprobs, GPU top-k/top-p), ported from
  `vllm/v1/sample/`.
- **2026-07-04** — **M1.7 done** (`38a8846`; ports `ff366c6` SamplingMetadata +
  LogprobsTensors + make_sampling_metadata, `f940fa3`+hardening core sampling
  ops, `aac5138` penalties/masks/builtins, `38a8846`+hardening Sampler.forward
  pipeline). **The V1 sampling pipeline that turns the forward's logits into
  sampled tokens (+ logprobs) is in place — the last engine primitive before the
  M1.8 step loop.** Pieces: **SamplingMetadata** (T0 field subset from M1.5
  InputBatch's per-slot arrays via make_sampling_metadata; logitsprocs plugin
  graph flattened to the 3 builtin fields min_tokens/logit_bias/min_p);
  **LogprobsTensors** (fleshes out the SamplerOutput logprobs payload — replaced
  the M1.1 opaque `optional<bool>`); **core ops** ApplyTemperature (temp<eps→1.0
  guard) / GreedyArgmax (lowest-index tie-break, **bit-exact vs torch.argmax**) /
  ApplyTopKTopP (sort-based pytorch path + apply_top_k_only fast path, ties-at-
  threshold kept) / ComputeProbs+ComputeLogprobs / RandomSample; **penalties**
  (rep on prompt|output union with sign-split divide/multiply, freq*count,
  pres*presence) + **masks** (bad-words n-gram suffix block, allowed-ids
  TRUE=exclude polarity) + **builtins** (min-tokens floor, logit-bias add, min-p
  prob threshold — argmax-invariance recorded: min_p=T, min_tokens/logit_bias=F);
  **Sampler.forward** (the exact ORDER: raw-logprobs snapshot BEFORE mutation →
  allowed → bad-words → non-argmax-invariant procs → penalties → sample{greedy
  snapshot; all_greedy early-return; temperature; argmax-invariant min_p;
  top_k_top_p; random; where(temp<eps) merge} → gather_logprobs with
  batched_count_greater_than ranks [`>=`, 1-based]). All 4 tasks reviewed PASS;
  each caught+fixed a real issue via adversarial review (Task 2: k>=1 guard +
  statistical CUDA-random test; Task 4: where-merge test hardened against
  predicate inversion). CPU ctest 52/52, warnings-as-errors clean.
  **RNG DEVIATION (accepted, recorded):** RandomSample ports the exponential-noise
  gumbel-max ALGORITHM (`q~Exp(1)` via splitmix64 hash of (seed,row,col) +
  inverse-CDF `-log(U)`, `argmax(probs/q)`, per-request seed override) — it is
  deterministic + distribution-correct (N=100k freq matches softmax within 3%),
  but is **NOT bit-exact vs torch's Philox4x32 + its exponential_() transform**.
  Greedy stays the bit-exact parity gate (argmax over f32 logits); random parity
  bit-exactness = **T1 carry (torch-Philox)**. This is consistent with the
  project's accepted-deviation discipline (greedy-matched deviations don't flip
  the M0-exit gate). **CUDA sampling kernels (cpu_sample/cuda_sample + the
  penalty/mask/builtin ops) are build-guarded — dgx-pending** on GB10 (this box
  is CPU-only; the CPU↔CUDA random parity test is statistical [>=98% row
  agreement] because host libm vs CUDA libdevice `log` can ULP-flip a near-tied
  argmax). **M1.8 WIRING DEPENDENCY (carry):** make_sampling_metadata emits empty
  defaults for generators/seeds, min_p/min_tokens/logit_bias, allowed_token_ids_
  mask, bad_words, and num_logprobs — the M1.5 InputBatch does not yet TRACK these
  per-slot (even though SamplingParams carries seed/logprobs/min_p/min_tokens);
  SamplingMetadata carries the fields ready to populate, so M1.8 (or an InputBatch
  extension) must wire the per-slot tracking + copy into make_sampling_metadata
  for these features to activate. Until then the sampler runs correctly on
  temperature/top-k/top-p/penalties (which ARE tracked) but the untracked
  features are inert. **DEFERRED (1:1 stubs):** logprob_token_ids (generative-
  scoring), spec-decode bonus-token (predict_bonus_token), thinking-budget,
  logprobs_mode variants beyond raw/processed; FlashInfer fused sampler = M2.4.
  Close-out asserts: `PendingRunnerOps()` empty (M1.7 added no parity-manifest
  goldens — validated in-test vs composed reference); CI green. NEXT: **M1.8
  EngineCore + step loop** — the schedule→execute→sample→update loop wiring the
  M1.4 Scheduler + M1.5 InputBatch/prepare_inputs + M1.6 paged attention + M0.9
  forward + M1.7 Sampler into the batched model runner, + the InputProcessor/
  OutputProcessor + the per-slot sampling-state tracking the above dependency
  needs, ported from `vllm/v1/engine/core.py` + `vllm/v1/worker/gpu_model_runner.py`.
- **2026-07-04** — **M1.8 done → M1 CLOSED** (`c1859d9`; ports `88821f3`
  Executor+EngineCore, `73a9509` InputProcessor, `f1ae018`+multiblock forward
  dense→paged, `9949f87`+3req batched runner, `c7ba3a5` OutputProcessor,
  `c1859d9`+abort-assert LLMEngine). **THE V1 ENGINE RUNS END-TO-END ON CPU** —
  schedule→execute→(paged forward + paged attention + batched GDN + MoE)→sample
  →detokenize→output, driven by LLMEngine.generate. Pieces: **Executor
  pass-through** (over ModelRunnerBase; collective_rpc/Ray/multiproc/Future
  collapsed to a direct call); **EngineCore.step** (has_requests→schedule→
  execute_model→sample_tokens→update_from_output; batch-queue/async/DP/grammar
  deferred); **InputProcessor** (text→EngineCoreRequest; RUNS SamplingParams
  PostInit/Verify — closes the M1.1 carry; max_tokens default + eos/stop wiring);
  **the forward dense→paged refactor** (Qwen3_5Model::Forward: full-attn now
  ReshapeAndCache+PagedAttention over the paged NHD KV cache, GDN now batched
  GDNAttentionMetadata + PERSISTENT ssm/conv-state — M0.9 dense kept as
  ForwardDense, registry unchanged so the dgx M0-exit gate is preserved);
  **GPUModelRunner** (KV alloc from KVCacheConfig, decode-first reorder +
  InputBatch::swap_states, execute_model/sample_tokens split, logits_indices
  gather, make_sampling_metadata + Sampler, sampled-token write-back);
  **OutputProcessor** (incremental detokenize + STRING-level stop + reqs_to_abort
  feedback; streaming DELTA/CUMULATIVE/FINAL_ONLY); **LLMEngine** (add_request/
  step/generate). All 6 tasks reviewed PASS; each adversarial review caught+fixed
  a real gap (Task2 verify-note, Task3 multiblock-anchor, Task4 3-req-chain
  self-consistency, Task6 abort-reaches-scheduler negative-control-verified).
  **KEY CORRECTNESS RESULTS:** paged forward == M0.9 dense **bit-exact**
  (max|diff|=0) incl. multi-block non-contiguous; decode-via-KV-cache exact; GDN
  fresh-vs-continuing (zeroing negative control fails 4.8e-2); the **four-way
  ordering identity** (attention seq_lens/block_table, logits_indices,
  SamplingMetadata row, write-back slot all agree post-decode-first-reorder);
  LLMEngine e2e greedy determinism + 2-req concurrent + streaming==non-streaming
  + max_tokens + stop-string→abort→scheduler-empty. CPU ctest 58/58,
  warnings-as-errors clean.
  **ARCH DECISION (recorded in .agents/vllm-v1-v2.md):** the MRV2 persistent-slot
  staged sampler + idx_mapping is axis-2 STORAGE (deferred to M2); M1.7's
  SamplingMetadata is the axis-1 CONTRACT via the V1 host-array algorithm, and it
  COMPOSES on ONE dense order (no idx_mapping needed) because prepare_inputs +
  attention metadata + logits gather + sampling + write-back all index the dense
  [0,num_reqs) order. The decode-first reorder IS axis-1 (applied here).
  **CARRIES CLOSED:** GDN-state zeroing (M1.6 caller obligation — now WIRED in the
  forward's GDN block: gather state[prefill_state_indices] + zero rows where
  prefill_has_initial_state==0); M1.7 seed tracking (wired into InputBatch::seeds
  → SamplingMetadata.generators). **STILL DEFERRED (marked):** MRV2 staged
  per-slot buffers + idx_mapping (M2); the InputBatch-side per-slot tracking of
  min_p/min_tokens/logit_bias/allowed_token_ids/bad_words/num_logprobs (temperature
  /top_p/top_k/penalties/seed ARE tracked — those sampler features work; the rest
  are inert until wired); prompt-logprobs; the EngineCoreOutputs.finished_requests
  DP-signalling field (safe at T0, revisit if a future OutputProcessor consumes it);
  cudagraphs/spec/LoRA/mm/PP/DP/grammar. **DGX-PENDING (milestone acceptance):**
  the real 35B greedy completion through the FULL paged loop on dgx.casa (this box
  is CPU-only) — the e2e correctness is proven on the synthetic hybrid-MoE model;
  the 35B numerical parity vs the M0-exit result via the paged batched engine is
  the outstanding gate. Also dgx-pending: ALL the CUDA kernels added across
  M1.6/M1.7 (reshape_and_cache, paged_attention, the sampling ops) — build-guarded,
  never run on GPU here. Close-out asserts: PendingRunnerOps() empty; CI green.
  NEXT: **M2 — Parity performance (gate #1)**: the throughput benchmark harness vs
  vLLM on dgx.casa (GB10) for Qwen3.6-35B-A3B-NVFP4, GPU perf kernels (FlashInfer-
  class paged attention, chunked-scan GDN, fused MoE), CUDA graphs, bf16 KV cache,
  the MRV2 staged device storage — AND the dgx bring-up of the whole M1 stack
  (run the CUDA kernels + the 35B paged greedy gate on real hardware).
- **2026-07-04 (M3.1 in progress — serving MVP)** — Started M3 (serving) since
  M1 is CPU-complete and M2 (throughput) needs dgx hardware to measure. M3.1
  plan committed (docs/superpowers/plans/2026-07-04-m3.1-openai-server.md).
  DONE: **Task 1 OpenAI protocol types** (`9b5c2c5` — CompletionRequest/
  ChatCompletionRequest + to_sampling_params, response/stream/usage/error shapes,
  nlohmann/json; upstream split protocol.py into engine/completion/chat_completion
  modules — collapsed into one mirrored protocol.{h,cpp}); **Task 2 serving
  handlers** (`9afc099` — OpenAIServingCompletion/Chat decoupled from HTTP: return
  a full Response (non-stream) or a vector of SSE `data:` chunk strings (stream)
  ending `data: [DONE]`; completion cadence + chat role-delta-first cadence;
  ChatPromptFn seam for the Task-3 template; drives the M1.8 LLMEngine; 60/60
  green). Also fixed a detokenizer streaming bug (`1313061`): GetNextOutputText
  now always trims streamed deltas to a complete UTF-8 boundary (was only guarded
  when stop_buffer_length_>0). REMAINING M3.1: Task 3 (minja-subset chat template
  for Qwen3.6, swaps into the ChatPromptFn seam), Task 4 (HTTP layer — vendored
  cpp-httplib + routes + examples/server CLI), Task 5 (records).
  **CARRY — serving UTF-8 sanitization:** our detokenizer keeps RAW bytes (not
  upstream's Python-str U+FFFD) for byte-exact OutputText; a genuinely-invalid
  multibyte run (mid-string orphan) can persist in output_text_ and make
  nlohmann::json::dump() THROW → 500 on both stream deltas and the final response.
  Real greedy Qwen produces valid UTF-8 so this is mostly a synthetic-model edge,
  but a robust server must lossy-encode to valid UTF-8 (U+FFFD for invalid runs,
  via the existing LossyStep helper) at the serving/RequestOutput boundary before
  json::dump — matching upstream's str semantics. Wire in Task 4 (or a serving
  helper). NOT blocking real-model streaming. **CARRY — dgx bring-up:** all
  M1.6/M1.7 CUDA kernels + the M1.8 batched-runner CUDA path + the 35B paged
  greedy gate remain unrun (CPU-only box); scripts/dgx-bringup.sh (`+`) runs the
  full CUDA ctest + the checkpoint-gated 35B forward gate on GB10. A "35B greedy
  through the paged LLMEngine" test still needs writing (RunQwen36Logits exercises
  ForwardDense, not the paged engine).
- **2026-07-04 (M3.1/M3.2 done — OpenAI serving)** — The OpenAI-compatible HTTP
  server runs over the M1.8 LLMEngine on CPU. `23d9f2c` (+ `9b5c2c5` protocol,
  `9afc099` serving handlers, `a99a65e` chat template, `1313061` detok UTF-8 fix,
  + engine-mutex). Pieces: **protocol types** (Completion/ChatCompletion request→
  to_sampling_params + response/stream/usage/error shapes, nlohmann/json; upstream
  split protocol.py per-endpoint — collapsed to one mirrored file); **serving
  handlers** (decoupled from HTTP: full Response non-stream / vector-of-SSE-chunks
  stream, chat role-delta-first cadence, ChatPromptFn seam); **minja-subset Jinja
  chat-template engine** (an original component like vt::; for/if/elif/else/set/
  interpolation + whitespace-control trim_blocks/lstrip_blocks + `-` markers,
  verified byte-identical to jinja2 on the Qwen3 template, loud-error on
  unsupported constructs); **cpp-httplib HTTP server** (routes + SSE via chunked
  content provider + OpenAI error shapes/status + `/v1/models` + a std::mutex
  serializing engine-touching requests since the LLMEngine is stateful/not
  thread-safe) + **examples/server CLI** (loads config/tokenizer/safetensors →
  LLMEngine, wires the real chat template). All reviewed PASS; the Task-4 review
  caught the missing engine-serialization (fixed + a 6-client concurrency test).
  Also fixed the detokenizer streamed-delta UTF-8 boundary (`1313061`) + added
  serving-boundary SanitizeUtf8 (raw-bytes detok → U+FFFD for invalid runs before
  json::dump, matching upstream str semantics — the recorded carry, CLOSED).
  CPU ctest 63/63. **DEPENDENCY DEVIATION (recorded):** cpp-httplib (MIT,
  header-only, third_party/httplib/) behind the VLLM_CPP_SERVER CMake option — a
  transport dep, not a compute/ML dep (consistent with the no-pytorch/no-ggml
  rule); the minja-subset Jinja engine is likewise original.
  **DEFERRED:** tools/tool_choice (M3.3), grammars/xgrammar (M3.4), logprobs
  payload, n>1 multiple choices, the `/metrics` endpoint, C-API + packaging
  (M3.5), GGUF model load (M0.10). **CARRY:** the example server's real-weights
  (safetensors 35B) load path is structured but unexercised on the CPU box (needs
  dgx or a downloaded checkpoint); real-model end-to-end serving is validated only
  on the synthetic model so far. NEXT (toward the serving MVP per protocol): **M3.3
  tool calling** (tools/tool_choice, Qwen/Hermes tool-parsers, streaming tool-call
  deltas, grammar-forced JSON) → **M3.4 grammars** (xgrammar core + structured-
  output manager + scheduler bitmask) → **M3.5 C-API + packaging** (include/vllm.h,
  shared lib, LocalAI-style dlopen smoke) → **M3.6 conformance suite** → **M3.7
  docs**. Also outstanding: **M0.10 GGUF model load** (k-quant dequant) and the
  **dgx bring-up** (CUDA kernels + 35B paged gate on GB10, scripts/dgx-bringup.sh).
- **2026-07-04 (M3.5 done — C API + libvllm packaging)** — The library-first
  packaging MVP gate is met. `0b252ec` (+ `d6a3f39` C-ABI core, `12ce21c`
  streaming, UAF/export review-fix). **A header-less FFI consumer (LocalAI via
  purego/cgo) can dlopen libvllm.so, dlsym the 11 vllm_* symbols, and drive
  generation** — proven by test_dlopen. Pieces: pure-C `include/vllm.h` (opaque
  vllm_engine handle, POD param structs, vllm_status, no-throw-across-boundary +
  thread-local vllm_last_error, documented ownership, VLLM_ABI_VERSION); the impl
  over the M1.8 LLMEngine (vllm_complete blocking, vllm_complete_stream +
  vllm_token_callback with early-stop, both SanitizeUtf8'd so the C strings are
  always valid UTF-8); a shared model_loader (LoadedEngine) shared by the C-API +
  examples/server; libvllm.so/.a with a linker version-script exporting ONLY the
  11 vllm_* symbols (nm-verified + a ctest [VerifyExports.cmake] that fails if any
  C++ internal leaks); examples/cli (vllm-cli, llama.cpp-style) + examples/server;
  install rules. **The adversarial review caught a real, ASan-confirmed
  heap-use-after-free** (both entry points reused a fixed request_id "0", and the
  stream path only aborted on the callback-stop branch — a mid-stream exception
  left "0" registered → the next add_request("0") freed-and-reinserted while the
  scheduler held the old Request). FIXED: unique per-call request ids (a per-handle
  atomic counter) so a leaked request can't collide + a RAII RequestGuard that
  aborts on EVERY exit path (early-stop, throwing callback, mid-stream error),
  noexcept dtor. Regression test (a throwing callback → VLLM_ERR_RUNTIME + engine
  stays reusable) is ASan-clean. Also fixed VerifyExports.cmake (CMP0057 for
  IN_LIST). Also added upstream-faithful LLMEngine::abort_request +
  OutputProcessor::abort_requests (needed for the stream early-stop teardown).
  CPU ctest 66/66; ASan/UBSan clean; C-header compiles as C11 -Werror.
  **DEPENDENCY DEVIATION:** none new (the C API is an original packaging layer,
  §9, like vt::/the minja engine). **CARRY (minor):** the no-throw guarantee has
  an OOM hole (SetError allocates inside the catch → a bad_alloc could escape);
  the C-API real-weights (safetensors/GGUF) load path is unexercised on the CPU
  box. NEXT toward the serving MVP: **M3.4 grammars** (xgrammar core + structured-
  output manager + scheduler bitmask — needed by M3.3 forced-JSON) → **M3.3 tool
  calling** (tools/tool_choice, Qwen/Hermes parsers, streaming tool-call deltas) →
  **M3.6 conformance suite** → **M3.7 docs**. Plus **M0.10 GGUF model load** and
  the **dgx bring-up** (CUDA kernels + 35B paged gate on GB10).
- **2026-07-04 (M3.4 done — grammars / structured output)** — Constrained
  decoding (JSON-schema, json_object, regex, choice, GBNF/EBNF) works end-to-end
  through the OpenAI server on CPU. `a66eef6` (+ `8343c4c` ABCs, `1351f88`
  manager/plumbing, `9b640ee` apply, `74eec60` native engine, fixes). **ARCH
  DECISION (Path C, recorded in the plan + consistent with the vllm-v1-v2
  contract-vs-storage pattern):** the parity-critical STRUCTURED-OUTPUT INTEGRATION
  is ported 1:1 (StructuredOutputManager, Scheduler::get_grammar_bitmask + the
  accept_tokens FSM advance, GrammarOutput, the runner apply_grammar_bitmask, the
  StructuredOutputBackend/StructuredOutputGrammar ABCs + the request key — what
  future PRs touch), and the GRAMMAR ENGINE is a from-scratch NATIVE backend (§9,
  ORIGINAL, like vt::/the minja chat engine) behind that ported seam. **xgrammar
  vendoring is a LATER parity-completion milestone** (a 2nd backend behind the SAME
  proven seam — upstream itself has 4 pluggable backends, so a native backend is a
  backend ADDITION, not a divergence; delivers the user's explicit "GBNF like
  llama.cpp" mandate). The bitmask INTERFACE ([num_reqs, ceil(vocab/32)] int32,
  bit set=allowed→-inf) is backend-invariant, so the scheduler/runner/sampler
  plumbing is written once and never changes when xgrammar lands. Pieces: the ABCs
  + key mapping; the manager (synchronous compile at T0, batched bitmask fill,
  -1 all-allowed inactive rows, should_advance/should_fill_bitmask = return-true
  stubs [reasoning-gating deferred]); GrammarOutput + get_grammar_bitmask + the
  accept_tokens advance; apply_grammar_bitmask in the runner (reuses the M1.7
  -inf masking op); **the native engine** (GBNF/EBNF parser + a stack-based
  push-down FSM matcher + a token-byte TRIE built once at compile → fill_bitmask
  as a single DFS over trie×FSM = sub-O(vocab), NOT per-token; byte-alignment via
  the tokenizer's inverse GPT-2 byte map; EOS allowed only at an accepting state;
  regex→GBNF + choice→GBNF); **JSON-schema→GBNF** (object required/optional
  comma-subset, types, enum/const, nested, json_object = a real JSON grammar;
  unsupported constructs THROW); **OpenAI response_format** wired through
  to_sampling_params → structured_outputs → the engine → grammar_init → constrained
  decode. **THE LOAD-BEARING INVARIANT** (fill_bitmask marks a token allowed IFF
  accept_tokens accepts it — else the sampler could emit grammar-INVALID output)
  is structurally guaranteed (shared AcceptByte + shared byte strings + exact
  pruning) AND guarded by an EXHAUSTIVE differential test (every vocab token × 8
  grammars/prefixes). Both adversarial reviews (Task 4, Task 5) PASS after fixing a
  real over-permit each (done-state EOS agreement; a required-key-not-in-properties
  that would accept schema-invalid JSON → now throws). CPU ctest 71/71.
  **DEPENDENCY DEVIATION:** the native grammar engine + JSON-schema→GBNF are
  original (§9); no external dep added (xgrammar deferred). **DEFERRED:**
  STRUCTURAL_TAG, reasoning/thinking gating, spec-decode multi-row bitmask, async
  ThreadPoolExecutor compile, key-order flexibility (fixed order = over-constrain,
  safe), xgrammar-parity (whitespace-flexibility/exotic-schema — xgrammar-only
  until vendored), guidance/outlines backends. NEXT toward the serving MVP: **M3.3
  tool calling** (now UNBLOCKED for forced-JSON — tools/tool_choice, Qwen/Hermes
  parsers, streaming tool-call deltas, grammar-forced JSON for required/named via
  M3.4) → **M3.6 conformance suite** → **M3.7 docs**. Plus **M0.10 GGUF model
  load** and the **dgx bring-up** (CUDA kernels + 35B paged gate on GB10).
- **2026-07-04 (M3.3 + M3.3b done — tool calling with RELAXED auto)** — OpenAI
  tool/function calling works over the chat server, and the KEY design point (per
  user feedback) is handled: `tool_choice=auto` is RELAXED — the model may reply
  in plain text OR call a tool. `18e3efb` (+ the M3.3 commits b315ef8/a14ce92/
  fe5034d/bdb4838/caa6fa4 + the M3.3b commits 6ba00d0/bcfe9f0/e6e497b/18e3efb).
  M3.3 pieces: tools/tool_choice protocol + ToolCall/DeltaToolCall; the Hermes/
  Qwen3 `<tool_call>{json}</tool_call>` parsers (gate model qwen35moe = Hermes
  format, verified via the tokenizer goldens) — non-streaming extract_tool_calls
  + the incremental streaming parser (re-parse-and-diff, args deltas concatenate
  to the full args); serving_chat wiring (tool_calls + finish_reason="tool_calls",
  streaming deltas); chat-template tools rendering (extended the minja engine with
  a `tojson` filter + a JSON value kind). **M3.3b — the RELAXED-auto design
  (USER-DIRECTED):** the user flagged that forcing `<tool_call>` on auto is wrong
  (the model might just reply). I cloned + analyzed BOTH llama.cpp (its lazy
  grammar / grammar-triggers / awaiting_trigger mechanism, src/llama-grammar.cpp:
  1339-1427) AND vLLM (its xgrammar **StructuralTag** — TriggeredTagsFormat for
  auto, TagsWithSeparatorFormat at_least_one/stop_after_first for required/named).
  **RECONCILED DESIGN (recorded in the plan):** vLLM's integration is
  grammar-agnostic (the lazy behavior lives INSIDE the matcher), so the 1:1 PARITY
  SEAM is vLLM's **STRUCTURAL_TAG** (a `structural_tag` param on
  StructuredOutputsParams, keyed (kStructuralTag, spec), compile_grammar(
  kStructuralTag, spec) — this UN-DEFERS the M3.4 STRUCTURAL_TAG stub), and we
  implement the **lazy matcher NATIVELY** (the llama.cpp mechanism: while awaiting,
  fill_bitmask = all-allowed [no constraint] + accept_tokens buffers free text and
  watches for the trigger word `<tool_call>`; on match, flip active + REPLAY the
  buffered bytes from the trigger offset so the JSON after `<tool_call>` is
  constrained). serving_chat maps tool_choice → the structural-tag spec: auto =
  lazy TriggeredTags (NOT forced), required = at_least_one, named = stop_after_first,
  none = nothing. The manager/scheduler/apply are UNCHANGED (grammar-agnostic).
  **Reviews caught + fixed two real bugs:** (1) the forced-JSON path emitted BARE
  JSON but the parser needs the `<tool_call>` wrapper → the forced tool call was
  dropped (fixed by forcing the wrapper in the grammar, then subsumed by the
  structural-tag approach); (2) **`<tool_call>`/`</tool_call>` are ADDED tokens (id
  248058/248059, special:false) in the REAL Qwen3.6 tokenizer** — the native
  backend excluded ALL added tokens from the trie/byte-decode, so the WORD trigger
  never fired + `</tool_call>` couldn't be consumed → auto tool calling was a
  silent no-op for the gate model. FIXED: added tokens are now grammar-matchable by
  their LITERAL CONTENT bytes (except stop/EOS), so the trigger fires + the call
  completes. The M3.3b review (PASS/PASS) confirmed auto does NOT force (triple-
  locked: fill all-allowed + is_terminated-while-awaiting + manager full-mask
  path), the replay is byte-identical to llama.cpp (mid-token/straddling verified),
  and the post-trigger fill==accept invariant holds. CPU ctest 73/73 (clean rebuild).
  **NOTE (process):** a subagent's incremental build reported green while main's
  full -Werror build was red (a header field added without a default initializer);
  now doing CLEAN rebuilds after header changes (recorded in memory). **DEFERRED:**
  Coder-XML/Mistral/pythonic tool parsers, TOKEN-id triggers in the structural-tag
  spec (only WORD triggers — added-token literal matching covers the gate model),
  parallel-tool streaming edges, the VLLM_ENFORCE_STRICT_TOOL_CALLING gating (vLLM's
  default auto sets no constraint; we always set the lazy tag when tools present —
  a stricter/more-helpful default, documented). NEXT toward the serving MVP: **M3.6
  conformance suite** + **M3.7 docs**; plus **M0.10 GGUF model load** and the **dgx
  bring-up** (CUDA kernels + 35B paged gate on GB10).
