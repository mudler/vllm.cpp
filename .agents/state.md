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
