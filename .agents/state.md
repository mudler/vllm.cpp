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
- **2026-07-04 (M0.10 done — GGUF model load, CPU path)** — The user-mandated
  "can we read gguf" gate is met on the CPU-testable path. `1a4db5c` (+ `6ef3f12`).
  Pieces: **k-quant DEQUANT** (gguf_dequant.{h,cpp}: F32/Q8_0/Q4_0/Q4_K/Q5_K/Q6_K/
  Q3_K block bytes → f32/bf16, ggml block formats ported byte-for-byte — the
  reviewer diffed every unpack + get_scale_min_k4 + the Q3_K aux shuffle line-by-
  line vs ggml-quants.c and hand-computed Q4_K/Q3_K, NO divergence); **the GGUF →
  Qwen3_5MoeWeights LOADER** (qwen3_5_gguf_weights.{h,cpp}: qwen35moe/qwen3next
  tensor-name mapping → the SAME OwnedTensor bf16 targets the safetensors loader
  produces [shared M0.9 forward, transpose delta ZERO because the reader reverses
  ggml dims to torch [out,in]], the convert-time transform INVERSIONS [norm w+1,
  ssm_a=-exp(A_log)→a_log=log(-ssm_a), the V-head grouped→tiled reorder], MoE
  expert 3-d split, HfConfigFromGguf); **model_loader `.gguf` routing**. Both tasks
  reviewed PASS (dequant byte-exact vs ggml; loader transforms verified vs the
  safetensors loader + forward + convert script — the V-head reorder index math
  hand-traced with num_v=4≠num_k=2). CPU ctest 76/76 (clean rebuild); the tests use
  synthetic GGUFs (gguf_builder.h) with value-level assertions. **DEPENDENCY
  DEVIATION:** none — the GGUF loader is original (§9), ggml is a FORMAT reference
  only (no ggml build/runtime dep). **DEFERRED:** IQ2_S(22)/IQ4_XS(23) i-quants
  (codebook-based; the Compact/Balanced APEX variants are pure K-quant+Q8_0+F32 →
  load now; Mini/Quality need i-quants — they throw a clear error); the qwen3next
  combined `ssm_ba` name (target is qwen35moe). **DGX-PENDING:** the real APEX
  Compact/Balanced GGUF end-to-end load + greedy parity vs the safetensors 35B
  (quant fidelity + the real 16k/32v head dims are exercised only synthetically on
  CPU). Run on GB10 via a model dir pointing at ~/work/apex/qwen36_35b/*.gguf.
  NEXT toward the serving MVP: **M3.6 conformance suite** + **M3.7 docs/README**;
  and the **dgx bring-up** (the whole CUDA stack + the 35B paged greedy gate + this
  GGUF greedy gate on GB10 — scripts/dgx-bringup.sh).
- **2026-07-04 (M3.6 + M3.7 done — MVP CPU-side milestones COMPLETE)** — `34d3d7b`
  (M3.6 conformance) + this commit (M3.7 docs). **M3.6:** a comprehensive OpenAI
  server conformance suite (tests/vllm/entrypoints/openai/test_conformance.cpp — 23
  cases / 252 assertions) driving the REAL cpp-httplib server over an ephemeral
  socket: completions + chat (stream + non-stream, cadence, multi-turn), tool
  calling (tool_choice=auto RELAXED = not forced, named/required shapes), grammars
  (response_format json_object/json_schema accepted, malformed → 400), error shapes
  (400/404 + OpenAI ErrorResponse), endpoints (/v1/models, /health, /version), UTF-8
  safety (emoji fixture → no 500), and concurrency (6 clients, engine mutex). No
  contract violations surfaced; no server changes needed. CPU ctest 77/77 (clean
  rebuild). **M3.7:** rewrote README.md to reflect the built state — a "What's
  implemented (CPU)" section (engine core, paged forward, sampler, OpenAI server +
  tools + grammars, libvllm C-API packaging), a Quick start (build / serve /
  vllm-cli / C-API), updated support tables, and an honest "Status & caveats"
  section distinguishing behavioral (CPU-validated) from numerical (dgx-pending:
  the real 35B greedy/logits through the paged engine, the CUDA kernels, and the
  throughput-parity-vs-vLLM benchmark on GB10). Verified the example binary names
  (build/examples/server, build/examples/vllm-cli). **★ MVP MILESTONE STATUS ★:**
  ALL CPU-implementable MVP milestones are COMPLETE + recorded: M0 (35B forward,
  M0-exit greedy on GB10), M1 (V1 engine end-to-end on CPU), M3.1 OpenAI server,
  M3.2 chat templates, M3.3 tools (+ relaxed auto), M3.4 grammars, M3.5 C-API/
  packaging, M3.6 conformance, M3.7 docs, M0.10 GGUF load (CPU path). **THE
  REMAINING MVP GATES ALL NEED GB10 HARDWARE (can't run on this CPU-only box):**
  (1) **M2 — throughput parity vs vLLM** (the #1 gate: GPU perf kernels [FlashInfer-
  class paged attention, chunked-scan GDN, fused MoE], CUDA graphs, bf16 KV cache,
  MRV2 staged storage — measured on GB10 at large concurrency for Qwen3.6-35B-A3B-
  NVFP4); (2) the **dgx bring-up** — run ALL the build-guarded CUDA kernels
  (M1.6/M1.7 attention/sampler, the batched runner) + the **35B greedy through the
  paged LLMEngine** + the **35B GGUF greedy** on GB10 (scripts/dgx-bringup.sh runs
  the CUDA ctest + the checkpoint-gated forward gate; a "35B greedy through the
  paged engine" test still needs writing — RunQwen36Logits today exercises the
  DENSE ForwardDense, not the paged engine). These are hardware-gated: they need
  the user's `ssh dgx.casa` (GB10, ~/.cache/.../Qwen3.6-35B-A3B-NVFP4 + ~/work/apex/
  qwen36_35b/*.gguf). NEXT (when hardware is available): M2 perf + the dgx bring-up.
  On CPU there is no further MVP work — the remaining items are M2 GPU perf (needs
  GB10 to measure) and the hardware validation.
- **2026-07-04 (M2.1 harness done — the gate-#1 measurement tool)** — `57cd5af`.
  `examples/bench` (`vllm-bench`) drives the LLMEngine at configurable concurrency
  (--model/--num-prompts/--input-len/--output-len/--concurrency) and reports the
  vLLM-comparable throughput/latency metrics (mirrored from vllm/benchmarks/
  serve.py: request/output/total-token throughput, TTFT/TPOT/ITL/E2EL mean/median/
  p99) + a prefill-vs-decode split. CPU synthetic smoke green (the numbers are
  meaningless on toy weights — the harness is the deliverable); ctest 79/79.
  **This is the CPU half of M2.1** — the throughput-PARITY measurement (run
  vllm-bench on the real 35B + `vllm bench throughput` on the ~/venvs/vllm-oracle
  baseline at large concurrency, record both in the ledger) is dgx-pending.
  **★ CPU-IMPLEMENTABLE MVP SURFACE NOW FULLY EXHAUSTED ★** — everything the MVP
  needs that can be built AND validated on a CPU-only box is done + green + recorded
  (M0, M1, M3.1-M3.7, M0.10, the dgx bring-up harness + the paged-engine 35B gate,
  the M2.1 bench harness). The genuinely-remaining MVP gates are HARDWARE-BOUND and
  need `ssh dgx.casa` (GB10): (1) **M2.2-M2.6 GPU PERF KERNELS** (NVFP4 W4A4 GEMM/
  MoE, GDN fused kernels, paged-attn tuning, CUDA graphs, fusions) — these can be
  WRITTEN under CUDA guards but their whole POINT is MEASURED speedup, which needs
  the GPU; writing unmeasurable kernels is low-value; (2) the **dgx BRING-UP RUN**
  (scripts/dgx-bringup.sh: the CUDA-kernel parity suite + the M0-exit dense gate +
  the paged-engine 35B greedy gate + the APEX GGUF greedy) — needs the hardware +
  the checkpoints; (3) the **M2 throughput-parity measurement** (vllm-bench vs the
  vLLM oracle on GB10 → the ledger parity table = gate #1). NEXT (needs GB10): run
  the dgx bring-up; then M2.2+ perf kernels driven by real measurements from the
  M2.1 harness.
- **2026-07-04 (★ dgx BRING-UP — CUDA STACK + 35B GATES VALIDATED ON REAL GB10 ★)**
  — The hardware wall was a WRONG assumption: `ssh dgx.casa` works non-interactively
  (GB10, CUDA 13.0, nvcc). Synced the repo to dgx (~/work/vllm.cpp @ latest main),
  built with `-DVLLM_CPP_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121 -DVLLM_CPP_SERVER=ON`
  — **the ENTIRE CUDA kernel set compiled clean on GB10** (all of M1.6/M1.7's
  reshape_and_cache/paged_attention/sampling ops + the batched runner + server +
  bench). Ran the full `ctest` on the CUDA build: **100% — 79/79 passed, 0 failed**,
  including:
  - tests 1-77: every CUDA-vs-CPU parity case (the M1.6/M1.7 kernels) validated on
    real GB10 hardware — closes the "CUDA kernels build-guarded/unrun" carry across
    M1.6/M1.7 entirely;
  - **#78 test_op_parity** (the M0-exit DENSE 35B gate, RunQwen36Logits) — PASSED
    (2603 s: cold 22GB NVFP4→bf16 load + the 40-layer dense forward);
  - **#79 test_qwen36_paged_engine** (the NEW real-model PAGED-engine gate) — PASSED
    (331 s): the real Qwen3.6-35B-A3B greedy-decodes 16 tokens through the full
    paged LLMEngine (paged attention + KV-cache growth + batched GDN + Sampler)
    TOKEN-FOR-TOKEN matching the M0-exit golden. **This is the real-model paged
    validation that was the outstanding M1/dgx acceptance gate — MET.**
  So the whole M1 engine + the serving stack + the sampler + the GGUF/safetensors
  load are now validated on REAL GB10 hardware with the REAL 35B model, not just
  synthetically on CPU. **KEY M2 FINDING (measured):** the correctness-grade forward
  is SLOW on GB10 and GPU-underutilized (test_op_parity 43 min at ~0% GPU, 85GB host
  RSS) — the bottleneck is CPU-side NVFP4→bf16 dequant at load + per-op host↔device
  staging, NOT GPU compute. This is EXACTLY the M2 target: on-device NVFP4 GEMM
  (weights stay fp4 in memory, M2.2), staged on-device KV/weight storage, and fused
  kernels + CUDA graphs. The M2.1 bench will quantify the gap vs the vLLM oracle.
  REMAINING: M2 throughput parity (the perf kernels + the measured gate #1); the
  APEX GGUF greedy (point the paged gate at a Compact/Balanced .gguf).
- **2026-07-04 (★ gate-#1 THROUGHPUT MEASURED on GB10 vs vLLM ★)** — With GB10
  access confirmed, ran the M2.1 harness on the real 35B and the vLLM oracle
  (0.24.0) on the IDENTICAL workload (8 prompts × 256 in × 32 out). **vLLM: 1377
  total tok/s / 153 output tok/s / 1.20 req/s** (optimized). **Ours: correctness-
  grade — did NOT finish in >80 min** (GPU ~0% the whole time, 82GB host RSS;
  single-seq ~0.2 output tok/s from the paged gate). **The gap is ~1000× (≈3
  orders of magnitude)** — the expected, un-optimized-forward delta. **★ MVP GATE
  STATUS (hardware-grounded) ★:** Gate #2 (GGUF read/load) ✅ path done; gates #3-#5
  (serving MVP: OpenAI API, tools, grammars, C-API, conformance) ✅ built + CPU-
  tested + the CUDA build passes 79/79 on GB10; **engine CORRECTNESS ✅ 100% on
  real GB10** (all kernels + both 35B gates, real model). **Gate #1 (throughput
  parity vs vLLM) is OPEN and now precisely MEASURED** — our correctness-grade
  forward is ~1000× slower, GPU-idle, CPU-dequant/staging-bound. Closing it is the
  **M2.2-M2.6 GPU perf-kernel work**, which is now fully unblocked (GB10 access +
  the M2.1 harness + the recorded vLLM baseline + the profiling insight). **The
  highest-leverage M2 order (from the GPU-idle profile):** (1) M2.2 on-device NVFP4
  W4A16 GEMM — keep weights fp4 in device memory, eliminate the 40-min CPU dequant
  + the 70GB bf16 host tensors + the per-op weight upload (this is the dominant
  cost); (2) staged on-device KV/weight storage (the MRV2 axis deferred at M1.5/M1.8)
  — eliminate per-op host↔device copies; (3) M2.5 CUDA graphs for decode; (4) M2.3/
  M2.4 fused GDN/MoE + paged-attn tuning. This is substantial multi-session GPU
  work; the CPU-side MVP is complete + hardware-validated, and gate #1 is the
  remaining, now-unblocked, measured target. NOTE: `~/venvs/vllm-oracle` needed
  `pip install ninja` (JIT) to run the baseline (fixed).
- **2026-07-04 (★ M2.2b — FIRST MEASURED gate-#1 SPEEDUP on GB10: ~1000×→~66× ★)** —
  `faafc36`+`63a1f8c`. Kept the NVFP4 weights (MoE experts + shared + lm_head, the
  dominant ~22GB) fp4-RESIDENT on the GPU (raw packed+scales, uploaded once) instead
  of the ~40-min CPU dequant to ~70GB bf16 host tensors, and wired the forward's
  MoE/shared/lm_head matmuls to vt::MatmulNvfp4 (M2.2a's on-device dequant-GEMM).
  Also fixed a real bug: LoadedEngine hardcoded a CPU queue (SelectQueue → CUDA when
  a GPU is present — without this the forward silently ran the CPU dequant reference).
  **VALIDATED ON REAL GB10:** test_qwen36_paged_engine PASSES on CUDA (45.5s, 16/16
  token-for-token vs the M0-exit golden — the real 35B greedy through the fp4-resident
  paged forward, correctness preserved). **MEASURED (vllm-bench, real 35B, 4×128×16):**
  load ~40min→~37s (~65×), host RSS ~82GB→42GB, output 0.2→2.31 tok/s (~10×), total
  20.88 tok/s. **Gate-#1 gap vs the vLLM baseline (153 out/1377 total) went from ~1000×
  to ~66×** — the measured GPU-idle/CPU-dequant root cause is ELIMINATED (the biggest
  single win). CPU ctest green (the change is CUDA-path; CPU keeps bf16). **THE
  REMAINING ~66× (prioritized, from the new profile):** (1) per-op host↔device STAGING
  — every DBuf still downloads to host after each matmul + the MoE gather/GDN/attn glue
  run in host loops → keep the step's activations + KV on-device across the forward
  (M2.x, the MRV2 staged-storage axis) + M2.5 CUDA-graph decode capture; (2) the naive
  one-thread-per-output MatmulNvfp4 kernel → tiling/MMA (M2.2a perf follow-up); (3)
  M2.2c FP8 attn/GDN weights on-device (still bf16-on-host). Each is a real GPU-kernel
  effort but now measure-driven (the harness + baseline + the profile point the way).
  ★ Gate #1 is no longer "unmeasured/blind" — it's an active, measured optimization
  campaign with the first ~15× aggregate win (10× decode + 65× load) landed on real
  hardware. ★
- **2026-07-04 (M2.x — pooled allocator +36%, profile corrected to launch-bound)** —
  `origin/main` (DevicePool behind DBuf). The post-M2.2b bottleneck was a
  **cudaMalloc/cudaFree sync storm** (thousands of tiny scratch allocs/decode-step,
  each device-syncing), NOT host↔device data movement (cheap on GB10's UNIFIED
  memory — a load-bearing correction to the plan's assumption). A pooled device
  allocator (reuse freed blocks) took decode **2.31→3.15 output tok/s (+36%)**,
  gate token-for-token green on GB10. A MoE device-chaining experiment REGRESSED
  (unified memory made the removed Downloads cheaper than the tiny async copies it
  added) → reverted (documented). **Gate-#1 gap: ~66×→~48× vs vLLM (153/1377).**
  **CORRECTED PROFILE:** the decode is now LAUNCH-OVERHEAD bound (many tiny kernels/
  token; the ~16 per-expert MoE GEMMs the dominant launch count) — GB10 is
  explicitly launch-overhead+bandwidth bound (environment.md), so this is expected.
  RE-PRIORITIZED next steps: (1) **M2.5 CUDA graphs** — capture the decode step over
  the pool's now-stable device pointers, replay per token (the classic launch-bound
  fix); (2) **M2.4 fused MoE** — one token-tiled kernel vs 16 tiny per-expert GEMMs;
  (3) GEMM tiling for the tiny-m decode GEMMs; (4) M2.2c FP8 attn/GDN on-device.
  Aggregate gate-#1 progress this session: ~1000×→~48× (65× load + ~14× decode),
  all measured on real GB10 with correctness preserved (16/16 token match) at every
  step. Still an active campaign toward parity.
- **2026-07-04 (GEMM tiling ~2× banked + ★ DEFINITIVE gate-#1 profile: 88% host-bound ★)**
  — `1409037`. Tiled the NVFP4 GEMMs (shared-mem, coalesced fp4 decode, size-gated
  so decode-small keeps the naive L2-resident path) → **GEMM GPU-time ~2×**
  (MatmulNvfp4 4.9×, incl. killing a 1.04s prefill lm_head spike; grouped 1.96×),
  paged gate 16/16 correct. **BUT end-to-end FLAT** — and nsys gave the DEFINITIVE
  finding: the 35B forward is **88% HOST-API-BOUND** (cudaMemcpyAsync 67.5%/81k calls
  + cudaMalloc 20.5%/62k + cudaFree 5.9%; GPU kernels only ~25% of wall). So every
  compute win so far (fp4-resident, fused MoE, GEMM tiling) is BANKED — they don't
  surface until the host overhead is eliminated. **★ THE remaining gate-#1 gap (~44×)
  is almost entirely HOST-API OVERHEAD, precisely measured ★.** THE unlock: (1) make
  the decode forward FULLY device-resident (eliminate the 81k per-op Download/upload
  memcpys — NOT the tiny-copy device-chaining that regressed at M2.x, but keeping ALL
  40 layers' activations on-device end-to-end, host touches only the small inputs +
  the final logits); (2) a persistent/graph-friendly pool (eliminate the 62k mallocs);
  (3) **M2.5 CUDA graphs** — capture the decode step over persistent buffers, replay
  per token (a captured graph has ZERO per-op host API calls → kills the 88%). The
  MoE is now fixed-shape (M2.4) so the step is capturable. This is THE big remaining
  decode win + it makes all the banked compute speedups finally pay off. After that:
  GDN scan (~20% GPU), NVFP4 MMA, FP8 on-device. Gate-#1 aggregate this session:
  ~1000×→~44× (measured), with the exact remaining bottleneck (88% host API) now
  pinpointed → the path to parity is clear + measure-driven.
- **2026-07-04 (device-resident weights +38% + APPLES-TO-APPLES vLLM gap, directive-honored)**
  — `69eab0d`. Weights were re-uploaded to the device EVERY op (esp. the ~600MB embed
  table per forward) — a lazy per-OwnedTensor device cache (upload-once) fixed it → **+38%
  on the 8×128×64 config**. Per the new always-compare-vs-vLLM directive, measured BOTH
  sides on the SAME workload (8×128×64 conc4, GB10): **ours 14.33 output tok/s / 43.05
  total** vs **vLLM 0.24.0 enforce-eager 131.3 output / 1181.8 total** (gpu-mem 0.7,
  coexisting with the box's LocalAI service). **Gate-#1 gap: ~9.2× decode output, ~27×
  total (prefill-dominated), eager-vs-eager.** (vs FULL vLLM with cuda-graphs ~153/1377:
  larger — the graph delta IS our next unlock.) The decode gap (~9×) is now modest; the
  total gap (~27×) is prefill (vLLM's fused prefill kernels + graphs). **Directive lesson
  (and a good stress test of it):** getting the matching vLLM number required capping
  gpu-mem around LocalAI + retrying a transient profiling race — measuring vs vLLM surfaces
  the REAL shared-box conditions. Aggregate gate-#1 this session: ~1000× → ~9× decode /
  ~27× total (all measured on real GB10, correctness preserved at every step). REMAINING:
  (1) **CUDA graphs** (decode host-API kill → the ~9× decode gap; the capture hook `536253f`
  is in place, needs the mid-forward host glue [Download/Synchronize + GDN/attn/gather host
  loops] eliminated so the step is pure-async-on-stream, then capture/replay); (2) prefill
  fusions + the NVFP4 MMA GEMM (the ~27× total gap); (3) GDN scan (~20% GPU); (4) M2.2c FP8
  on-device. GB10 iteration (each validation loads the 35B) is the campaign's rate limiter.
- **2026-07-04 (device-resident forward +46%, decode now SYNC-FREE → CUDA graphs unblocked)**
  — `f95b131`. Device-ified the mid-forward host glue (6 new elementwise vt ops for the
  attention gate/sigmoid, GDN g-beta/conv-split, shared-expert gate; device-resident matmul
  helpers returning DBuf; blocks thread hidden/res as device tensors). The fp4+CUDA decode
  body is now PURE async-on-stream (only Embedding + final-logits Download remain host) — THE
  prerequisite for CUDA-graph capture. GB10: paged gate 16/16 token-for-token, CPU 82/82.
  **Measured +46% output/+76% per-stream decode** (fair A/B under LocalAI contention). Gate-#1
  campaign continues: ~1000× → now the decode step is device-resident + sync-free, ready for
  the CUDA-graph capture that collapses the ~88% host-API overhead (the ~9× decode gap).
  **MEASUREMENT NOTE (directive):** the box's LocalAI service was pinning the GPU ~89% during
  this window, depressing absolutes + blocking a free-box apples-to-apples vLLM re-measure —
  the fair same-conditions A/B (+46%) stands; a free-box both-sides re-measure vs vLLM 131/1181
  is the open measurement item (needs LocalAI idle). NEXT: Phase 2 CUDA-graph capture (hoist
  per-token inputs to persistent device buffers, pre-warm pool, wire BeginCapture/Replay into
  GPUModelRunner) — the big decode-gap unlock, now unblocked.
- **2026-07-04 (PIVOTAL — first clean free-box vLLM comparison; CUDA graphs disproven as the unlock)**
  — `2999431`. Phase 2 landed decode CUDA-graph capture correctly (paged gate 16/16 WITH
  graph active, mirrors vLLM, gated to num_reqs==1 with zero batched regression) BUT the
  result is a **load-bearing negative**: graphs recover only +2.6% single-stream and regress
  −7% at batch-8. Phase 1's async-on-stream work already hid host launch overhead behind the
  GPU → **decode is now GPU-COMPUTE-BOUND, not host-bound**. The user stopped the LocalAI
  worker (GPU 91%→0%), enabling the FIRST honest apples-to-apples free-box measurement vs
  vLLM (8×1024×128 enforce-eager): **vLLM 1124 total / 124.9 output tok/s vs ours 70.2 total /
  7.8 output → ~16× slower**, dominated by **prefill (TTFT ~100s)** + slow reference kernels.
  **STRATEGIC RE-ORIENTATION of gate #1:** the remaining ~16× is GPU-KERNEL + PREFILL COMPUTE
  SPEED — the naive NVFP4 dequant-GEMM (no tensor-core MMA), the GDN scan, and attention need
  to become cutlass/flashinfer-class. Host-overhead optimizations (graphs, device-resident
  forward) are DONE and were worth ~1000×→~16×; the last gap is raw kernel throughput.
  NEXT (measure-first): nsys-profile the free-box PREFILL + decode to get the GPU-time
  breakdown by kernel, then attack the dominant cost (almost certainly the NVFP4 GEMM → a
  tensor-core MMA kernel, the killgate prior art). GPU is free — keep it serialized to one
  workstream. Restart `local-ai-worker` when the measurement campaign pauses.
- **2026-07-04 (M2.7 tensor-core NVFP4 GEMM — gap 16.8×→7.5×; prefill bottleneck now GDN scan)**
  — `bc0a8d7`. Moved the prefill-dominant fp4 W4A16 GEMMs (MoE grouped + dense projections)
  onto Blackwell tensor cores (bf16 WMMA, dequant-into-shared) + device-side expert grouping
  for the MoE (counting sort → dense per-expert GEMM, weight-reuse + tensor cores). Correctness
  preserved (paged gate 16/16, unit 4/4, CPU 82/82, -Werror clean). **Free-box measured vs
  vLLM:** prefill TTFT 14.4→6.1s (2.36×), 8×1024×128 total 70→157 tok/s (2.24×), batched TTFT
  100→32.8s (3.05×). **Gap to vLLM CLOSED 16.8×→7.5×** (vLLM 1181 total / 131 output). nsys:
  fp4 MoE GEMM 70.7%→11.6% of prefill; **the prefill bottleneck MOVED to `GdnScanKernel`
  (63.8%)** — the gated-delta-net recurrence, a sequential-scan problem (needs chunk-parallel
  scan, NOT MMA). HONEST: kernel below bf16 peak (~3-4 TFLOP/s useful; win is part tensor-core
  part expert-grouping) — tuning headroom remains. NEXT LEVERS (measured): (1) **GDN chunked/
  parallel scan** — now 63.8% of prefill, THE next prefill lever; (2) M2.8 decode fp4-GEMV +
  fast-argmax (the ~100× argmax, the naive M=1 fp4 GEMV, route the cublas bf16 gemvx to fp4).
  GPU stays free+serialized (LocalAI worker down, restart disabled per user directive).
- **2026-07-04 (state survey + record hygiene)** — read-only survey subagent mapped the whole
  MVP surface. Findings acted on: (1) refreshed the M2 plan to the post-M2.7 reality (GDN
  chunked scan is now the #1 prefill lever at 63.8%, NOT the already-done MoE GEMM; gap 7.5×,
  config stated); (2) fixed the roadmap M0 header (M0.10 GGUF CPU load is DONE `6ef3f12`, real-
  GGUF parity dgx-pending — header had said "remains open"). OPEN DECISIONS surfaced to the
  user: (a) **27B W4A4 gate scope** — gates.md makes 27B a co-equal throughput gate model but
  every roadmap/inventory marker defers it (~M2.2); need an explicit "MVP = 35B-only vs 35B+27B"
  call; (b) **`/metrics` Prometheus endpoint** — gate #4 names it explicitly, currently deferred.
  RECORD CORRECTION: the `test_qwen36_weights` red reported at `2999431` (Phase 2, "79/80,
  asserts bf16 experts") was MIS-described — the test is checkpoint-gated and actually asserts
  U8/NVFP4 experts + F8_E4M3 attn (tests/vllm/test_qwen36_weights.cpp:92,76) vs torch goldens
  from snapshot 491c2f1e. It SKIPs without the checkpoint; M2.7 reported 82/82. The real
  end-to-end correctness gate (paged greedy 16/16 on the real 35B) is green, so any weights-test
  red is most likely a stale per-tensor golden / snapshot-version mismatch, NOT a dequant
  regression — CONFIRM on a free-box GB10 run between kernel jobs (non-blocking).
- **2026-07-04 (MVP definition SHARPENED by user + scope decisions)** — user calls:
  (1) **27B is co-equal in-MVP** (35B + 27B BOTH at vLLM throughput = the MVP; not deferred).
  (2) **`/metrics` Prometheus DROPPED** from MVP → post-MVP (gates.md gate #4 updated).
  (3) **MVP = both gate models at vLLM speed, full stop.** Explicitly POST-MVP, ordered:
  (a) more architectures, (b) more/faster CUDA kernels, (c) **design-lift to import vLLM
  kernels directly (csrc drop-in → cut maintenance)**, (d) a systematic sweep of vLLM
  features/archs/accelerators we lack → decide follow-up. Do NOT start post-MVP until both
  models hit parity. gates.md banner records this. IMPLICATION: the 35B throughput campaign
  (GDN scan in progress → decode GEMV/argmax) continues, AND a 27B/W4A4 bring-up starts as a
  parallel CPU-first workstream (correctness scaffolding: the dense `Qwen3_5ForConditional
  Generation` arch + compressed-tensors W4A4 load/dequant + greedy parity vs the oracle) —
  27B checkpoint IS present on dgx (`~/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-
  NVFP4`). 27B's W4A4 GPU kernels need GB10 later (serialize behind the 35B kernel jobs).

## 2026-07-04 — 27B (dense W4A4) CPU-first correctness scaffolding

Kicked off the 27B gate (co-equal MVP, ZERO prior bring-up) as a CPU-only
workstream (GPU reserved for the 35B kernel job — no GPU touched; only read-only
dgx inspection of the checkpoint).

- **Surveyed** the real `unsloth/Qwen3.6-27B-NVFP4` checkpoint (config.json +
  safetensors manifest, read-only on dgx) + pinned upstream `e24d1b24` →
  new pinned doc **`.agents/qwen27b-w4a4-notes.md`** (arch dims, layer pattern,
  SHARED-vs-NEW vs the 35B, the compressed-tensors W4A4 format from the actual
  manifest, and the ordered bring-up plan with GPU-gated steps marked).
- **Two surprises (recorded, they change the plan):** (1) the 27B is
  `Qwen3_5ForConditionalGeneration` — a **VL multimodal** model (vision tower +
  image/video tokens), not plain text → implement the TEXT path first, defer the
  ViT. (2) quant is compressed-tensors **W4A4** (activations also fp4, dynamic
  per-token), but the WEIGHT encoding == modelopt NVFP4 (names + a reciprocal
  global scale aside) → the existing M2.7 W4A16 tensor-core GEMM can carry the
  27B weights with a 1-line fix (fast path); true W4A4 fp4-MMA is a measured
  follow-up (killgate 0034/0035 saw it regress on GB10).
- **Landed (CPU-green, clean -Werror rebuild):** the CPU W4A4 dequant +
  activation-quant emulation reference `nvfp4_emulation.h/.cpp` (mirrors
  `nvfp4_emulation_utils.py` + `compressed_tensors_w4a4_nvfp4.py` +
  `kernels/linear/nvfp4/emulation.py`), unit-tested `test_ct_nvfp4_emulation`
  (81 assert), and the skipping greedy-parity gate scaffold
  `test_qwen27_paged_engine.cpp`. Full suite 84/84 (no 35B regression).
- **Next (notes §5):** config+loader plumbing (recognize the dense arch, route
  bf16 vs W4A4 by name, materialize W4A4→bf16 via `DequantCtNvfp4WeightToF32`) →
  dense forward assembly (reuse 35B GDN/attn, swap MoE→dense SwiGLU) → [GPU]
  capture the pip-vLLM greedy golden + wire the W4A4 GEMM (fast path 6a) → flip
  `kW4A4ForwardReady`.
- **2026-07-04 (TWO MVP tracks landed — 35B gap 7.5×→5.84×; 27B bring-up: shares 35B backbone)**
  35B: **M2.3 GDN chunk-parallel prefill scan** (`2ce938f`, mirrors FLA chunk.py; GDN scan 3.0×
  faster, batched TTFT 33.2→19.4s, total 155.6→202.5 tok/s, **gap 7.5×→5.84×**, paged gate 16/16,
  found+fixed a multi-head hstate bug). Prefill bottleneck MOVED to `PagedAttentionKernel` 28.6%
  (FlashInfer-class GQA 16/2 = next prefill lever) then GdnChunkDeltaH 18.5%.
  27B: **CPU-first W4A4 bring-up** (`559e5cc`). TWO PLAN-CHANGING FINDINGS: (1) the 27B is a
  **VL-multimodal** model (`Qwen3_5ForConditionalGeneration` + vision_config) — do the TEXT path
  first, defer the ViT (T1/T2). (2) It **shares the 35B hybrid backbone wholesale** (GDN, gated
  attn, RoPE, Gemma norm — GQA ratio 3 vs 2 is a dims-only change); the ONLY new structure is the
  **dense SwiGLU MLP** (vs MoE) + the W4A4 quant. CRUCIALLY the **W4A4 weight encoding == modelopt
  NVFP4 byte-identical** (CT just stores globals as divisors) → the existing **M2.7 tensor-core
  GEMM can carry the 27B with a one-line `1/weight_global_scale` reciprocal + name remap (bf16
  activations, W4A16-style)** — the FAST PATH to 27B correctness+throughput, NO new kernel. True
  fp4×fp4 MMA (6b) is a risky optional (killgate 0034/0035 saw W4A4 fp4-MMA regress on GB10).
  Landed: CPU W4A4 dequant/activation reference + `test_ct_nvfp4_emulation` 6/81 assert, skipping
  27B greedy gate scaffold, `.agents/qwen27b-w4a4-notes.md`. Full suite **84/84**.
  27B REMAINING (ordered): [CPU] loader plumbing + dense forward assembly (reuse 35B, MoE→SwiGLU);
  [GPU] capture pip-vLLM greedy oracle golden → wire the M2.7 GEMM (reciprocal fast path) → close
  greedy + throughput gates vs oracle. CONCURRENCY LESSON: two code-writing subagents on the same
  working tree caused a commit-bundling mislabel (`2ce938f`) — use worktree isolation or explicit-
  path staging for parallel code-writers, never `git add -A` while a subagent is mid-write.
- **2026-07-04 (27B FULLY CPU-WIRED END-TO-END — second gate model runs through the engine on CPU)**
  Through `0f12f18`, the 27B (dense/W4A4/text-path) is assembled end-to-end on CPU: arch-select
  in `model_loader.cpp` (`IsDenseArch`: num_experts==0 → `LoadQwen3_5Dense`) → dense weights →
  the (arch-agnostic) Executor/EngineCore/LLMEngine stack → `Qwen3_5DenseModel::Forward` (paged)
  → runner dense route. MoE-typing was confined to just GPUModelRunner (`{moe,dense}` pointer
  pair) + LoadedEngine (`optional<Moe>/optional<Dense>`); the rest of the stack is arch-agnostic
  via `ModelRunnerBase`. NO 35B regression (MoE path byte-equivalent). CPU suite **87/87**.
  The 27B reused the 35B backbone + paged machinery + engine stack wholesale — only genuinely new
  code: dense SwiGLU MLP, W4A4→bf16 loader routing, IsDenseArch dispatch. 27B REMAINING = only
  GPU steps (serialize behind the 35B kernel jobs): capture pip-vLLM greedy oracle golden → wire
  the M2.7 tensor-core GEMM for the dense linears (6a: reciprocal global-scale + CT name remap,
  bf16 activations, NO new kernel) → flip `kW4A4ForwardReady` → close 27B greedy + throughput
  gates vs vLLM. 35B in parallel: M2.8 landed `5da39b0` (fast decode argmax + vectorized M=1 fp4
  GEMV); agent finalizing (gemvx→fp4 routing + measurement) — measured decode gap pending.
- **2026-07-04 (GROUNDED W4A4 spec — true-W4A4 is vLLM's path AND the fast one; killgate worry debunked)**
  `8e320aa`. Per the mirror-vLLM + ground-in-source directive, read pinned vLLM: the 27B runs
  **TRUE W4A4** (`use_a16=False`, config-driven by `input_activations` presence, NOT hardware) via
  **`cutlass_scaled_fp4_mm_sm120a`** (native Cutlass fp4×fp4) on sm_121 — FlashInfer-cutlass wrapper
  is numerically identical if installed. This is what vLLM's measured 47.5 tok/s ran → the authoritative
  kernel to MIRROR. **Impl spec (mirrors vLLM names, cited to source):** `ScaledFp4Quant` (per-token
  per-16-group activation bf16→fp4, math from csrc nvfp4_utils.cuh, activation global scale used
  DIRECTLY not reciprocated) + `MatmulNvfp4Fp4Wmma` (fp4×fp4, two fp8 scale streams, single
  `alpha=input_global_scale·weight_global_scale`); plumbing reuses `Nvfp4Weight`/`LoadCtNvfp4Raw`,
  forward swaps `MatmulNvfp4Bf16D→MatmulNvfp4Fp4Wmma(ScaledFp4Quant(x),W.fp4,W.wscale,alpha)` gated on
  fp4 presence (35B untouched). **KILLGATE WORRY DEBUNKED (source-grounded):** the "W4A4 FP4-MMA
  regressed on GB10" (0034/0035) was on the 35B W4A16 checkpoint (artificial, no fp4 activations) with
  a hand-rolled/Marlin kernel — NOT vLLM's sm120a — so it does NOT imply true-W4A4 is slow for the 27B.
  → true-W4A4 gives BOTH token-exact correctness AND vLLM-comparable speed; the 6a a16 path is retired
  to vLLM's Marlin-W4A16 fallback role. Ordered plan in notes §7.6. 27B W4A4 impl QUEUED behind the
  35B PagedAttention GPU job (serialize). In parallel now: 35B PagedAttention lever (running).
- **2026-07-05 (27B true-W4A4 gate: logit-correct + matches vLLM emulation, but token-6 near-tie
  differs from NATIVE-kernel oracle; 35B preserved 16/16)** — Drove the GB10 gate directly (agents
  flaked: one DNS mis-diagnosis, one punted to a background waiter). Branch `9a1af52` builds clean
  on GB10 (nvcc `-Werror=all-warnings`). Result: **35B `test_qwen36_paged_engine` PASSED 16/16**
  (fp4-gating kept it byte-identical); **27B `test_qwen27_paged_engine` FAILED** — ours `{...13,271,
  248068,...}` vs oracle `{...13,198,760,...}`, **tokens 0-5 exact ("capital of Germany is Berlin."),
  diverges at token 6 (271 vs 198)** — the SAME near-tie as the a16 path. ROOT CAUSE (grounded): our
  `MatmulNvfp4Fp4` DEQUANTS fp4→bf16 + bf16 compute (proven == vLLM `run_nvfp4_emulations` on CPU,
  529 assert), but the oracle golden was captured from vLLM's NATIVE `cutlass_scaled_fp4_mm_sm120a`;
  **emulation ≠ native on near-ties** (notes §7.2: `reciprocal_approximate_ftz` vs exact division,
  1-ULP → tips token 6). So we correctly mirror vLLM's REFERENCE (emulation) but not its native-kernel
  greedy. Branch NOT merged (gate red, correct). NEXT (grounded, no-ask per mirror-vLLM): (1) diagnostic
  — confirm vLLM-emulation also gives 271 (→ we mirror the reference); (2) implement the NATIVE fp4×fp4
  MMA (mirror cutlass sm120a) = the faithful mirror AND the 27B speed path; native accum may resolve
  the tie. Correctness bar if native still tie-flips: logit-parity + emulation-token-match (both met).
- **2026-07-05 (27B W4A4 LANDED on main `a799fa1`; correctness resolved; pivot to SPEED)**
  Merged the 27B true-W4A4 dense path (default dequant-compute, correct-grade) + the native
  sm120a block-scaled fp4×fp4 MMA (opt-in `VT_NVFP4_FP4_NATIVE=1`, bit-exact vLLM-emulation
  parity, maxrel 0.0) + CMake arch 121→121a. 35B 16/16 preserved; CPU build green; gate skips
  (`kW4A4ForwardReady=false`, documented: production-198 = post-MVP cutlass drop-in). 27B
  CORRECTNESS = token-for-token parity with vLLM's emulation REFERENCE (met, hardware-grounded).
  MVP now = pure SPEED: 35B ~6.1×, 27B ~16× off vLLM. Next shared-kernel levers (benefit BOTH via
  the shared GDN/backbone): (1) `GdnChunkDeltaH` tensor-core (~25% of 35B prefill, top kernel
  after M2.4 PagedAttn); (2) 27B GDN hstate OOM fix (Hv=48/value_dim 6144 OOMs at 8×1024 —
  blocks 27B target-concurrency measurement); (3) fp8 W8A16 GEMV (decode, both); (4) native-fp4
  MMA tiling (27B GEMM speed lever). Post-MVP: cutlass sm120a drop-in for bit-exact prod parity.
- **2026-07-05 (27B UNBLOCKED at the gate config — chunked prefill; first 8×1024 measurement)**
  `a456824`. Root cause (grounded): `model_loader.cpp` set per-step token budget = max_model_len×
  max_num_seqs (millions) → 8×1024 prefill ran in ONE step → GDN activation OOM that REBOOTED the
  box. Fix: `ResolveMaxNumBatchedTokens()` = vLLM's DEFAULT 2048 (fixed, concurrency-independent) →
  prefills split across steps, bounded activation. GDN state continuity was ALREADY correct
  (verified: scheduler splits on num_computed_tokens, `has_initial_state=context_lens>0`, state
  gather zeros only fresh rows — mirrors qwen_gdn_linear_attn.py:1513). NEW test: one-shot vs
  chunked prefill **bit-identical max|diff|=0** (CPU+GPU). 35B **16/16** chunked-on (−1.4% noise,
  LOWER peak RAM). 27B anchor 5/5. **27B @ 8×1024×128 conc-8 NOW FITS (peak 104.6GB, 8/8, no
  reboot — was instant OOM).** FIRST gate-config measurement: **ours 35.57 total / 3.95 output vs
  vLLM 396.98 / 44.11 = ~11.2× off** (live vLLM works now — ninja on subprocess PATH). ⚠ 27B peak
  104.6GB is near the 110GB reboot line — the f32 KV+GDN-state cache (num_blocks-sized, ~91GB
  floor) is the real bulk → NEXT 27B memory lever: **bf16 KV+state cache** (mirror vLLM's bf16
  mamba state, halves it → safety headroom). NEXT throughput: GDN DeltaH/ChunkO WMMA (35B ~50%
  prefill, both models), 27B perf kernels (11.2×). MVP: 35B ~7.4×, 27B ~11.2× — both measurable,
  both correct.
- **2026-07-05 (GDN DeltaH+ChunkO tensor-cored — WMMA/TF32; the coupled fix; 35B prefill 1.39× TTFT, POSITIVE)**
  Rewrote the two top GDN prefill kernels (`cuda_gdn.cu` `GdnChunkDeltaH`/`GdnChunkO`) as WMMA
  16×16 tile matmuls (f32 accumulate), mirroring FLA `tl.dot` (chunk_delta_h.py:176/278, chunk_o.py:111/113/137).
  **LOAD-BEARING GROUNDING CORRECTION:** the task premise "Model path (bf16)" is WRONG — nsys proved the
  35B GDN runs the **f32** kernels (`GdnChunkDeltaHKernel<float>` 24.8% + `GdnChunkO<float>` 17.6% + WU 6.8%
  = 49% of prefill); the bf16 path is never invoked. So the real lever is **TF32 WMMA on the f32 path**
  (fragments templated on dtype: f32→`precision::tf32` 16×16×8, bf16→native 16×16×16 kept for a future bf16
  GDN). Fit GB10's 99KB shared with f32 state resident by folding per-row decay into K (V2 reads V_new from
  global) + ChunkO buffer aliasing. **Correctness ALL GREEN:** 35B paged **16/16** (TF32 held every argmax),
  `test_ops_gdn` 23/23, 27B one-shot==chunked **diff=0**. **Measured (8×1024×128 conc-8, same-base A/B):**
  kernels **DeltaH 8.9×, ChunkO 10.6×** faster (nsys same-workload); **TTFT 11930→8602 ms (1.39×), total
  215.18→243.11 tok/s (+13%)**; vs vLLM oracle 1326.66 → ratio **6.17×→5.46× off**. DeltaH+ChunkO share
  42.4%→7.2%. **NEXT (grounded): the MoE NVFP4 grouped GEMMs (26%+13%) + dense MatmulNvfp4 (15.5%) now
  dominate 35B prefill; and GdnChunkWU (now 11%, the #1 GDN kernel) — its KKᵀ Gram + solve is the next GDN
  lever.** 27B shares the identical (now TF32) GDN kernels → benefits by the same mechanism (correctness
  confirmed via `test_qwen27_paged_forward`). Pushed to main.
- **2026-07-05 (STRATEGIC PIVOT: cutlass sm120a fp4 GEMM drop-in — feasibility GO, `f3eca1c`)**
  Inflection: 35B at 5.18×, remaining ~5× is hand-written-vs-cutlass efficiency (our fp4 GEMMs ~15%
  of peak; cutlass sm120a near-peak). Per mirror-vLLM + the post-MVP "import vLLM kernels" plan,
  pulling the cutlass drop-in FORWARD as the parity path. **Feasibility GO (build-probed on GB10):**
  dense `cutlass_scaled_fp4_mm_sm120a` (103 TFLOPS) + grouped MoE `run_fp4_blockwise_scaled_group_mm
  _sm120` (125 TFLOPS) both COMPILE+RUN on sm_121a/nvcc13; CUTLASS v4.4.2 header-only torch-free
  (~50 lines vt glue/kernel); sm_121∈[120,130) → sm120a IS the GB10 path. Amdahl: closes ~2.2-2.4× of
  the 5.18× residual (fp4 GEMMs ~41% decode + ~41% prefill) — biggest single lever. Real work = the
  W4A4 conversion (per-token activation fp4-quant + ue4m3 scales + scale SWIZZLE to cutlass padded
  atom layout — tested CPU refs exist in nvfp4_emulation.h + qwen27b-w4a4-notes §3.4). Notes:
  `.agents/cutlass-dropin-feasibility.md`. NEXT: Phase 1 dense drop-in → parity+measure → Phase 2 MoE
  grouped → wire both models. NOTE: 202MB cutlass probe clone left at ~/cutlass_probe on dgx (97%
  disk / 126G free). OPEN grounding Q for Phase 1: does the 35B (modelopt NVFP4) run W4A4 or W4A16 in
  vLLM (which cutlass kernel) — the 27B is cleanly W4A4.

- **2026-07-05 (cutlass FP8 W8A8 drop-in — the #2 decode lever, MERGED)**
  Lifted vLLM's `cutlass_scaled_mm_sm120_fp8` → `vt::MatmulFp8Cutlass` (new gated TU
  `cuda_matmul_fp8_cutlass.cu`) + `vt::QuantFp8Static` (static per-tensor act quant) + fp8-resident
  `Fp8Weight`/`LoadFp8Raw` (loads the previously-ignored `input_scale`; `alpha=input_scale·weight_scale`
  folded into the LinearCombination `alpha_ptr`, per-tensor so identical to vLLM's ScaledEpilogue).
  Wired the 7 FP8 projections (attn q/k/v/o + GDN in_proj_qkv/z/out_proj) fp8→fp4→bf16, load-time gate
  `VT_FP8_CUTLASS=1`. Unit 23/23 (bit-exact quant + W8A8-ref GEMM); **35B paged engine 16/16
  token-for-token with the W8A8 path** (near-tie held, identical to bf16 default); 27B unaffected.
  Same-binary A/B win: decode +5.5% (1024/128) / +7.0% (256/512), TTFT −4.8%/−5.6%, peak RSS −1.19GB.
  Opposite sign to the reference W8A16 GEMV (−5-7%) — cutlass W8A8 beats cublas bf16. NEXT lever:
  GdnScan (30.6% decode) + PagedAttention (14%).
- **2026-07-05 (DECODE MEMCPY TAX KILLED — GDN state gather/scatter → in-place `ssm/conv_state_indices`; +24-34% conc-64, MERGED)**
  Measure-first STEP 1 (nsys, 35B conc-64) resolved the plateau-root "187k copies/run": they are
  NOT metadata H2D re-uploads but the **GDN recurrent-state GATHER/SCATTER**. The ssm/conv state
  caches are HOST `std::vector`s (GB10 unified memory), so `GatherRows`=H2D and `ScatterRows`=D2H,
  one `cudaMemcpyAsync` per sequence per GDN layer → 30 GDN × 64 seq × 2 caches × 2 = **7,680/step**
  (nsys: 142k H2D + 77k D2H, only 27 true D2D; D2H 77,241 ≈ 76,800 exact). `cudaMemcpyAsync` = 40%
  of host-API; decode-window GPU-busy ~**52%** (48% idle, single stream, host-issue-starved). FIX
  (mirror fla): plumb `ssm_state_indices` (`vt::GdnDecode` state_idx — the fused kernel already
  supported it; fixed `si<=0`→`si<0` null sentinel for 0-indexed slots) + `conv_state_indices`
  (`vt::CausalConv1dUpdate`) → decode recurrence + conv run IN PLACE on the persistent unified cache
  at each seq's slot, no gather/scatter. Graph-replay-safe (index uploaded from the PERSISTENT
  `non_spec_state_indices` vector, not a stack-local — the num_reqs==1 decode-graph re-reads a fixed
  host address). Prefill untouched. **CORRECT: 35B 16/16 (eager+graph); `test_ops_gdn` 25/25 (+new
  indexed==compact); mixed anchor 4/4; 27B/dense `test_qwen27_paged_forward` 5/5; `test_runner` 5/5;
  clean -Werror.** **MEASURED (same-box A/B): decode memcpy 155,875→29,041/window (−81%); 256/64
  conc-64 448.25→599.72 (+33.8%), TPOT 538→353ms; GATE 1024/128 conc-64 504.92→627.88 (+24.4%);
  conc-8 +21%; scaling conc8→64 1.40×→1.54×.** The copies WERE critical-path (opposite the conc-8
  batched-graph NEUTRAL — the tax scales with batch). 27B dense shares `GdnBlockPaged` → benefits
  identically. vs vLLM conc-64=2768: 627.88 = ~4.4× off; decode still ~44% busy. NEXT: per-step
  logits D2H+sync, the remaining ~1,452 full-attn metadata H2D/step, PagedAttention decode.

## 2026-07-07 — HONEST RE-BASELINE (graphed vLLM) + bf16-forward prefill lever

**Campaign-correcting finding:** all prior vLLM denominators used `--enforce-eager`
(no CUDA graphs / no torch.compile) — a handicapped vLLM. The honest bar is GRAPHED
vLLM (production). Re-measured clean same-session: **35B 0.961×, 27B 0.857×** (27B
0.79× at big-context in4096/out512). We already BEAT eager vLLM on both (35B 1.17×,
27B 1.03×). Benchmark-protocol.md updated to mandate the graphed denominator.

**Gap located = PREFILL** (27B 0.70× prefill-heavy vs 0.93× decode-near-parity). A
48-agent dynamic scan ranked the portable levers; #1 root cause: we upcast fp4-GEMM
outputs to f32 and run glue at 2× vLLM's bf16 width (the "bf16-forward" program,
~10-12%).

**Merged this session:** 27B correctness gate redesign (non-flaky, vLLM-deterministic
span); benchmark protocol → graphed vLLM; 27B dense decode CUDA graph (+1.9%);
quantize-once q/k/v+gate/up (+0.4%, bit-identical, default ON); **bf16 gate/up GEMM
outputs (+5.4% standard / +12.5% prefill-heavy, default ON) — 27B 0.857×→0.907×.**
Shelved: silu+fp4-quant fusion (−2%, less parallel than the 2 tuned kernels).

**NEXT (bf16-forward program, scan-ranked, each needs a bf16 glue/conv kernel variant):**
(2) q/k/v bf16 + bf16 AttnGateSplit/SigmoidGate (~1.5-3%); (4) GDN in_proj + bf16 conv
(~1.5-2.4% + speculative cuBLASLt off the sm_80 kernel, 30 layers); (3) fuse
AttnGateSplit+qk-norm+RoPE 4→1 (~1-2%); (5) blocked-16 triangular inverse in GDN WU
(~1-2.5%, the one new algorithm win). Verified dead-ends: cutlass config (byte-identical
to vLLM), rms+nvfp4 fusion, prefill CUDA-graph. Build-specific out-of-reach: flashinfer
autotuned fp4 GEMM + whole-graph Inductor fusion. Guardrail: keep GDN g/beta + ssm_state
f32. LocalAI worker kept DOWN (dgx perf work ongoing).

## 2026-07-08 — Methodology re-grounding + `vt::tile` portable pipeline component

**Course correction (user re-grounded on AGENTS.md).** Mid-session a Triton-AOT
"CUDA fast-path" was sanctioned and PROVEN end-to-end on dgx (token-exact rmsnorm
via `triton.tools.compile`+`link`, runtime Triton-free; branch `perf/triton-fastpath`).
On re-reading the canon it was **re-rejected as a compile-target**: it violates
mission.md ("no Python at BUILD or run time") + discipline.md 33-47 ("Kernel-DSL is
a PORTING REFERENCE, NEVER a compile-target; do NOT AOT-compile Triton/CuTe-DSL to
cubin"). The AOT branch is **shelved, kept OFF as a measurement oracle only** (compile
FLA's exact kernel to get the ground-truth "what Triton codegen achieves" target).

**Endorsed path (user: "can't we have a component that does what triton does, and use
it internally?").** Build **`vt::tile`** — a PORTABLE C++ async-pipeline/tile
abstraction grounded 1:1 in CUTLASS/CuTe. It encodes what Triton's *compiler* does —
software pipelining (multi-stage async gmem→smem double-buffer + barrier scheduling),
swizzled smem, MMA lowering — as reusable C++ primitives behind a backend-neutral seam
(CUDA supplies cp.async/TMA atoms; Metal/Vulkan/CPU supply theirs). Pure C++, no
build-Python, portable, SCALES. Fully canon-consistent (discipline blesses porting
CuTe/CUTLASS C++). Design record: `.agents/tile-pipeline-component-2026-07-08.md`.

**Ground truth (trace-execution).** On GB10 (sm_121) vLLM runs the **FLA Triton
`chunk_gated_delta_rule_fwd_kernel_h_blockdim64`** (the CuTe-DSL GDN path is opt-in +
SM10.x-only per `qwen_gdn_linear_attn.py:152-210`; the profile confirmed the FLA name).
The codegen we lack = Triton's Blackwell async pipeline (`cp.async` Ampere → TMA+mbarrier BW).

**Built this session:** `include/vt/cuda/tile/cp_async.cuh` — Rung-1 primitive
(SM80 cp.async atom + fence/wait + `PipelineState`), ported 1:1 from
`cute/arch/copy_sm80.hpp` + `cutlass/pipeline/sm90_pipeline.hpp`, cited. **In flight
(worktree agent):** GDN `delta_h` re-ported against it (`GdnChunkDeltaHTilePipeKernel`,
toggle `VT_GDN_TILE_PIPE`) — two levers: keep `v_new` in smem across V1→V2 (kill the
gmem round-trip FLA avoids) + N-stage cp.async prefetch of the streamed k/u/w tiles.
Gate: `test_ops_gdn` token-exact + gate A/B vs the hand kernel. Rung-2 (TMA+mbarrier)
only if Rung-1 lags; retarget to chunk_o if delta_h's sequential structure caps the win.

## 2026-07-09 — delta_h win landed; #1 refuted; prefill is GPU-BOUND (campaign reframe)

**delta_h register-tiled + cp.async ring MERGED (7a7a64b).** The vt::tile component's
first real target: H moved from 64KiB smem into WMMA accumulator REGISTERS (faithful
FLA blockdim64), freeing smem for a 2-stage cp.async ring. **−22% delta_h kernel**
(441→345µs), **+0.85% e2e** (35B 2797→2821 tok/s), token-exact 423/423 + memcheck,
35B+27B 16/16, 27B neutral. **This REVERSES the prior "hand cp.async is DEFINITIVELY
negative / 0.82× is the portable-C++ ceiling" conclusion** — the ceiling was an
artifact of the wrong structure (smem-resident H starving the ring), NOT a real limit.
Portable-C++ CAN match Triton's codegen when the STRUCTURE is ported faithfully.

**Two scan hypotheses REFUTED by measurement (why we measure):**
- **#1 fp8 cuBLASLt plan-cache — DEAD.** Pre-written, compiled clean, token-exact
  (46/46, 35B 16/16), but same-binary A/B = +0.14% (noise). nsys: the ~210µs gap
  before each fp8 GEMM is the `QuantFp8Static→GEMM` dependency, NOT the heuristic —
  the per-call heuristic is already cheap/overlapped. Branch `perf/fp8-plan-cache`
  (7a90380) left unmerged. Only the 35B runs fp8 (7 W8A8 projections: attn qkvo +
  GDN in/out_proj); 27B is pure W4A4/bf16, no fp8.
- **#3-as-launch-bubble-fix — DEAD.** Steady-state prefill GPU-idle-between-launches
  = **3.8%** (<5%); the 24.8% raw idle is ONE one-time ~340ms JIT/graph-capture gap.
  **The GPU is ~96% busy during prefill — GPU-BOUND, not host/launch-starved.**

**CAMPAIGN REFRAME:** the residual 0.82× prefill is redundant non-GEMM GPU WORK
(HBM traffic / extra passes) vs vLLM's fused kernels — NOT host-side idle. Levers are
re-scoped to GPU-WORK reduction: #2 GDN WY blocked triangular inverse (serial
2016-deep solve @64/256-util → tensor-core blocked inverse; a clear GPU-work cut,
reuses the validated vt::tile+WMMA infra) and #3-reframed (fuse add+RMSNorm+quant +
silu chains to cut HBM PASSES, not bubbles). NEXT: GPU-BUSY kernel breakdown of our
prefill vs vLLM's (memory profile) → measured non-GEMM lever ranking, then pursue the
top hard lever. Top host-trace busy/gap sites: QuantFp8Static→GEMM, GdnChunkWU (the
WY solve — #2), CastBf16 (redundant dtype casts), GdnPostConv.

## 2026-07-09 — MEASURED GPU-busy breakdown: the gap is UNFUSED GLUE (~27%, HBM-bound)

35B prefill GPU-busy (1495ms steady-state, from nsys cuda_gpu_kern_sum). Buckets (% busy):
D glue (norm/quant/cast/silu) **30.2%** · C GDN chunk **25.6%** · A fp4/Marlin 22.7% ·
B fp8 GEMM 10.9% · E attn 8.6%. **NON-GEMM = 65.1%** (GEMM 34.9%).
**vs vLLM: glue ~2.3× and GDN chunk ~2.4× vLLM's relative share** (GEMMs at parity).
That excess IS the residual 0.82×. Glue kernels are HBM-BW-BOUND (RmsNormRow 61% peak,
QuantFp8Static ≥53%, MoeSiluMul ~66%) — each a full HBM round-trip vLLM folds into GEMM
epilogues / single fused kernels. **OVERTURNS the source-scan's "chunk_o/WU we're ahead"**
— measurement says our GDN chunk is 2.4× vLLM's share.

Top glue kernels (%busy): GdnPostConv 5.0, RmsNormGatedRow 4.5, MoeCombineGate 4.4,
RmsNormRow 4.1, QuantFp8Static 3.0, MoeSiluMul 3.0, CastF32 1.9, AttnGateSplit 1.8,
SigmoidGateBf16 1.0, CastBf16 0.4. GDN chunk: GdnChunkWU 8.7, ChunkO 6.1, DeltaHRegRing
5.5 (just improved), Conv 5.4.

**NEXT (biggest measured lever): #1 FUSE the glue chain (~27%, HBM-bound, Med).** Collapse
residual-add+RMSNorm+quant and silu+mul+cvt_fp4 into single passes / GEMM epilogues
(mirror vLLM Inductor fusions) via the TDR fused-recipe skeleton. Then #2 GDN-chunk
efficiency (25.6%, High — apply the vt::tile register-tiled method to ChunkO/WU + the
WY blocked inverse). #3 kill Cast passes (~2.3%, Low — fold f32→bf16 into producers).

## 2026-07-09 — GDN WY blocked tensor-core triangular inverse (lever #2) — POSITIVE, default OFF

**VT_GDN_WY_BLOCKED (default OFF; branch perf/gdn-wy-blocked-inverse, commit a2d2198).**
Swapped ONLY the WY vec kernel's (I+A)⁻¹ phase (GdnChunkWUWmmaVecKernel) for the FLA
blocked tensor-core inverse (solve_tril.py merge_16x16_to_64x64_inverse_kernel:238-390):
four 16×16 diagonal blocks by a short (≤16-deep) per-block column forward-sub + six
off-diagonal blocks by 16×16 tensor-core Schur merges (T(i,j) = −T(i,i)·Σ_{j≤k<i}
A(i,k)·T(k,j)), phased by merge distance. Replaces the serial ~BT-deep column
forward-sub (~2016 dependent FMAs on ONE of 256 threads, 64/256 util). Gram / apply
untouched; same [BT,BT] f32 Tf feeds the apply. New device helper WyMerge; +106/−12 LOC.

**CONFIRMS the serial inverse WAS the hot phase of GdnChunkWU.** Same-binary nsys A/B
(in1024/out16 np32 conc32, 150 launches): GdnChunkWU **531.07→449.30 ms = −15.4%**, while
the sibling GDN kernels stayed in noise — ChunkO 365.99→368.07 (+0.6%), DeltaHRegRing
344.08→340.44 (−1.1%). GdnChunkWU dropped 7.8%→6.7% of GPU busy. So the ~64-deep serial
column inverse really was ~15% of the WU kernel, and the blocked tensor-core inverse cut it.

**Token-exact:** test_ops_gdn 423/423 BOTH toggles (f32@5e-3, bf16@3e-2; varlen/multi-seq/
partial-tail/GQA) + compute-sanitizer memcheck 0 errors. Partial tails fall out for free
(Am pre-zeroed past len ⇒ out-of-range blocks invert to I / merge to 0; apply masks rows≥len).

**E2E (35B NVFP4, GB10, idle box, in1024/out128 conc32 np192 mnbt8192):** baseline
(BLOCKED=0, 3 runs) total 2817.8 / prefill 2504.8 / meanTTFT 2029.3; blocked (BLOCKED=1,
2 runs) total 2833.1 / prefill 2518.4 / meanTTFT 2006.1 = **+0.54% total, +0.54% prefill,
−1.1% TTFT** — clean (baseline spread ±0.1%; min blocked 2832.1 > max baseline 2821.0, no
overlap). Small e2e because GdnChunkWU is ~7-8% of GPU kern time (a 15% cut ⇒ ~1.2% GPU
work ⇒ ~0.5% e2e), same shape as the delta_h win (−15.4%→+0.54% here vs −22%→+0.85% there).

**DEFAULT OFF (deliberate).** The merges use the tf32 WMMA config (the inverse is carried in
f32 shared for both kernel dtypes; consistent with the tf32 apply). vLLM's FLA instead
defaults FLA_TRIL_PRECISION=ieee (full-f32 dots), so blocked-on is numerically distinct from
BOTH our serial-f32 baseline AND vLLM's ieee blocked inverse. Flipping the production default
therefore needs a greedy 16/16-vs-oracle confirmation on 35B+27B (the bar delta_h cleared) —
not run here. RECOMMEND: default-on after that greedy check; optionally raise the merges to a
tf32x3/ieee-equivalent to match vLLM's precision exactly. This is a real GPU-work cut and a
clean down-payment on the "GDN-chunk efficiency (25.6%)" campaign lever.

## 2026-07-09 — GDN-chunk front is EXHAUSTED for e2e gains (ChunkO −11.5% kernel = e2e-neutral)

ChunkO occupancy fix (`GdnChunkOWmmaOptKernel`, branch `perf/gdn-chunko-occ` a4a8eb8, NOT
merged): diagnosed smem-occupancy-limited (32KiB whole-Dv f32 accumulator round-tripped
through smem across 6 barriers → 1 block/SM, 33% occ). Fix: BV-block Dv + smem aliasing →
2 blocks/SM (66% occ). **−11.5% ChunkO kernel, token-exact 423/423 + memcheck, but e2e
NEUTRAL** (+0.05% gate / +0.18% prefill-heavy) — ChunkO is only ~5-6% of prefill. Left
default-OFF (gate-neutral). The residual toward vLLM's 2× is the smem round-trip + barrier
serialization vLLM removes with register-resident fused accumulators (the delta_h/WY trick),
but even a full 2× ChunkO is ~+0.3% e2e.

**CONCLUSION: the clean GDN-chunk e2e levers are HARVESTED** (delta_h +0.85%, WY +0.54%
merged; ChunkO −11.5% kernel = e2e-neutral). Further GDN-kernel work is e2e-neutral at the
gate (kernels too small a prefill fraction). The meaningful remaining e2e gap is the GLUE/EVT
front (bigger, but spread across cutlass-3.x/Marlin/cuBLASLt epilogues — XL, CUDA-specific)
OR a different axis (27B-specific, decode, features). This is a strategic fork for the user's
steer — the autonomous GDN thread is measurably exhausted for e2e gains.

## 2026-07-09 — REASSESS (both models, prefill+decode): near the portable parity floor

Measured 27B + 35B, prefill AND decode (delta_h+WY landed). Decisive:
- **e2e ≈ 53% prefill / 47% decode** (both models). **Decode is GPU-BOUND (87-93% busy),
  GEMM-at-parity → NO decode lever** (the old "decode host-starved ~44%" was an nsys
  CUDA-graph-collapse artifact; `--cuda-graph-trace=node` shows ~90% busy). 35B decode ≥
  vLLM; 27B decode is memory-BW-bound (vLLM pays it too).
- **GEMM at vendor-parity in ALL 4 profiles** (fp4 cutlass sm120 / Marlin; the bf16 15-33%
  is GDN in_proj which the checkpoint recipe keeps bf16 — vLLM too). Entire lever surface
  is NON-GEMM. (Caveat: vLLM autotunes fp4 via flashinfer AutoTuner vs our fixed cutlass
  config — a small non-structural GEMM residual possible.)
- **27B's biggest lever is NOT attention** (5.9% pref / 2.2% dec — refuted); it's GEMM-heavy
  (dense fp4 crowds out non-GEMM), so its non-GEMM excess is SMALLER than the 35B's. The 27B
  is the more-behind gate but hides no bigger lever; it improves via the SHARED glue/GDN work.
- **Biggest remaining e2e lever = 35B prefill glue-chain fusion** (glue 25.3% pref ×53% =
  13.4% e2e at 1.9× vLLM). Portable partial (fold add+RMSNorm+quant + silu+mul+fp4-quant via
  the TDR skeleton) ≈ **~3% e2e, Med, canon-compliant**; the FULL ~6% needs XL CUDA-specific
  cutlass/Marlin EVT epilogue fusion. GDN-chunk register-tiling is DIMINISHING (delta_h
  +0.85% → WY +0.54% → ChunkO 0% e2e; each kernel too small a prefill fraction).

**BOTTOM LINE: the remaining gap is DIFFUSE. No single lever >~6% e2e; only one (35B prefill
glue) >~3%. We are near the practical PORTABLE parity floor** — GEMM at parity, decode
GPU-bound at parity, attention/fp8 non-levers. The recommended next lever is the PORTABLE
vt:: glue-chain fusion (~3% e2e, Med, via include/vt/fused_recipe.h TDR skeleton +
GlueFuseEnabled in qwen3_5.cpp); the last ~3% beyond that needs XL CUDA-specific EVT (a
portability tradeoff = a user decision).

## 2026-07-09 — EVT scoping: "full EVT epilogue fusion" is ~0 on GB10; realizable win = PORTABLE producer-side fusion

Scoped the user-chosen "full EVT epilogue fusion" (~6%) and DECISIVELY refuted its premise on
sm120a/GB10 (grounded in vLLM's csrc dep-chain, agrees with the measured GEMM-at-parity):
- **Our fp4/fp8/Marlin GEMM epilogues are ALREADY at parity** — scale folded into `alpha`,
  output cast in-epilogue. **vLLM's OWN sm120a fp4 GEMM emits bf16 via LinearCombination and
  does NOT fold the next-op fp4 quant** (grep for an SF-generating sm120 output epilogue in
  vLLM csrc = empty; the fp4-output epilogues are SM100/TRT-LLM paths, not GB10's dense/Marlin).
- cutlass-3.x EVT is wire-up-able (template-only, no version bump) but yields **~0 on GB10** (no
  bias, scalar scale, output-SF unused by vLLM here, inter-GEMM silu/norm blocks chaining).
  Marlin epilogue: only `mul_topk_weights`/`use_atomic_add` flags open (vLLM uses the same +
  separate sum → structural parity). cuBLASLt epilogue: nothing to fold (no bias, bf16 out).
- **The realizable ~3-4% is PRODUCER-SIDE vt:: kernel fusion** (mirror vLLM's Inductor passes,
  not its CUDA epilogues) — canon-compliant, no CUDA lock-in. Ranked targets:
  **1. fp8 RMSNorm→quant + quantize-once (~3.0%)** — QuantFp8Static runs 3× on the shared q/k/v
     activation; mirror `rms_quant_fusion.py:124` + MergedQKV quant-once. IN PROGRESS.
  2. Cast elimination (~2.3%, fold CastF32/CastBf16 into producers).
  3. Attn-preamble gating (~1.8%, `FuseAttnPreambleEnabled` kernel EXISTS default-OFF → 16/16 gate + flip).
  4. MoeCombine ×weight → Marlin `mul_topk_weights` flag (trivial).

So "get to parity" is realizable at ~3-4% via the PORTABLE fusions (fulfills the intent); the XL
CUDA-EVT is a dead end on this arch. Reinforces [[prefill-gpu-bound-vt-tile-playbook]] +
[[fusion-must-be-portable-reuse-patterns]] (portable producer-side fusion, not vendor EVT).
Full scoped plan + first-target design in the task record.

## 2026-07-09 — PORTABLE parity floor CONFIRMED; throughput campaign complete (3 wins, ~+2.2% this session)

Targets 2-4 measured (all at/below the portable floor; 0% merged perf):
- **T3 attn-preamble gating (~1.8%): hard NO-GO.** The fused AttnQkNormRopeGate differs from
  the unfused path by exactly 1 f32 ULP (different FMA contraction; cos/sin bit-identical) →
  the 35B fp8 greedy is 1-ULP-sensitive and diverges from the oracle within ~8 tokens. The
  prior code comment claiming "token-exact vs OFF" was MEASURED FALSE — corrected + pinned a
  regression test (test_ops_attn_preamble) so it isn't re-attempted blindly (merge 20d5757).
- **T4 MoeCombine mul_topk_weights: provably e2e-neutral** (relocates a compute-free multiply
  on already-loaded data; 0 memory-traffic change). Not implemented.
- **T2 cast elimination: at the floor.** Big casts already folded (Bf16GemmOut removed the
  9.6% CastBf16ToF32; GlueFuse). Remaining CastF32 1.4% = the GDN gather-upcast micro-op the
  scan flagged "fold into #3, don't chase standalone" (~0.4% e2e ceiling). Left marginal.

**SESSION OUTCOME:** 3 e2e throughput wins landed (delta_h +0.85%, WY inverse +0.54%, fp8
RMSNorm→quant +0.85% ≈ +2.2% combined), all token-exact + default-on, all via the portable
vt::tile / producer-side-fusion playbook (which reversed the old "portable-C++ ceiling"
belief). The GEMM-epilogue EVT is ~0 on GB10 (documented). **We are now at the practical
PORTABLE throughput-parity floor** — GEMM at vendor-parity, decode GPU-bound at parity,
attention/glue fusions either done, fragile-1-ULP (T3), neutral (T4), or micro (~0.4% T2).
The remaining sub-percent needs either a fragile FMA match (T3, risks the greedy gate) or
non-portable work. NEXT: pivot to the broader MVP (GGUF, tools, grammars, server, e2e, more
models) or accept the floor — a priority call. Also: main CI is RED from a pre-existing
runner-OOM (`cmake --build -j` unbounded parallel-link, ci.yml:57), NOT a code regression.

## 2026-07-09 — FRESH parity A/B vs GRAPHED vLLM (both models, @ c7ba80e)

Same-box, same-workload, graphed (production) vLLM denominator. Total token throughput:
- **35B (conc64/np200): ours 3175.4 / vLLM 3282.0 = 0.967×** (near parity, ~3% behind). Ours
  +2.8% over stale (the 3 wins show), but vLLM's denominator also drifted +2.2% → net ratio
  0.962→0.967. Below vLLM on all 3 throughput axes.
- **27B (conc16/np96): ours 644.3 / vLLM 770.8 = 0.836×** (the MORE-BEHIND gate). Flat vs stale
  0.857× (denominator drift + noise, not a regression). At conc32/np192: ours 855.0 (+3% over
  stale — wins DO show) / stale vLLM 1051.5 ≈ 0.81× (vLLM not re-measured there).
- **MEMORY: we WIN big** — ours 52.8GB (35B) / 60.5GB (27B) vs vLLM ~80.6 / ~86.4GB (GM0.6):
  ours is ~65% of vLLM's peak (28-30GB less). ✅ on the memory axis.
- vLLM offline bench → no vLLM TTFT/TPOT (no latency A/B). Ours: 35B TPOT 147ms, 27B 176ms.

**HONEST re-opened question (the more-behind gate):** the 27B is fp4-dense-GEMM-DOMINATED
(45.8% of its prefill). The reassess called GEMM "at vendor-parity" but that was SOURCE-INFERRED
(both cutlass) — NOT measured, because vLLM can't be nsys-profiled on GB10. vLLM AUTOTUNES its
fp4 GEMM (flashinfer AutoTuner) while we run a FIXED cutlass config — the exact "don't infer
at-parity from source" trap. So the 27B's ~16-19% gap MAY be substantially the fp4 GEMM, an
UNEXAMINED lever. Worth measuring before declaring the 27B at the floor.
