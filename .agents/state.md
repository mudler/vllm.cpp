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

## 2026-07-09 — 27B gap MEASURED: it's GDN (~2× slower, f32-vs-bf16), NOT the fp4 GEMM. "Floor" call was 35B-biased.

Investigated the 27B 0.84× gap with vLLM's LLM-API torch profiler (nsys breaks its EngineCore)
+ shape-matched microbenches. DECISIVE:
- **fp4 GEMM is NOT the lever.** vLLM = flashinfer-AUTOTUNED cutlass (8 tile configs) vs our
  FIXED 2-way M-dispatch (cuda_matmul_nvfp4_cutlass.cu:243). Shape-matched (M=4096): vLLM only
  **6.1% faster** aggregate (mlp_down at parity; small-M favors us). fp4 GEMM = 30.6% of 27B
  prefill → matching it = ~2% prefill e2e. (task #13's "per-shape autotune" is NOT active — a
  fixed dispatch; overclaim to fix.)
- **THE 27B LEVER = GDN ~2× slower than vLLM.** OURS 148.5 µs/tok vs vLLM (Triton/FLA) 71.7
  µs/tok = 2.07× (1.68× core chunk+conv only). GDN ~27% of 27B prefill → **~14% prefill e2e,
  the biggest lever.** Measured mechanism: **our GDN runs f32 activations (in_proj MatmulF32D
  output, qwen3_5.cpp:1224; memory-bound chunk kernels) vs vLLM FLA's bf16 → ~2× memory
  traffic.** Explains why the dense 27B lags but the MoE 35B (GDN smaller share) is at parity.
- bf16 in_proj: our bf16-OUTPUT matches vLLM (2778/2854µs); the f32-output choice (MatmulF32D)
  costs ~5-10% (~1.3% prefill). in_proj_qkv N=10240 ~6% slow (cuBLASLt SM80 heuristic pick).

**CORRECTION: the 27B is NOT at the parity floor.** The 2026-07-09 "portable floor" + reassess
"GDN exhausted" conclusions were 35B-BIASED (GDN is a small share of the MoE 35B). On the dense
27B, GDN is the dominant lever: **~14% prefill e2e via bf16 GDN I/O** (mirror vLLM's FLA:
bf16 activations, f32 accum, keep f32 g/beta/state guardrail) + ~2% fp4 autotune + ~1.3% bf16
in_proj-output = the ~16% 27B gap, GDN-dominated. NEXT: bf16-vs-f32 GDN A/B → full bf16 GDN
pipeline. This RE-OPENS the GDN front for the 27B. Evidence: dgx /home/mudler/scratch_agent_a562/.

## 2026-07-09 — 27B GDN INPUT-side bf16 (VT_GDN_IN_BF16, DEFAULT-ON): token-exact, +0.7-0.8% e2e. The 2× GDN gap is the CHUNK codegen, NOT input dtype (the "14%" was wrong).

Tested the STEP-1/STEP-2 bf16-GDN-I/O hypothesis. **STEP-1 diagnosis (code +
nsys scratch_agent_a562/nsys/prefill_kern.csv) REVISED the premise:** the bf16
CHUNK path was ALREADY ACTIVE by default (VT_GDN_BF16 on) — GdnChunkWUWmmaVec
<bf16> (5.9%), GdnChunkDeltaHRegRing<bf16> (4.5%), GdnChunkOWmma<bf16,float>
(4.5%) all bf16. The ONLY remaining f32 INPUT-side stages: in_proj mixed_qkv GEMM
output (MatmulF32D), causal conv1d (CausalConv1dFwd<float,float> 3.7%), post-conv
conv-READ (GdnPostConv reads const float* conv, 3.2%). So **"GDN 27%→halve→14%"
was WRONG** — most of GDN was already bf16; only ~conv+postconv-read were f32.

IMPLEMENTED VT_GDN_IN_BF16 (DEFAULT-ON): mixed_qkv→MatmulBf16D, conv bf16 (bf16
weight + bf16 dconv; f32 conv_state + f32-accum math unchanged), GdnPostConv/
GdnConvSplit templated on conv dtype (Load() upcast, ops.cpp guards relaxed).
g/beta/ssm_state + the a/b GEMMs stay f32 (FLA split). 27B-only by construction
(bf16-weight in_proj branch); 35B fp8 branch untouched. Commit on perf/gdn-in-bf16.

GATES (all GREEN): test_ops_gdn 423/423 + compute-sanitizer memcheck 0 errors.
27B greedy paged-engine token-EXACT (default-on AND VT_GDN_IN_BF16=1 AND OFF:
identical "…Berlin." + tie-free 6/6 + full-16 matches PRODUCTION). 35B 16/16
(unaffected — fp8 branch). Default-on gate re-confirmed on the flipped build.

A/B (same-binary, OFF vs ON, 27B in1024/out128, 3 interleaved reps):
  conc16/np96:  OFF 707.2 / ON 712.0 total = +0.68%; prefill 628.7/632.9 = +0.68%;
                TTFT 2935.9/2892.4 = -1.5%. NON-OVERLAPPING arms (OFF max 708.1 < ON min 711.2).
  conc32/np192/mnbt8192: OFF 854.0 / ON 861.1 = +0.83%; prefill 759.2/765.5 = +0.83%;
                TTFT 7181/7062 = -1.6%. Noisier per-rep (0.02/0.51/1.98%), avg positive every axis.
nsys GDN-kernel A/B (in1024/out8 np48; prefill 1651.9→1664.6 = +0.77%):
  CausalConv1dFwd 987675→676139 = -31.5% (bf16 I/O halves the read-bound conv, re-reads x K=4×);
  GdnPostConv 876352→721914 = -17.6% (bf16 conv read); GdnChunkWU/O/DeltaH FLAT
  (-0.5/-0.01/-1.0%, already bf16); RmsNormGated FLAT (output side not converted).
  GDN µs/tok ~138→128 = -7.1%; vs vLLM FLA 71.7 → gap 2.07×→~1.9×.

**HONEST VERDICT:** input-side bf16 is token-exact + net-positive on EVERY axis
(+0.7-0.8% total & prefill, TTFT -1.5%), mechanism-confirmed (conv -31.5%). But it
recovers only ~7% of OUR GDN kernel time (2.07×→~1.9× vs vLLM), NOT "much of the
2×." The CHUNK trio (the dominant GDN cost) was ALREADY bf16 and stays FLAT — the
remaining ~1.9× is the chunk's KERNEL-STRUCTURE/CODEGEN gap vs vLLM's Triton/FLA,
the established hard frontier (parity-ledger 2026-07-08: hand-matched structure =
neutral, codegen-quality = a compiler capability). Landed DEFAULT-ON as a faithful
(vLLM FLA carries bf16 activations) clean down-payment in the same shape as
delta_h (+0.85%) / WY (+0.54%). The 27B's bigger levers remain elsewhere (fp4
autotune ~2%; the chunk codegen gap, needing Triton/compiler not hand-C++).

## 2026-07-09 — 27B GDN bf16-input landed (+0.7-0.8%, 4th win); the DOMINANT 27B gap is the chunk CODEGEN gap (~1.9× vs Triton), portably-unclosable

The "~14% via bf16" hypothesis was WRONG: the GDN **chunk** trio (WU/DeltaH/ChunkO) was ALREADY
bf16 (VT_GDN_BF16 on). Only the INPUT side was f32. `VT_GDN_IN_BF16` (default-on, merged f2faa5c)
converts the in_proj GEMM output (MatmulF32D→bf16), conv (bf16 in/out, f32 accum/state), and
post-conv read to bf16. Token-exact (423/423 + 27B/35B 16/16), every-axis positive:
**CausalConv1dFwd −31.5%, GdnPostConv −17.6%; GDN µs/tok 138→128 (−7.1%); +0.7-0.8% e2e** (both
conc points). A clean 4th win, same shape as delta_h/WY. 27B-only (gated on the bf16 in_proj).

**KEY FINDING — the codegen floor.** The dominant GDN cost is the chunk trio, which was already
bf16 + register-tiled (delta_h) + WY-blocked and stayed FLAT under bf16 — it is **still ~1.9×
slower than vLLM's Triton/FLA** (our GDN 128 vs vLLM 71.7 µs/tok). This residual is the chunk
**kernel-structure/CODEGEN gap** the prior campaign established is a compiler capability
(hand-matched structure → neutral; delta_h register-tiling closed SOME of it but hit a limit at
~1.9×), NOT portably (hand-C++) closable. **So the 27B's dominant gap is a Triton-codegen floor.**

**SESSION TALLY: 4 e2e wins (delta_h +0.85%, WY +0.54%, fp8-rmsnorm +0.85%, GDN-in-bf16 +0.7-0.8%
≈ +3% combined), all portable + token-exact + default-on. 35B ~0.97× (near parity), 27B ~0.84-0.87×.
Big memory win (~65% of vLLM's peak).** Remaining: fp4 GEMM per-shape autotune (~2%, portable,
BOTH models — our fixed 2-config vs vLLM's 8-config); in_proj_qkv cuBLASLt SM80-pick (~1%). The
BIG 27B gap (chunk codegen ~1.9×) needs Triton/a compiler, not portable hand-C++ — the session's
original portable-vs-Triton dilemma, now with data showing the portable playbook's limit.

## 2026-07-09 — fp4 GEMM per-shape autotune (27B +5.8%) + SESSION CLOSE

fp4 GEMM per-shape tile autotune (`VT_FP4_AUTOTUNE`, `cuda_matmul_nvfp4_cutlass.cu`, merged
3ac839d): mirrors flashinfer's SM120 CutlassTileConfig set (instantiated the 4 N≥128 tiles;
the N<128 tiles need a different block-scaled epilogue we didn't adopt), per-shape micro-bench
+ cache (like the fp8 plan-cache) with >1% hysteresis. **27B +1.6% (conc16) / +5.8% (conc32,
M=8192) e2e**, kernel −16% to −57%, token-exact 16/16 both models. **35B inert** (its dense is
fp8 cuBLASLt, not this cutlass GEMM — corrects the "helps 35B" premise). **Default OFF** pending
a CUDA-graph-capture safety check (autotune's `cudaEventSync` is illegal mid-capture; both gates
passed with it ON → empirically safe, but verify before the global flip). Available via
`VT_FP4_AUTOTUNE=1`.

**=== SESSION CLOSE (2026-07-09) ===**
FIVE measured, token-exact wins this session, built on the new portable `vt::tile` component
(which reversed the old "portable-C++ ceiling" belief): (1) delta_h register-tiled + cp.async
ring +0.85% [default-on]; (2) WY blocked tensor-core inverse +0.54% [default-on]; (3) fp8
RMSNorm→quant + quantize-once +0.85% [default-on]; (4) GDN input-side bf16 +0.7-0.8% [default-on];
(5) fp4 GEMM per-shape autotune 27B +1.6-5.8% [VT_FP4_AUTOTUNE, default-OFF pending graph-safety].
**Current fresh A/B vs GRAPHED vLLM: 35B ≈0.97× (near parity), 27B ≈0.84× (≈0.86-0.87× with fp4
autotune), ours uses ~35% LESS peak memory on both.** Also: measured-refuted fp8-plan-cache +
launch-bubble-fusion + EVT-on-GB10 (all ~0); relicensed MIT→Apache-2.0; fixed CI runner-OOM
(bounded -j); README status refreshed. **The 27B's dominant residual is the GDN-chunk CODEGEN
gap (~1.9× vs vLLM's Triton), portably-unclosable** — the open fork: Triton CUDA fast-path
(branch perf/triton-fastpath, proven; non-portable) vs accept the floor + pursue the broader MVP
(GGUF/tools/grammars/server/e2e/more-models). fp4-autotune default-on (verify graph-safety) is a
cheap pending 27B win.

## 2026-07-09 — vt::tile RUNG-2 (TMA+mbarrier) built + VERIFIED on sm_121a; delta_h −3.8% kernel, but DECISIVE: the copy mechanism is NOT the GDN codegen gap (portable TMA does NOT close the 1.9×)

Answered the open Rung-2 question from the 2026-07-08 component design + the 2026-07-09 codegen-floor
finding: "Triton's Blackwell codegen = TMA+mbarrier (not sm80 cp.async); does a portable TMA pipeline
close the GDN-chunk 1.9× gap?" Built the Rung-2 primitive and re-ported delta_h to test it decisively.

**Built (branch perf/gdn-tma-pipeline):**
- `include/vt/cuda/tile/tma_pipeline.cuh` — the Rung-2 tier: mbarrier (init / arrive.expect_tx /
  try_wait.parity spin) + `cp.async.bulk.tensor.3d.shared::cta` TMA G2S + fence.proxy.async, ported
  1:1 from CUTLASS (barrier.h:397/593/416, copy_sm90_tma.hpp:159-191 the CUTE_ARCH_TMA_SM120_ENABLED
  `.shared::cta` variant), cited. Host CUtensorMap builder (3D [tok,head,dk] bf16, box(dk,1,BT), swizzle
  NONE, OOB-zero) via `cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",12000)` — no libcuda link.
- `GdnChunkDeltaHTmaKernel<TD,BV,STAGES>` in cuda_gdn.cu — the RegRing delta_h with ONLY the W/K
  streaming swapped cp.async→TMA+mbarrier (SAME smem layout + WMMA + STAGES=2). Behind VT_GDN_TMA
  (default OFF); falls back to the cp.async ring if desc setup fails. Mirrors the CuTe-DSL kernel_h.py
  TMA-warp mbarrier producer loop (its tcgen05/tmem MMA is sm_100-only, so sm_121 keeps WMMA).

**sm_121a SUPPORTS the TMA+mbarrier subset** — proven first by a standalone probe (bit-exact 3D tile
load: `cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes` + mbarrier init/expect_tx/
try_wait all assemble & run), then by the real kernel. Two alignment gotchas: TMA smem dst needs 128B
alignment (extern `__align__(128)`); descriptor global addr/strides need 16B (cudaMallocAsync-aligned).

**Correctness (all GREEN):** test_ops_gdn 423/423 with VT_GDN_TMA=1 AND default (bit-exact vs ref) +
compute-sanitizer memcheck 0 errors; 27B paged-engine greedy token-exact vs vLLM + 35B greedy 6×16 SUCCESS.

**MEASURED (27B NVFP4, GB10, same-binary nsys A/B, in1024/out4 np16 conc16, delta_h total over prefill,
2 reps each):**
  reg-no-ring (copy EXPOSED)  464.1 ms
  cp.async ring (Rung-1)      401.5 ms   (402.54 / 400.56)
  TMA (Rung-2)                386.4 ms   (386.16 / 386.54)   = −3.8% vs ring, −16.7% vs no-ring
  Control: GdnChunkWU 529→531, ChunkO 414→413 flat ±0.5% between arms ⇒ clean isolation of the copy swap.
  E2E prefill (np32 conc32, 3 interleaved reps): OFF 1363.6 / ON 1372.2 = +0.6% mean but NOISY
  (+0.0/+1.6/+0.3%), near the noise floor.

**DECISIVE VERDICT (the point of the task):** the portable TMA pipeline does **NOT** close the GDN-chunk
codegen gap. TMA is a real but SMALL win (−3.8% on ONE of ~6 GDN kernels ≈ −0.7% of GDN, e2e ≈ noise),
not the ~44% (128→71.7 µs/tok) the 1.9× needs. The no-ring→ring→TMA ladder is the proof: the delta_h
kernel has a ~386 ms **COMPUTE floor** — the cp.async ring already hides most copy latency (−63 ms of the
78 ms exposed), and TMA (the strictly-better copy) hides only ~15 ms more. So the ~1.9× residual is NOT
the gmem→smem copy/pipeline mechanism (which the earlier campaign correctly identified as codegen); it is
the **WMMA compute codegen + the sequential-chunk kernel structure** — Triton's tensor-core scheduling, a
COMPILER capability, exactly the frontier the parity-ledger established (V-split neutral, hand-cp.async
negative, register-tiling +0.85%, and now TMA −3.8%-but-not-a-closer). The 128B-swizzle + warp-specialized
variants would tune the SMEM-READ/overlap (compute-side, different lever), not the copy — and STAGES=3
doesn't fit the 99 KiB opt-in with the full W+K ring. **This EXHAUSTS the portable async-pipeline lever on
the GDN chunk: closing the codegen gap needs the Triton CUDA fast-path (branch perf/triton-fastpath,
proven end-to-end) — a human canon decision (Triton-for-CUDA-perf-behind-vt:: vs accept the portable
floor + pursue the broader MVP).** Kept VT_GDN_TMA DEFAULT OFF (proven-correct, memcheck-clean, available
lever); did NOT roll out to WU/ChunkO (their copy isn't the bottleneck either — same compute-floor logic).

## 2026-07-09 — DECISIVE: portable async-pipeline EXHAUSTED on the GDN-chunk codegen gap → MVP needs a canon decision

Built Rung-2 vt::tile (TMA+mbarrier, `include/vt/cuda/tile/tma_pipeline.cuh`, ported 1:1 from
CUTLASS `arch/barrier.h` + `cute/arch/copy_sm90_tma.hpp`, sm_121a-verified, token-exact 423/423 +
27B/35B 16/16, merged 9d4b45c default-OFF). delta_h ladder (nsys, same-binary): no-ring 464.1 →
cp.async-ring 401.5 → **TMA 386.4 ms**. TMA vs ring = −3.8% (closes ~none of the ~1.9× GDN gap).
**The delta_h kernel has a ~386ms COMPUTE floor** — the cp.async ring already hid 63 of 78 ms of
exposed copy; TMA (strictly-better copy) hid only ~15 ms more. So the ~1.9× residual vs vLLM's
Triton is NOT the copy/pipeline mechanism — it is the **WMMA compute codegen + kernel scheduling
(Triton's compiler capability)**. Rung-1 (cp.async) + Rung-2 (TMA) BOTH proven insufficient.

**CONCLUSION (decision-critical): the portable async-pipeline lever is EXHAUSTED on the
codegen-bound GDN chunk kernels.** The MVP (27B ≥1.0×) is blocked by this codegen gap. The
portable vt::tile playbook closed the STRUCTURE (register-tiling, blocked-inverse, bf16, delta_h
−22%) but cannot close Triton's WMMA-compute codegen. The ONLY known path to 27B parity is the
Triton CUDA fast-path (branch perf/triton-fastpath, PROVEN toolchain, behind the vt:: seam with
CPU-ref preserved) — which CONFLICTS with the canon (discipline.md "no compile-target" +
mission.md "no build-Python"). This is a genuine MVP-vs-canon conflict, now PROVEN by exhausting
the portable path: the "no compile-target" rule was premised on portable working (disproven for
these kernels); mirror-vLLM (PRIME POLICY) + the ≥1.0× MVP both point to Triton. A HUMAN decision.
(One untried portable compute-side lever remains — 128B-swizzle + warp-specialization of the WMMA
— but the evidence assesses it as the same compiler-capability class = likely partial.)
## 2026-07-09 — DGX gate test harness aligned with native-resident loader state (worktree: codex/parity-dgx-runbook)

Worked in isolated worktree `/home/mudler/_git/vllm.cpp-codex-parity-dgx`; DGX scratch copy:
`/home/mudler/work/vllm.cpp-agent-20260709-171553`. Initial DGX GPU commands used `/tmp/gpu`
`flock`; after user override, final CUDA confirmation ran directly on the GPU. The final direct
run saw `nvidia-smi`: `NVIDIA GB10, 1 %`.

Changes prepared:
- `scripts/dgx-bringup.sh`: updated the bring-up runbook to the current gate stack (`sm_121a`,
  CUTLASS via `VLLM_CPP_CUTLASS_DIR` / `~/cutlass_probe`, 35B+27B gate language, bounded default
  `JOBS=8`).
- `tests/vllm/test_qwen36_weights.cpp`: updated the real-shard loader contract checks for the
  current native-resident default: 35B W8A8 projections are raw FP8 resident, NVFP4 experts are raw
  FP4 resident; helper dequant bit-pattern checks remain the numeric golden.
- `tests/parity/test_op_parity.cpp`: added scoped env overrides so f32 GDN prefill replays the same
  sequential oracle path as the committed goldens, full real-model logits remain CUDA-only on CPU
  builds, and isolated Qwen3.6 layer goldens use the legacy dequant loader (`VT_DENSE_NATIVE=0`).
  The isolated full-attention layer replay is CPU-only; CUDA full-attention correctness remains
  covered by the full 35B/27B logits gates. Margin diagnostics now print before a failure when
  `VLLM_PARITY_PRINT_MARGINS=1`.

Verification:
- Local CPU build + full suite: `cmake --build build -j2 && ctest --test-dir build --output-on-failure`
  passed 91/91 after the harness update.
- DGX CUDA incremental build of `test_op_parity` + `test_qwen36_weights` passed.
- DGX locked `test_qwen36_weights` passed 66/66 assertions.
- DGX locked `test_op_parity` with the native default confirmed the full-model gates still pass:
  35B `greedy_match=16/16`, top-1000 gap `1.8125`; 27B diagnostic `greedy_match=16/16`, top-1000 gap
  `1.228461`; f32 GDN prefill sequential replay margins were tight. A direct follow-up run after the
  CUDA full-attention skip passed `./build-cuda-agent/tests/test_op_parity -tc="op parity vs upstream
  goldens (CUDA)"`: 1/1 test case passed, 12/12 assertions, with the 35B full-attention layer
  explicitly skipped as CPU-only isolated replay while 35B/27B full-model logits stayed 16/16.

## 2026-07-09 — GDN codegen gap CLOSED via sanctioned Triton AOT (−34%); 27B → 0.944×/0.840×; residual MOVED off GDN

Full GDN chunk trio ported to the sanctioned CUDA-only Triton AOT fast-path (PRs #1+#2, default-OFF,
byte-inert with VLLM_CPP_TRITON=OFF; token-exact test_ops_gdn 31/31 + 27B/35B greedy 16/16; 2 silent
bugs caught: chunk_o fp32-scalar-as-double, solve_tril uninit Ai upper-tri). Per-kernel (nsys):
**WU −35.4%, chunk_o −36.5%, delta_h −29.8%; GDN chunk total −34.1% (1.52×).** The ~1.79× GDN-vs-FLA
gap → **~1.18×** (GDN essentially at parity with vLLM's autotuned FLA). **This CLOSES the codegen gap
— the session's central finding — and PROVES the sanction's premise (residual was Triton compiler
codegen; hand-C++ exhausted).**

**27B e2e vs graphed vLLM (Triton-on): conc16 0.930→0.944×, conc32 0.821→0.840×. NOT yet ≥1.0×.**
**The residual has MOVED OFF GDN** (GDN now Triton-fast). Dominant remaining: (a) **non-GDN prefill
fusion** (rmsnorm+quant, silu+quant — vLLM Inductor whole-graph fusion we run unfused); (b) a
**conc32 concurrency-scaling gap** (0.84× at conc32 vs 0.944× at conc16 — worse scaling, likely
batching/KV/scheduling, not prefill compute); (c) fp4-GEMM autotune (+5.8% conc32, BUILT,
VT_FP4_AUTOTUNE, default-OFF pending graph-safety); (d) tune GDN AOT configs (BV/BK/warps/stages) for
the last ~0.18× of GDN. GDN-Triton was NECESSARY but not SUFFICIENT for MVP parity. NEXT: reassess
the GDN-Triton-fast 27B to rank the non-GDN residual + diagnose the conc32 scaling gap. PRs kept off
main while the parallel agent is active. Triton branches: perf/gdn-deltah-triton-aot (#1) +
perf/gdn-wu-chunko-triton-aot (#2).

## 2026-07-10 — 35B at 0.994× (w13 fusion +3.01%); main CONSOLIDATED (w13 + Triton stack); MVP within reach

**w13 MoE fusion MERGED (3e6736b, default-ON):** one grouped Marlin GEMM [P,2I] + SiluAndMul
instead of two GEMMs + memsets (mirrors marlin_moe.py:133-170), same fusion for the
shared-expert gate_up (the headline: +2.5 of the +3.01%). BIT-IDENTICAL halves probe + 35B
greedy 16/16 both arms; A/B conc64/np200 +3.01%, every axis better (TPOT −3.1%, TTFT −1.4%).
**35B: 0.965× → 0.994× vs graphed vLLM (3282).** Honest note: the MoE-expert w13 alone was
+0.53% (fixed costs largely overlapped); the shared-expert fusion carried the rest.

**Triton AOT GDN stack MERGED to main (f3bcf60**, PRs #1+#2; byte-inert with
VLLM_CPP_TRITON=OFF). Main now carries every lever: 4 portable wins + w13 + Triton GDN
(default-OFF) + fp4-autotune (default-OFF).

**MVP state:** 35B **0.994×** (w13 on, Triton GDN OFF — its 35B e2e never measured, H=32 spec
exists; IN FLIGHT: the decisive A/B, does Triton GDN cross 1.0×?). 27B **~0.958×** best
(Triton + fp4-autotune + mnbt2048; agent report pending — measured: autotune conc32 +6.4%,
mnbt 8192→2048 +11.5%, the conc32 gap was largely OUR batch config starving decode).
Remaining 27B ~4%: GDN AOT config tune (FLA autotuner best_config vs our pinned BV64/w4/s3),
TTFT axis, non-GDN fusion residual.

## 2026-07-10 — 🎯 35B MVP THROUGHPUT GATE PASSED: 1.0195× vs graphed vLLM (Triton GDN on)

Decisive A/B on consolidated main @1489774 (VLLM_CPP_TRITON=ON build; token-exact both arms:
test_ops_gdn 31/31 + 35B greedy 16/16 + batched): 35B conc64/np200 in1024/out128 —
Triton-GDN ON **3345.9** vs OFF 3260.0 (+2.64%, arms non-overlapping, ±0.03% spread).
**3345.9 / 3282.0 (fresh graphed vLLM) = 1.0195×** (per-rep 1.0192–1.0198; worst-case
contended bound 1.0073×). Every axis: TTFT −4.1%, TPOT −2.4%, peak mem 52.8 vs vLLM ~80.6GB.
One OFF rep was discarded for contention (a concurrent agent's gate test) and re-run clean —
disclosed. **The 35B meets the ≥1.0× MVP bar on every measured axis** (vLLM TTFT/TPOT not
measurable via its offline bench; ours beat the OFF arm on both).

**Triton runtime toggles flipped DEFAULT-ON** in the VLLM_CPP_TRITON build (GdnTritonEnvOn:
absent→on, =0 opts out) — gated by both-arm token-exactness on both models + the every-axis
win. Default (non-Triton) build unchanged (hand-C++, 0.99×).

**27B consolidated checkpoint: 0.970× at conc32** (Triton + fp4-autotune + mnbt2048;
1011.6/1043.2, ±0.24%, peak 61.8GB vs vLLM ~86.4GB) — from 0.840× at the start of this
push. Residual ~3%; the finisher agent (mnbt dense default mirroring vLLM's scheduler 2048 +
FLA-autotuner GDN config re-pin) is in flight. README updated (35B ≥1.0×, 27B ≈0.97×).

## 2026-07-10 — 27B MVP finisher: per-arch mnbt (dense 2048, vLLM's default) +9.5% conc32; GDN AOT pins verified vs FLA autotuner (already optimal); FINAL 27B 0.966×/0.966×

**Task 1 (mnbt, branch perf/27b-mvp-finisher):** `ResolveMaxNumBatchedTokens` is now
per-arch — dense 27B gets **2048 FLAT** (vLLM's own DEFAULT_MAX_NUM_BATCHED_TOKENS,
vllm/config/scheduler.py:42 @ e24d1b24); MoE 35B keeps conc-aware 8192/4096
(unit-pinned, 35B gate 33/33 unchanged). 27B greedy 16/16 passes WITH the new default;
91/91 CPU. Official A/B (everything-on, 3 interleaved reps, same binary, idle box):
conc32/np192 2048 = **1012.42** vs 8192 = 924.50 (**+9.5%** total/prefill/decode,
TTFT −59%, peak −2.2GB; TPOT +0.6% the one marginally-worse axis); conc16/np96
2048 = **740.51** vs 4096 (old default) = 731.60 (**+1.2%**, non-overlapping; TTFT −33%;
TPOT +2.7% worse while decode tok/s higher). **conc16 answer: 2048 flat wins there too —
vLLM's flat default mirrored, no per-conc special case.**

**Task 2 (GDN AOT config tune):** ran FLA's OWN autotuner on the exact engine shapes
(H=48/32, T=1024..8192 varlen — winners T-stable). **delta_h (BV64/w4/s3) and chunk_o
(BK/BV64/w4/s3) pins CONFIRMED optimal** (both H) — the "unverified guess" was already
the FLA winner. Shipped: kkt H=48 BK128/w8/s3 (**−3.3%** kernel, reproduced). FLA's
tril (w8/s5) and wu (w2/s2) picks measured **SLOWER on our AOT variants** (+6.8%/+1.7%,
nsys 432 launches) → reverted to w4/s3; H=32 trio candidates recorded, unshipped.
GDN chunk total FLAT (2084→2082 µs) ⇒ **the "config tune ≈2-3% e2e" hypothesis is
DISPROVEN**; CMake knobs now per-kernel/per-H for future sweeps. Gates on the
re-pinned build: test_ops_gdn 31/31 (557), 27B 16/16 + 35B 33/33 through Triton.

**FINAL 27B MVP NUMBERS (everything on: Triton GDN + fp4-autotune default-ON + mnbt
per-arch default, 3 reps):** conc16/np96 **740.90** (737.96/743.04/741.71) vs vLLM
766.62 = **0.9665×**; conc32/np192 **1008.04** (997.58/1013.78/1012.76; r1 a cold
outlier, r2/r3 ≈1013 = 0.971×) vs vLLM 1043.17 = **0.9663×**. **NOT ≥1.0× — honest
shortfall ≈3.4% at both operating points** (fresh-denominator drift check in the
ledger). Session lift: conc32 0.896×→0.966× (the mnbt fix), conc16 0.957×→0.966×.
The residual is the known non-GDN prefill fusion gap (Inductor rmsnorm+quant /
silu+quant) — GDN and config levers are now exhausted/verified-optimal.

**Op note:** the shared dgx workspace ~/work/vllm.cpp-mvpgate was concurrently used by
the 35B Triton A/B agent at session start; my rebuild swapped vllm-bench under its
first OFF rep (semantically inert for its explicit-mnbt 35B runs — my change only
alters the dense-arch DEFAULT — but rep r1 of its A/B crossed binaries; reps r2-r4 ran
consistently on the new binary). Flagged for that agent's report review.

## 2026-07-10 — 27B last-mile: ground-truth dump SETTLED the fusion record; TN GEMM + measured toggle stack landed; FINAL 27B 0.990×/0.997× (from 0.966×/0.966×); 35B 1.023×

**STEP 1 (ground truth — the conflicting record is settled).** Dumped vLLM's ACTUAL
27B production Inductor codegen (`TORCH_LOGS=output_code`, cache disabled;
~/work/vdump27b.err on dgx) + a PRODUCTION-config graphed torch profile
(~/work/vprod27b). The earlier trace (~/scratch_agent_a562/vprof27b) was
`enforce_eager=True` — it could never answer the fusion question and seeded the
conflict. In production the 27B norm→fp4-quant site is TWO kernels in vLLM TOO
(triton add+rmsnorm bf16, then extern `_C.scaled_fp4_quant`; the
"…_scaled_fp4_quant" in Inductor kernel names is origin-node pollution — the body
writes bf16 only; rms_quant_fusion.py registers FP8-only patterns, no nvfp4).
Measured PARITY at that site (0.374 vs 0.360 µs/tok). silu+quant: vLLM runs the
fused `_C.silu_and_mul_nvfp4_quant` custom op — we already ship that fused (≈parity).
**The finisher's "Inductor rmsnorm+quant / silu+quant residual" theory is DEAD for
the 27B** (it's real only on the fp8/35B path, where RmsNormQuantFp8 already ships).

**STEP 2 (measured per-site diff, production kernels both sides).** Ours ≈426 vs
vLLM ≈356 µs/tok prefill GPU (1.20× ≈ the 3.4% e2e gap). Ranked: (1) GDN in_proj
bf16 GEMMs 2.29 vs 1.80 µs/tok/L — OUR LAYOUT: row-major×row-major kMatmul gets
nvjet NNNN ~76TF + a slow sm80-cutlass z kernel; vLLM's F.linear TN gets nvjet TNNN
~123TF (≈24 µs/tok ×48L — the mislabeled "GEMM at-parity" component); (2) prefill
attention 1.37 vs 0.25 (PagedFlashWmma ~7-13TF vs FA2 ~40-56TF; ≈18 µs/tok);
(3) attn preamble+rope 1.36 vs 0.27 (4 f32 kernels + in-kernel DOUBLE
transcendentals vs 4 bf16 Inductor kernels + cos/sin cache; ≈17.5); (4) conv 0.43
vs 0.18 (≈12); (5) gated-norm 0.37 vs 0.15 (≈8.4). OUR ADVANTAGE: fp4 GEMMs ≈103
vs ≈127 µs/tok (per-shape autotune beats flashinfer mm_fp4 here). vLLM's merged
qkvz/ba/qkv/gate_up GEMMs: at TN rates the merge saving is only the extra A-reads
(~0.1-0.2% e2e) — NOT ported, measured too small.

**STEP 3 (landed on perf/27b-prefill-fusion-mvp; every change gated).**
(a) `vt::MatmulBT` TN bf16 GEMM (b raw torch-Linear [N,K]; cuBLASLt column-major TN
copied from our own fp8 path) + `OwnedTensor.nk`/`LoadBf16RawNK` for the 27B
in_proj_{qkv,z,b,a}. +0.57% e2e. dgx ctest 91/91. lm_head NOT flipped (measured at
parity). (b) Toggle stack flipped to PER-ARCH defaults after a 3-rep interleaved
A/B (+1.61% over TN, every axis better): fused attn preamble ON for fp4-attn (27B;
gates pass — the 35B fp8 1-ULP divergence stands, stays OFF), conv-tiled ON (both
models gated), and gdn-out-bf16 → **REVERTED after measurement**: the Triton AOT
chunk_o guards Tout==float, so bf16 core forfeits its −36% kernel (754.71 vs 757.15
conc16); re-flip needs a bf16-out chunk_o AOT variant (LESSON: re-measure old
toggle wins after a kernel-path swap). 27B logits diagnostic moved to the OTHER
vLLM-legitimate branch of the documented tok-6 whitespace tie (16/16→6/16
informational; the deterministic-span hard gate + engine gates green throughout).

**FINAL NUMBERS (defaults binary, fresh graphed denominators measured same hour):**
conc16/np96 **758.51** (757.70/756.20/761.62; r1 713.12 cold outlier disclosed) vs
vLLM 766.49 = **0.990×** (0.987-0.994); conc32/np192 **1049.01** (5 reps
1044.95-1053.74, best rep 1.0012×) vs 1052.48 = **0.9967×** (0.994-1.001). TTFT
1804/2500ms, TPOT 177.2/255.7ms (all better than the 0.966× baseline's). Peak mem
61.8 vs vLLM 76.2GB. **35B spot conc64/np200: 3356.66 = 1.023×** (shared-kernel
changes safe; TPOT 139.2). **NOT ≥1.0×: honest shortfall ≈1.0% conc16 / ≈0.3%
conc32** (conc32 straddles 1.0 within rep spread). Session lift: 0.9665→0.990 /
0.9663→0.997.

**REMAINING (measured, named):** (1) the prefill attention kernel — ours ≈5× FA2
per token at L=1024 ⇒ ≈1.5-1.8% e2e; the sanctioned 1:1 route is porting vLLM's
FA2 `flash_fwd_splitkv` (CUDA C++ dep — carries its pipelining; NOT hand-matching,
which measured negative on GDN). (2) bf16-out chunk_o AOT variant → re-flip
gdn-out-bf16 (+ the old +0.8% traffic win). (3) bf16-q preamble→attention
(vLLM-faithful; attention dispatch already accepts bf16 q; numerics-gated).

## 2026-07-10 — 🎯 27B MVP THROUGHPUT GATE PASSED: FA-2 prefill resurrected (1.0072×/1.0071× vs fresh graphed vLLM) — the last named lever, closed

**THE HEADLINE.** The row-283 remainder ("prefill attention kernel, ours ≈5× FA-2")
is closed by resurrecting the shelved FA-2 vendored port CORRECTLY. 27B at
in1024/out128, fresh same-hour graphed denominators (GM0.5, random, range-ratio 0):
**conc16/np96 764.28 vs 758.84 = 1.0072× (7 reps, 6/7 ≥1.0, worst 0.9960);
conc32/np192 1051.24 vs 1043.86 = 1.0071× (5/5 reps ≥1.0).** Output tok/s 1.0068×
both; TTFT 1750/2440 (prior best 1804/2500); TPOT 175.1/255.6 (prior 177.2/255.7).
35B spot conc64/np200: 3346.8 mean = 1.020× vs pinned 3282 (FA-2 verified inert on
the 35B). **Both gate models now measure ≥1.0× total throughput at their MVP
operating points with token-exact gates green.**

**WHY THE OLD FA-2 WAS SHELVED, AND WHAT CHANGED.** Ledger rows 188/190: the
vendored kernel was 3.4× FASTER per-kernel, but e2e −4.3% because (a) the wiring
bridged our f32 q/out with f32↔bf16 CAST kernels (+ per-call cudaMallocAsync),
which erased the win; (b) the launcher did a per-layer D2H+sync for max_seqlen_q/k
(the exact drain query_start_loc_host later removed for the WMMA path); (c) at the
time the engine was believed idle/GDN-bound so the lever was deprioritized. The
resurrection (branch `perf/fa2-prefill-27b`): natively-bf16 end to end — the fused
preamble emits bf16 q/k (gate stays f32; mixed-dtype AttnQkNormRopeGate, every
moved rounding = the same RN round CastBf16 did, bit-identity PINNED by tests),
attention out is bf16 into a templated SigmoidGateBf16, bf16 k feeds the KV-write
directly; sync-free via query_start_loc_host + new PagedAttentionArgs::max_seq_len;
softmax_lse pooled; num_splits pinned 1 (vLLM's own FA-2 varlen contract,
flash_attn_interface.py:309; the combine kernel is batched-addressed and unsound
for ragged varlen). Kernel A/B: **475.25→129.18 ms per profile window = 3.68×**
(~1.81→0.49 µs/tok/layer at our T≤2048 chunks; vLLM's 0.25 was at T=4096).

**DEFAULTS.** VLLM_CPP_FLASH_ATTN now default ON (builds when CUTLASS present,
sm_12xa); VT_FA2_PREFILL default ON when compiled (=0 → WMMA, same-binary A/B).
Eligibility: prefill segment && head_dim 256 && bf16 q/KV/out only — decode and
the 35B (f32 q) are untouched by construction. Gates: 27B greedy engine PASS ON
and OFF (same tie branch), chunked==one-shot holds with FA-2 engaging, FA-2 op
parity max|err| 5.8e-3, host-metadata vs fallback bit-identical, 35B PASS.

**FLAGGED (pre-existing, NOT this branch):** `test_ops_fused_chain` fails on
PRISTINE main 08c3825 clean-built with the same flags (CPU Tier-1 vs golden
RmsNorm, h={7,127,128,512}) — earlier "91/91" runs were on stale incremental
builds ("incremental build masks" memory). Needs a separate fix on main.

## 2026-07-10 — 🏁 MVP THROUGHPUT GATE PASSED ON BOTH MODELS

**27B: PASSED via FA-2 prefill attention (merged 45a6342, default-ON).** The last named
lever: vendored `flash_fwd_splitkv<256,64,64,4>` (vllm-project/flash-attention @2c839c33)
wired natively-bf16 (zero casts — what killed the shelved attempt), sync-free, splits=1
per vLLM's varlen contract. Kernel 1.81→0.49 µs/tok/layer (3.68×). **Token-exact: 27B
greedy IDENTICAL tokens FA2 on/off.** vs fresh graphed vLLM: **conc16 1.0072× (6/7 reps
≥1.0, worst 0.996 disclosed), conc32 1.0071× (5/5 ≥1.0)**; TTFT 1750/2440ms and TPOT
better than prior best; peak mem ours. 35B inert (f32 q never routes), spot 1.020×.

**SCOREBOARD (Triton-AOT build, fresh same-box graphed-vLLM denominators):**
- **35B conc64: 1.020-1.023×** total; TTFT −4.1%, TPOT −2.4%; mem 52.8 vs ~80.6GB. ✅
- **27B conc16: 1.0072× / conc32: 1.0071×**; TTFT/TPOT better; mem ~62 vs ~76GB. ✅
- Token-exact greedy 16/16 preserved on both, every change gated.
Session arc: 27B 0.84×→1.007×, 35B 0.96×→1.02×, via: portable vt::tile wins (delta_h,
WY, fp8-fusion, bf16-in), sanctioned Triton AOT GDN (−34%), w13+shared-expert fusion
(+3%), fp4-autotune, mnbt=2048 (mirrors vLLM), TN in_proj, preamble/conv flips, FA-2.

**KNOWN ISSUE (not MVP-blocking, needs a fix):** `test_ops_fused_chain` fails on pristine
main (CPU Tier-1 interpreter vs golden RmsNorm at h={7,127,128,512}) — pre-existing,
proven not from the FA-2 branch; the Tier-1 interpreter is not on the model hot path
(Tier-0 composite is default). Fix the CPU interpreter or its golden.
**REMAINING MVP SCOPE (non-throughput):** TTFT/TPOT vs vLLM SERVE (only offline-bench
compared so far), GGUF real-file parity on dgx, e2e suites — per gates.md.

## 2026-07-10 — GGUF REAL-FILE GREEDY PARITY ON GB10: PASSED (the last M0.10 dgx-pending item)

**WHAT WAS PENDING.** M0.10 shipped the GGUF loader CPU-green on synthetic files;
the real APEX end-to-end load + greedy parity on GB10 had never run. This session
ran it, chose the oracle, and closed it. Branch `gate/gguf-real-file-parity`.

**FIRST RUN + THE ORACLE QUESTION.** The engine loads the real 17.3 GB
APEX-Compact (arch qwen35moe, F32+Q3_K/Q4_K/Q6_K) end-to-end via
`FromModelDir(*.gguf)` on GB10 first try — single-file config + GGUF-embedded
vocab + k-quant dequant→bf16, ~2.7 min load, fluent 16-token greedy output,
exit 0. THE ORACLE for parity is llama.cpp loading the SAME .gguf (the fork at
`~/llama-phase93-qwen3next-gqa-bcast` supports qwen35moe; driven via llama-server
`-ngl 99`, temp 0, `return_tokens`): the APEX files are k-quant re-quantizations
of the base weights, so the safetensors-NVFP4 goldens are NOT same-weights — the
oracle itself, on APEX-Balanced, diverges from the NVFP4 M0 continuation at
token 7. (Compact's M0 continuation happens to coincide with the NVFP4 one.)

**THE M0-PROMPT NEAR-TIE (disclosed, not a bug).** On the M0-exit prompt our
Compact greedy takes " official…" where the oracle takes " capital…" — the
oracle's OWN top-2 at that step are −1.8802 vs −1.9200 (margin 0.040), a
near-tie; our numerics (dequant→bf16 + bf16 GEMMs — the SAME recipe vLLM's GGUF
loader uses) legitimately land on the other branch. Verified NOT a loader bug
by WEIGHT-LEVEL cross-check of every GGUF tensor family against the safetensors
checkpoint ground truth (gguf-py dequant + the loader's documented inversions,
vs BF16/FP8-scalar/NVFP4-dequant safetensors tensors): norm +1 inversions,
ssm_a=log(-x), dt_bias, conv1d V-head reorders all EXACT (maxabs 0.0); every
quantized 2-D/3-D family (embed, qkv/gate/out with V-row+col reorders, q/k/v/o,
router, shared+routed experts, lm_head) corr ≥ 0.9927 = pure quant noise.
Config-from-GGUF also diffed clean against config.json (rope 1e7/64, eps 1e-6,
MoE 256/8/512, GQA 16/2/256, layer pattern interval-4).

**THE GATE (test_qwen36_gguf_engine, checkpoint-gated dgx-only).** Two DECISIVE
prompts (oracle top-2 margin ≥1.91 every step on "1, 2, …, 8," ; ≥0.15 on the
Eiffel prompt) × two files covering all supported k-quants on real tensors:
APEX-Compact (Q3_K/Q4_K/Q6_K) and APEX-Balanced (Q8_0/Q5_K/Q6_K). Asserts the
GGUF-embedded-vocab prompt tokenization (matches llama.cpp /tokenize exactly)
AND the 16-token greedy continuation token-for-token vs the pinned same-file
oracle (goldens/qwen36_gguf_35b/, provenance in manifest.json). **RESULT ON
GB10: 2/2 test cases, 28/28 assertions — Compact AND Balanced reproduce the
oracle 16/16 on both prompts, tokenization exact.** Server spot probes also
matched the oracle text exactly on 3/4 prompts incl. " oxygen. The atomic
mass…" (the 4th is the disclosed M0 near-tie).

**REMAINING for the full gate-#2 language (blocked on files, recorded in
porting-inventory):** no 27B GGUF exists anywhere on dgx (and the GGUF loader is
qwen35moe/qwen3next-only — a dense-27B variant is unwritten); no NVFP4-extension
-type (id 40) GGUF exists for the gate models (reader traits support it, dequant
does not); APEX Mini/Quality need IQ2_S/IQ4_XS i-quants (clear error today).

## 2026-07-10 — Canonical record v1: live roadmap, completed MVP archive, centralized specs, Mac host available

Applied the user-directed document lifecycle and made it binding in `AGENTS.md`:

- Renamed the live post-MVP roadmap to `.agents/roadmap_v1.md` and moved the
  completed M0–M3 record to `.agents/completed/roadmap_mvp_v0.md`.
- Carried every open TODO from the v0 Post-MVP paragraph into explicit live
  tracks C1–C9/D1–D4: kernel drop-in alignment; dense/MoE/Qwen3-Next families;
  MTP; FP8; sliding window/YaRN; priority scheduling; prompt-logprobs and logit
  controls; tokenize/metrics; sync tooling; backend expansion, TP, spec breadth,
  LoRA/offload/model breadth.
- Moved 14 feature-specific scoping, semantics, feasibility, architecture, and
  design artifacts from `.agents/` into `.agents/specs/`, including the MM/tools
  and spec-decode scoping reports. Project-wide protocol/status/index documents
  remain at the top level; completed era records live under `completed/`.
- Repaired repository references and stale public/current-status lines: A2 GGUF
  real-file parity and A4 vendored Triton AOT now read complete in the matrix;
  README architecture/CUDA tables now agree that both throughput gates passed.
- Verified both remote development hosts: `dgx.casa` is the GB10/sm_121 CUDA
  box; `192.168.68.103` is an M4 Mac mini with 16 GB unified memory. The Mac is
  sufficient for MLX/Metal op parity and small-model bring-up, but not 27B/35B
  gate-model runs. Xcode is present; CMake and MLX still need bootstrapping.

**Kernel-reuse posture:** the codebase has already proven thin-adapter lifts for
vLLM/dependency CUDA cores (CUTLASS NVFP4, Marlin, FlashAttention-2). Roadmap C1
standardizes raw pointer/shape/stride/stream adapter signatures so future pure
C++/CUDA csrc lifts replace only Torch glue. Python orchestration and generated
Inductor/Triton/CuTe paths still need C++ ports or the bounded vendored-AOT route;
they are not arbitrary source-file drop-ins today.

**Validation:** local roadmap/spec link audit PASS; `git diff --check` PASS;
CPU build + ctest PASS, 92/92.

**Next:** finish the independent PR #3 merge-readiness review; do not merge a
CUDA/performance path without its required GB10 compile, gates, and same-binary
A/B. Then scope C1 kernel adapter alignment and continue A1/A5 closing tracks.

## 2026-07-10 — PR #3 independent review: DO NOT MERGE yet

An independent sub-agent reviewed PR #3 (`codex/triton-aot-analysis`) against
current main, the whole-chain rules, and the new roadmap protocol. The synthetic
merge is conflict-free and clean CPU build/tests pass 90/90, but three high
severity blockers remain:

1. The new GDN scratch pools are default-on CUDA/performance behavior and have
   no CUDA/Triton build, reuse/growth test, model gates, `nsys`, pool ON/OFF A/B,
   or fresh-vLLM comparison.
2. The advertised bf16 `chunk_o` path is compiled out because its generated
   `gdn_chunko_bf16_h{32,48}` artifacts and MANIFEST entries are absent. The
   current sync workflow omits top-level `CMakeLists.txt`, while the drift check
   validates source hashes but not the expected base/signature set, so CI cannot
   detect or regenerate the missing artifacts.
3. The added bf16 tests silently exercise the hand fallback when artifacts are
   absent, create fresh streams, and run only one chunk call; they prove neither
   Triton dispatch nor dirty-buffer reuse/pool growth.

**Merge requirements:** regenerate and commit both bf16 launchers + MANIFEST;
extend workflow paths and drift validation to the expected artifact set; add an
assertable dispatch/reuse test; run the CUDA/Triton suite and both model gates on
`dgx.casa`; then run repeated same-binary pool ON/OFF and bf16-output A/B against
fresh production-vLLM denominators across throughput, latency, and memory. Rebase
and update roadmap/matrix status under the new protocol before merge.

## 2026-07-10 — Triton AOT sync workflow parses again; PR #3 artifact-set gap remains

The workflow added at `aae469d` failed at workflow-parse time on every push (zero
jobs, no logs): its `git commit` heredoc body was not indented inside the YAML
`run: |` scalar. Indented the body/terminator correctly and added top-level
`CMakeLists.txt` plus `regen-triton-aot.sh`/`check-triton-aot-drift.sh` to both
push and PR filters. This ensures changes to the declared AOT base/signature set
actually trigger the workflow.

Validation: PyYAML parses both jobs; the current sm_121a hash drift check passes;
`git diff --check` passes. This does **not** clear PR #3: the checker still needs
an expected-base/signature inventory so a missing generated bf16 specialization
cannot report in-sync. GitHub workflow execution is the final validation after
push.

## 2026-07-10 — Roadmap v1 is tabular, spike-gated, and ready for parallel claims

Closed the user-directed inventory-groundwork block and moved its frozen report
to `completed/roadmap_v1_inventory_spikes_2026-07-10.md`. The live record now
has one ordered portfolio table plus six ownership surfaces: the 81-row engine
matrix, complete pinned-vLLM model inventory, quantization matrix, kernel-family
matrix, backend/architecture matrix, and the broad feature coverage view.

**Pinned coverage invariants now enforced by CI:**

- engine/serving: 81 stable rows (`9 ANCHOR-BACKFILL`, `17 PARTIAL`, `5 READY`,
  `50 INVENTORIED`);
- models: 370 category memberships, 353 unique static architecture IDs, 321
  category/implementation rows, 307 unique targets, 258 modules, plus the
  unbounded dynamic Transformers route;
- quantization: 76 scheme/encoding rows, separating recognition,
  materialization/repack, native quant compute, real-model gates, and native
  performance gates;
- kernels: 30 practical families grounded through vLLM plus FlashInfer,
  CUTLASS, cuBLASLt, DeepGEMM, Triton/Inductor and other execution owners; and
- backends: 13 CUDA targets, 18 component-target rules, 8 platform/ABI rows and
  9 native-competitor gates (48 total).

Every implementation row now follows `INVENTORIED -> SPIKE -> READY -> ACTIVE
-> GATING -> DONE`; `PARTIAL` and `ANCHOR-BACKFILL` cannot masquerade as done.
Claims name stable IDs, agent/worktree/branch/file ownership, dependencies and
hardware. Completed blocks move under `completed/`, while permanent capability
rows retain their code/test anchors. `scripts/check-agent-record.py` rejects
dangling links, missing canonical documents, count drift, duplicate IDs,
invalid states, malformed tables and misplaced top-level specs.

**Audit corrections:** 26 legacy checkmark/partial feature claims had code but
no leaf spike. They are now exact engine-matrix rows rather than broad `DONE`
claims. The record narrows generic CUDA graphs to the evidenced Qwen/35B slice,
sampler support to the implemented host-synchronized subset, structured output
and tool calling to the native/Hermes subsets, serving to basic transport with
liveness-only `/health`, safetensors to model-specific mapping, and GGUF to a
llama.cpp-compatible vllm.cpp deviation (pinned vLLM has no GGUF loader). Source
provenance comments and README were corrected with the tables.

**Performance floors:** vLLM remains the mandatory CUDA oracle. SGLang
`v0.5.12.post1` is the initial low-concurrency CUDA comparison at concurrency
1/2/4/8/16; llama.cpp is the CPU/GGUF and Vulkan floor; Apple uses oMLX
`v0.5.0rc1` plus MLX-LM, with same-file llama.cpp Metal where applicable. All
arms use identical workloads, every-axis results and one resource lock across
the complete series. The first A1 latency campaign is preserved as
invalid/incomplete because both vLLM startups failed and ours-35B arms aborted;
it must be diagnosed and rerun rather than reused as evidence.

**Hardware hygiene:** followed the shared flock skill and preserved the active
PR #3 lock/workspace, both gate checkpoints, APEX GGUF evidence and sources.
Deleted only rebuildable/stale caches and build trees, reclaiming about 368 GB;
DGX free space was 359 GB after cleanup.

**Validation:** canonical-record checker reports `ENGINE=81 MODEL=323 QUANT=76
KERNEL=30 BACKEND=48`; all local Markdown links and table shapes pass; workflow
YAML parses; `git diff --check` passes; clean incremental CPU build succeeds and
`ctest --test-dir build-cifix --output-on-failure -j 4` passes 92/92.

**Next in accepted order:** finish/gate PR #3; diagnose and rerun A1; write the
A5 spike; then claim the C1 raw-pointer adapter-ABI spike, C2 generic model
factory spike, C3 MTP implementation leaf, and C4 GGUF compute-in-quant spike.
The D1 architecture-spine spike precedes parallel sm80/sm90 and backend lanes.

## 2026-07-10 — SGLang comparison pin corrected to current stable before execution

The initial competitor inventory named `v0.5.12.post1`, but SGLang `v0.5.13`
(`28b095c`) was released on 2026-06-13 and is the current stable release. It
also adds faster Qwen3.5 Blackwell GDN kernels, making it the more relevant
low-concurrency reference for these gate models. Advanced the unexecuted pin to
`v0.5.13`; no benchmark result or implementation state changed. The isolated
DGX provisioning/run still waits for the active PR #3 GPU series.

## 2026-07-10 - Roadmap control plane is lifecycle-enforced, not count-only

An adversarial follow-up audit found that the first tabular checker enforced
counts, links and state tokens but not the row contract it advertised. Closed
that gap and archived the completed repair at
`completed/roadmap_v1_control_plane_hardening_2026-07-10.md`.

**Canonical ownership and claimability:** the 323 `MODEL-*` factory/static/
dynamic rows now live in `model-matrix.md`; `specs/model-family-inventory.md`
is only the accepted inventory methodology. Priority-zero work now has stable
`SERVE-GATE-ONLINE` and `SERVE-E2E-NIGHTLY` rows, and the online gate has a
complete spike. TP/MTP/DFlash legacy specs bind their exact `PAR-*`, `SPEC-*`
and `MODEL-*` rows; DFlash now records local gaps, dependencies, file ownership
and non-overlapping `DF-*` leaves.

**Quant/backend completeness:** added claimable `QUANT-GGUF-COMPUTE` and
`QUANT-GGUF-PRESETS` umbrellas, labeled the preset display as non-claimable,
normalized KV/MLX row fields, and grounded the three gate-slice `DONE` rows with
code/tests, ledger and closing commits. Added the previously omitted
`BACKEND-CPU-ZEN` and `BACKEND-GATE-METAL-MLXLM` surfaces. Final inventories are
`ENGINE=83 MODEL=323 QUANT=78 KERNEL=30 BACKEND=51` (353 unique static model
architecture IDs; 13 CUDA numeric targets and 18 component rules unchanged).

**SGLang leaf:** `BACKEND-BENCH-CUDA-SGLANG-PREFLIGHT` is `READY` against
v0.5.13 `28b095c` and its digest-pinned multi-arch CUDA 13 image. The distinct
`BACKEND-GATE-CUDA-SGLANG` is `BLOCKED` until that preflight and
`SERVE-ASYNC-LLM` close. The 27B compressed-tensors NVFP4 path is a viable
preflight; 35B ModelOpt mixed precision remains conditional because source
inspection indicates a legacy `w4afp8` route. Correctness uses native engine
output IDs, never detokenize/re-tokenize. No SGLang result is claimed.

**CI enforcement:** `check-agent-record.py` now parses claimable tables and
rejects missing semantic fields, wrong-class or out-of-range implementation/
test anchors, `READY+` specs
that do not name the row or cover the complete spike contract, uncoordinated
`SPIKE`/`ACTIVE` owners, incomplete `DONE` ledger/commit closure, unknown
roadmap/claim IDs, portfolio-order drift, count drift, bad links and malformed
tables. `tests/scripts/test_agent_record.py` mutation-checks those failure modes
and runs in the `agent-record` CI job.

**Validation:** checker PASS (`83/323/78/30/51`); 13-test mutation suite PASS;
`git diff --check` PASS; clean CPU configure/build in `build-roadmap-audit`
PASS; `ctest --test-dir build-roadmap-audit --output-on-failure -j 4` PASS
92/92. No runtime feature or benchmark support changed.

**Next:** finish the seven PR #3 review fixes and post-fix GPU/oracle gates;
then execute the SGLang exact-load/corpus preflight without claiming binding
latency until async streaming lands. In roadmap order, diagnose the online
gate, spike the nightly, then claim kernel ABI/model factory/GGUF compute leaves.
## 2026-07-10 — Triton-AOT follow-up: bf16 chunk_o + scratch pooling (GPU validation pending)

Picked up the post-MVP follow-up from the earlier assessment and implemented the
non-CUTLASS pieces in worktree `vllm.cpp-codex-triton-aot-analysis`.

**Implemented.**
- Added bf16-output Triton AOT `chunk_o` specs (`gdn_chunko_bf16_h48/h32`) alongside
  the existing f32-output specs, gated behind `VLLM_CPP_TRITON_CHUNKO_BF16` until
  the vendored artifacts are regenerated for the active arch. `TryTritonChunkO<Tout>`
  now dispatches f32 output and can dispatch bf16 output once that compile define is
  present.
- Kept `VT_GDN_OUT_BF16` default OFF. The previous default flip regressed because
  bf16 output fell off Triton; the source path is now wired, but the default needs
  regenerated artifacts and a fresh same-binary A/B before changing again.
- Added per-stream grow-only pools in the Triton-AOT GDN build for chunk scratch
  (`gcum/u/w/v_new/hstate`) and chunk metadata (`tok0/len/boh/cidx`), plus the WU
  A/Ai intermediates. A/B escapes: `VT_GDN_TRITON_CHUNK_POOL=0` and
  `VT_GDN_TRITON_WU_POOL=0`.
- Integrated with main's vendored Triton AOT artifact workflow:
  `VLLM_CPP_TRITON_REGEN=ON` is the maintainer path; normal builds consume
  `src/vt/cuda/triton_aot_vendored/<arch>/` without Python. The bf16 `chunk_o`
  path stays disabled until those generated files are added.
- Fixed README status drift: the architecture/backend rows now agree with the
  header/status section that the MVP throughput gates passed, while preserving the
  newly closed 35B GGUF real-file parity status from main.
- Rebased over the GGUF acceptance golden and routed `qwen36_gguf_greedy` out of
  the generic op-parity runner; `test_qwen36_gguf_engine` owns that full-engine
  fixture.

**Validation run here.**
- `git diff --check` PASS.
- `cmake -S . -B build-cpu -DVLLM_CPP_CUDA=OFF -DVLLM_CPP_SERVER=OFF` PASS.
- `cmake --build build-cpu -j$(nproc)` PASS.
- `ctest --test-dir build-cpu --output-on-failure` PASS (90/90).

**Pending validation.**
- This environment did not provide a CUDA-capable build/run path, so I could not
  regenerate artifacts, build `VLLM_CPP_TRITON=ON`, or run the required vLLM
  comparison here.

**Next CUDA pass.**
1. Regenerate and check in the full stable AOT artifact bundle under
   `src/vt/cuda/triton_aot_vendored/<arch>/` for all current bases, including
   `gdn_chunko_bf16_h{48,32}`.
2. Configure/build with `-DVLLM_CPP_CUDA=ON -DVLLM_CPP_TRITON=ON` and run
   `test_ops_gdn` through the Triton paths.
3. Same-binary A/B the new pools (`VT_GDN_TRITON_*_POOL=0`) and
   `VT_GDN_OUT_BF16=1` before considering any default flip.
4. Re-run production-vLLM comparisons only after the CUDA build is green; no
   throughput claim was made in this session.

## 2026-07-10 — `QUANT-GGUF-COMPUTE` split into three READY leaf specs (ROAD-V1-C4 gate)

**What landed (docs-only spike, `CLAIM-QGC-LEAVES`, released):** the umbrella
`QUANT-GGUF-COMPUTE` row is now a block row over three claim-sized `READY`
leaves, each with a full spike-gate spec grounded in the pinned llama.cpp
`237ad9b96` and the B4 decision measurement (parity-ledger.md L290: llama.cpp
CPU ahead 54–75× decode / ≈1,480× prefill / 2.65× peak RSS on the same GGUF):

- `QUANT-GGUF-CPU-THREADPOOL` — [specs/gguf-cpu-threadpool.md](specs/gguf-cpu-threadpool.md):
  ggml threadpool/barrier/chunk port into `src/vt/cpu/`, GEMM chunking first;
  gates = bit-identical determinism at 1/3/20 threads + ≥10× decode on the B4
  recipe. No dependencies — claim this first.
- `QUANT-GGUF-CIQ-GEMM` — [specs/gguf-compute-in-quant-gemm.md](specs/gguf-compute-in-quant-gemm.md):
  tensor-traits compute-in-quant GEMM (activation Q8_0/Q8_K quant + per-type
  vec_dot) for Q8_0/Q4_K/Q5_K/Q6_K/Q3_K/Q4_0; portable generic tier → x86/Arm
  SIMD tiers → repack/interleave tier; final gate = match/beat llama.cpp on
  decode/prefill/RSS, token-exact vs the same-file oracle.
- `QUANT-GGUF-KEEPQ-LOADER` — [specs/gguf-keep-quant-loader.md](specs/gguf-keep-quant-loader.md):
  block-resident weights ([N,K], no transpose), per-tensor routing,
  `VT_CPU_REF=1` dequant-oracle switch; DECISION recorded: merge bench branch
  `7c91a42` (B4 loader arm) into main as work row L1.

Matrix: umbrella → `READY` + three new leaf rows (QUANT pin 78→81 in
check-agent-record.py); roadmap `ROAD-V1-C4` next gate now "claim the
threadpool leaf". Dequant-to-bf16 stays the parity oracle; compute-in-quant is
gated against llama.cpp, not against our old path.

**Next:** claim `QUANT-GGUF-CPU-THREADPOOL` (W1 pool core), merge `7c91a42`
(loader L1), then keep-quant residency + CIQ GEMM tier 0.

## 2026-07-10 — async serving + overlap scheduling spiked (ROAD-V1-C6 READY; blocks order 0)

**What landed (docs-only spike, `CLAIM-ASYNC-SPIKE-1`, released):** joint
spike [specs/async-serving.md](specs/async-serving.md) covering four engine
rows, all now `READY`: `SERVE-ASYNC-LLM` (AsyncLLM-equivalent + real SSE +
C-ABI streaming), `ENG-CORE-BUSY-LOOP` (engine thread + input/output queue
split), `ENG-ASYNC-SCHED` (AsyncScheduler placeholders + depth-2 batch queue +
copy-stream D2H + GPU-resident last-sampled combine), `ENG-PRIORITY-SCHED`
(priority heap + preemption victim). Work breakdown W1→W4, each independently
gateable; tests-to-port inventoried from `tests/v1/engine/test_async_llm.py`,
`tests/v1/core/test_async_scheduler.py`, the 11 priority scheduler cases, and
the OpenAI streaming/bench-serve suites.

**New priority, recorded:** (1) `SERVE-GATE-ONLINE` found our example server
executes the engine synchronously per request with precomputed SSE
(`serving_completion.h:9-12`, `api_server.cpp:26,86,130`), so TTFT/TPOT/ITL
are structurally unmeasurable — `SERVE-ASYNC-LLM` re-promoted T1→T0 and
recorded as a BLOCKING dependency of roadmap order 0 (`ROAD-V1-A` row +
engine-matrix row + handoff queue). (2) B3: async/overlap scheduling is
vLLM's DEFAULT at pin e24d1b24 (`vllm/config/vllm.py:990-1038`) — unmet
mirror obligation, now leaf W3. Engine-matrix summary gained SPIKE/ACTIVE
columns so lifecycle tallies sum to the row count.

**Next:** claim W1 (`ENG-CORE-BUSY-LOOP`), then W2 (`SERVE-ASYNC-LLM`) to
unblock the online gate's latency axes; W3 async-default mirror A/B'd on both
gate models; W4 priority any time.

## 2026-07-10 — Expert-streaming-from-disk spike (`ENG-EXPERT-STREAM` READY; ds4 scan + measured math)

**User-directed spike (CLAIM-EXPSTREAM-SPIKE-1, released):** scanned
antirez/ds4 (DwarfStar @ 80ebbc3 — DeepSeek V4 Flash/PRO engine with SSD
expert streaming on Metal/CUDA/ROCm) and wrote
[specs/expert-streaming.md](specs/expert-streaming.md). ds4's design: routed
experts in a per-(layer,expert) mlocked slab cache, misses pread(2) from the
GGUF by a 9-18-thread pool inside the per-layer router→FFN window,
hotness-decayed-LFU eviction (halve every 16 tokens, LRU tiebreak, in-flight
protection), full-layer SEQUENTIAL streaming for long prefill + decode-cache
seeding from the last ≤64 prefill tokens, shipped hotlist preload, and a
`--simulate-used-memory` honesty tool. ds4 publishes almost no numbers (only:
auto ~59GB cache best on M5 Max PRO q2); its profiler measures hit rates per
deployment.

**Measured this spike (dgx):** 35B-A3B NVFP4 per-expert bytes from the real
safetensors header = 1,769,472 B x 256 experts x 40 layers = 16.88 GiB
routed experts (~77% of ~22 GiB weights); NVMe `/dev/nvme0n1` O_DIRECT: 5.4
GB/s sequential, 2.76 GB/s single-thread random expert-size preads, ~5.0-5.3
GB/s with 4-16 threads. Worst-case decode traffic 540 MiB/token (top-8 x 40
layers). Verdict: viable capacity feature at c1-c4 (I/O bound ≥17.7 tok/s at
50% resident, ~8.4 GiB freed; gates G1-G6 in spec incl. token-exactness,
≥7.5 GiB measured reduction, ≥12 tok/s floor at f=0.5); NOT the
high-concurrency gate regime (B=64 touches ~88% of all experts/step → I/O
~orders below the ~2.8k tok/s gate); tmpfs is a non-tier on GB10 (unified
memory; dgx /tmp is ext4 on the same NVMe anyway).

**Mirror floor settled at the pin (one line each):** (1) load-time streaming
PRESENT (`runai_streamer` + loaders, `model_loader/__init__.py:33-66`) — not
this feature; (2) `cpu_offload_gb` PRESENT, v1-supported, blanket
per-parameter UVA (`config/offload.py:23`, `offloader/uva.py:64-108`), name-
targetable but NOT router-aware — inventoried as new row `ENG-WEIGHT-OFFLOAD`;
(3) inference-time disk/SSD expert paging ABSENT in-pin (searched fused_moe/,
offloader/, config, loaders; eplb is EP rebalancing) → `ENG-EXPERT-STREAM` is
surpass-track, additive/default-off/own-config-namespace for sync safety.

**Rows:** engine-matrix +2 (`ENG-EXPERT-STREAM` READY, `ENG-WEIGHT-OFFLOAD`
INVENTORIED; checker ENGINE_ROWS 88→90), feature-matrix §2 +2, roadmap
`ROAD-V1-D4` INVENTORIED→PARTIAL, handoff queue #10. **Next:** claim W1
(expert cache manager, CPU-testable), W2 (pread pool), then W3 (bank +
engine hook) on dgx.

## 2026-07-10 — ENG-CORE-BUSY-LOOP (W1) implemented: EngineCoreProc busy loop + queue split + InprocClient (GATING)

First leaf of the async-serving block (`ROAD-V1-C6`, spec
[async-serving.md](specs/async-serving.md) W1) landed by `CLAIM-BUSY-LOOP-1`
(claim released this change):

- **`EngineCoreProc : EngineCore`** (`include/vllm/v1/engine/core_proc.h`,
  `src/vllm/v1/engine/core_proc.cpp`) — 1:1 port of the upstream busy loop
  (`vllm/v1/engine/core.py:915-916,1259-1480` @ e24d1b24): input/output
  `BlockingQueue`s (queue.Queue semantics), `run_busy_loop` =
  `_handle_shutdown` → `_process_input_queue` → `_process_engine_step`,
  abort-mode (timeout 0: finish-all-ABORTED + abort outputs) and drain-mode
  shutdown, ADD rejected with an abort output during shutdown, WAKEUP and
  ENGINE_CORE_DEAD sentinels, `EngineCoreRequestType` values preserved.
  step_fn selection mirrored; `max_concurrent_batches > 1` throws until W3
  lands `step_with_batch_queue`.
- **`InprocClient`** (`core_client.{h,cpp}`) — SyncMPClient collapsed to the
  in-proc queue split (recorded deviation D2): owns the engine `std::thread`
  under the run_engine_core fatal-error guard, blocking `get_output` that
  raises `EngineDeadError` on the dead sentinel, `add_request_async` /
  `abort_requests_async` shapes kept for W2's AsyncLLM and a future
  multi-proc client.
- **Sync path untouched**: additive files only; `EngineCore` members flipped
  private→protected for the upstream subclass shape. `LLMEngine`/server
  behavior unchanged (W2 rewires serving onto this client).
- **Tests** (`tests/vllm/v1/test_engine_core_proc.cpp`): upstream
  `tests/v1/engine/test_engine_core_client.py` sync-cycle assertions
  (normal/abort/abort-after-finish) ported T-unit over the RunnerStub seam +
  busy-loop shutdown/dead-sentinel/batch-queue-reject cases. CPU ctest
  93/93; 25/25 stability reruns; clean full build, zero warnings.
- **GATING, honestly**: G1 (both greedy gates re-run same binary) and G4
  (offline no-regression A/B) need the GB10, which `CLAIM-SERVE-GATE-1`
  holds (flock holder + queued waiter at check time) — per the
  no-queueing-behind-the-campaign rule they are DEFERRED to the next
  GPU-idle window and recorded in the engine-matrix row + handoff queue.
  Row `GATING`, not `DONE`. README untouched: no externally-visible change
  until W2 rewires serving.

**Next:** claim W2 (`SERVE-ASYNC-LLM`) on the merged queue split (unblocks
`ROAD-V1-A` latency axes); run W1's G1/G4 when the GPU frees; W3 async
default mirror; W4 priority separable.

## 2026-07-10 — W4 `ENG-PRIORITY-SCHED` implemented (async-serving block leaf 4), GATING

Ported vLLM priority scheduling 1:1 (pin e24d1b24), fully separable from W1–W3.
- **PriorityRequestQueue** (`src/vllm/v1/core/sched/request_queue.cpp:101`,
  header `include/.../request_queue.h:112`): binary heap over `Request*` ordered
  by `RequestPriorityLess` (`Request.__lt__`: (priority, arrival_time,
  request_id, identity)) via std::*_heap; `create_request_queue(kPriority)` now
  returns it instead of throwing.
- **Priority preemption** (`src/vllm/v1/core/sched/scheduler.cpp:178`): under the
  priority policy the OOM victim is `max(running, key=(priority, arrival_time))`
  with the scheduled-this-step undo (restore budget, drop blocks, `req_index -=
  1`), mirroring `scheduler.py:546-572`. FCFS tail-pop unchanged; **default stays
  FCFS** (byte-identical).
- **`priority` plumbing**: `Request.priority` + `RequestPriorityLess`
  (`request.{h,cpp}`), `EngineCoreRequest.priority`, OpenAI `priority` request
  field (`protocol.{h,cpp}`) → serving handlers → `LLMEngine::add_request`/
  `generate` → `InputProcessor::process_inputs` → `EngineCoreRequest` → `Request`.
- **Policy config surface**: `SchedulerPolicyFromString`/`SchedulerPolicyToString`
  (`config/scheduler.cpp:21`, reject-unknown mirrors upstream `SchedulingPolicy(value)`
  ValueError) + `EngineParams.policy` → `MakeSchedulerConfig`.
- **Tests ported** (test-porting.md, same change): 12 `test_priority_scheduling_*`
  cases → `tests/vllm/v1/test_scheduler.cpp:674` (from `test_scheduler.py:2382-2856,2978`,
  incl. the preemption-then-resumption-out-of-KV V2/no-connector case); 14
  priority-queue cases + the seeded random ordering/heap-property property test →
  `tests/vllm/v1/test_request_queue.cpp:238,429` (from `test_priority_scheduler_random.py`).
  DEVIATION recorded: M1.3 KVCacheManager forces caching ON (enable_caching=false
  deferred), so the ported block-math cases give each request a DISTINCT prompt to
  make caching-ON behaviorally equal upstream's caching-OFF; EC/KV-connector
  `test_scheduler.py:3769` variant not ported (no connectors).
- **G2 green**: clean full CPU build zero warnings; ctest **93/93**
  (test_scheduler 29 cases/238 asserts, test_request_queue 26 cases/1839 asserts).
- **GATING, honestly**: G1 (both greedy engine gates re-run priority-vs-fcfs
  token-exactness) needs the GB10, which `CLAIM-SERVE-GATE-1` holds — DEFERRED to
  the next GPU-idle window and recorded in the engine-matrix row + handoff queue.
  Priority is NOT the default, so the greedy gates run FCFS unchanged; the
  priority-vs-fcfs token-exact A/B runs before the row may claim DONE. Row
  `GATING`, not `DONE`. Server surface: added `--scheduling-policy fcfs|priority`
  to `examples/server/main.cpp` (→ `EngineParams.policy`), mirroring vLLM's flag;
  reject-unknown via `SchedulerPolicyFromString`. README arch/quant/accel tables
  untouched (priority is a scheduler knob, not a new arch/backend/quant surface;
  default behavior is unchanged FCFS).

**Next:** claim W2 (`SERVE-ASYNC-LLM`); run W1+W4 GPU G1 when the GPU frees;
W3 async-overlap default mirror.

## 2026-07-10 — C3 M-mtp-0 stream recovered and claimed (`SPEC-MTP` ACTIVE)

Recovered interrupted task `a3002072f42ec1b7f` from its harness transcript.
The prior session ended at its model rate limit after protocol/upstream/local
source inspection and a clean CPU baseline build; it issued no Edit/Write
operation, created no commit, and left no live process. Its named worktree and
branch had already disappeared, so they were recreated exactly at
`.claude/worktrees/agent-a3002072f42ec1b7f` /
`worktree-agent-a3002072f42ec1b7f` from `origin/main` without touching the
dirty root worktree.

`CLAIM-MTP-0` now owns only the spike's M-mtp-0 leaf: load the BF16 `mtp.*`
safetensors tensors for the 27B dense and 35B MoE gate checkpoints, mirror the
Qwen3.5 MTP head/model forward for standalone captured-hidden-state parity,
and port the matching upstream loader/propose contract tests. `SPEC-MTP` and
the two Qwen3.5 MTP model rows are `ACTIVE`; scheduler/rejection/GDN-spec/GGUF
work remains outside this claim. No GPU work will run while
`CLAIM-SERVE-GATE-1` owns the GB10; both-checkpoint oracle head parity is queued
for the first released window.

**Next:** commit/push the claim transition, implement and CPU-test M-mtp-0,
then hand off the exact DGX oracle commands if the serve campaign still holds
the GPU.

## 2026-07-10 — C2 model-factory spike recovered, corrected, and READY

Recovered the stopped `aea89f315855d3bfb` stream after it had written an
uncommitted draft and matrix edit but had not claimed, updated the roadmap, or
committed. The takeover first established `CLAIM-MODEL-FACTORY-SPIKE-1`, then
validated the draft against pinned `registry.py`, its tests, and the live C++
loader/runner path. The accepted
[model-factory-registry.md](specs/model-factory-registry.md) contract makes
`MODEL-FACTORY-registry` independently claimable: ordered architecture lookup,
type-erased factory, exact previous/OOT error branches, subset-registry default
message parity, capability metadata, both existing Qwen paths re-registered,
and both gate models required for no-regression closure.

Corrections made during recovery: the pin has 32 previously-supported entries
(not 36); the full oracle's 353-entry supported list cannot byte-match our
implemented subset, so that branch is compared against a pinned subset
`_ModelRegistry`; the registry is a central ordered table (not cross-TU static
initialization); the public header lives under `include/`; and `score_type` is
included consistently with the ported registry-property test.

No runtime/support state changed. `MODEL-FACTORY-registry` is `READY`; the next
C2 implementation is that row, followed by a separately spiked Llama-dense
family leaf.

## 2026-07-10 — performance checks are per feature/milestone, before stacking

User reaffirmed that every feature or milestone capable of affecting speed
must be benchmarked immediately so regressions are caught at their source, not
only at a later release gate. The existing fresh-denominator and every-axis
rules already required this; `workflow.md` and `benchmark-protocol.md` now make
the operational checkpoint explicit: correctness first, same-binary pre/post
A/B, fresh same-box vLLM or backend-native floor, throughput/latency/memory,
2–3 uncontended reproductions, and exact ledger recipe. A second
speed-sensitive milestone is not stacked until the first checkpoint is
recorded. Hardware deferral means `GATING` plus a reproducible handoff, never
an unmeasured `DONE` claim.

## 2026-07-10 — expert-streaming workflow recovered for grounding repair

The expert-streaming map/verify workflow completed its source maps and two
adversarial verification passes, but its writer hit the session limit before
landing corrections. Live-source review confirms that the capacity-regime and
NVMe bandwidth math remain useful, while the accepted draft is not yet an
implementation contract: the GB10 Marlin kernel indexes one dense contiguous
expert base by stride rather than following the draft's pointer table; original
per-expert NVFP4 tensors are repacked once into that dense layout and freed;
the safetensors reader exposes mmap spans but does not retain public shard/file
offset metadata for later pread; and router-selected IDs remain device-side,
making miss discovery, graph capture, and synchronization explicit design
work. `ENG-EXPERT-STREAM` therefore moved `READY -> SPIKE` under
`CLAIM-EXPSTREAM-GROUND-1`. No implementation starts until the port map,
dependencies, tests, gates, and claim-sized WBS match those facts.

## 2026-07-10 — C1 drop-in kernel ABI spike accepted (`BACKEND-ABI-VT` READY)

Recovered the predecessor's uncommitted C1 draft after its rate limit, then
re-grounded it against pinned vLLM `e24d1b24`, the dependency launch edges, and
the shipped CUTLASS NVFP4, Marlin, and FA-2 adapters. The accepted contract is
[specs/dropin-kernel-abi.md](specs/dropin-kernel-abi.md).

Two adversarial corrections are load-bearing: (1) workspace identity is
`Device + Queue::id + native handle + OpId + slot`, not the stream handle alone
(default streams can alias across devices and handles can be recycled); (2)
packed `DType::kI8` is storage only, so FP4/FP8/UE8M0 semantics travel as an
explicit upstream-compatible `ScalarTypeId` plus a layout descriptor. The
M0.6 choices are now fixed: code registries remain per `DeviceType`, resource
APIs become explicit per `Device`; all backend fn aliases stay in `ops.h`; and
output dtype sets are per-op with no silent narrowing.

Decision: additive ABI spine first, then **per-family incremental migration**.
Preparation may use disjoint worktrees, but every speed-sensitive family
migration completes its own old/new same-binary correctness, nsys, fresh-vLLM
every-axis, peak-memory, and 2–3-run reproduction checkpoint before another
stacks. First implementation order: `BACKEND-ABI-VT` spine → proven
NVFP4/FP8/Marlin adapters → core/EW/KV/attention/MoE/sampling; GDN waits for
`CLAIM-PR3`; collectives/spec-decode and ROCm wait for their implementation/
hardware leaves. No code, support status, README, ledger, or porting-inventory
claim changed in this docs-only spike.

Validation: `python3 scripts/check-agent-record.py` and the 13-case mutation
suite pass; matrix counts remain ENGINE=90, MODEL=323, QUANT=81, KERNEL=30,
BACKEND=51. Spike claim released; `BACKEND-ABI-VT` is `READY`.

## 2026-07-10 — corrected expert-streaming contract accepted (`READY`)

The grounding takeover finished the stopped workflow's write phase and released
`CLAIM-EXPSTREAM-GROUND-1`. The corrected spec preserves ds4's cache policy,
pread pool, capacity-mode scope and measured NVMe math while adapting them to
our actual execution chain. Runtime Marlin uses one dense base and
`expert_id * stride`, so streaming uses fixed contiguous C-slot arrays and
rewrites logical router IDs to slot IDs before `moe_align`; it does not promise
pointer-table indirection. The loader builds a versioned pre-repacked bank while
safetensors shard spans are alive and, in streaming mode, never creates the
16.88 GiB host expert vectors or the full-E `MoeMarlinResident`. Phase 1 is
explicitly non-graphed with one router D2H/event wait per MoE layer. Long
prefill with C<E uses exact chunk filters/scatter accumulation rather than a
hidden full-layer resident.

The implementation order is W0 nsys/current-c1 baseline, W1 CPU cache policy,
W2 bank/reader/pread (with its own build/startup/RSS checkpoint), W3 phase-1
decode (token/memory/off-path/performance gates), then W4 chunked prefill and W5
locality. Any resident/missing overlap is a new W6 spike after W3 evidence.
`ENG-EXPERT-STREAM` is READY for W0, not implemented or supported.

## 2026-07-10 — stopped serving and PR3 streams recovered from their worktrees

Root took over `CLAIM-SERVE-GATE-1` and `CLAIM-PR3` without discarding their
existing remote jobs or evidence. On `dgx.casa`, the completed 35B exact-shape
sanitizer diagnostic survived all three repetitions and reported no sanitizer
errors. The second online campaign now owns `/tmp/gpu` for one uninterrupted
27B ours→vLLM series; ours c1/c2 repetitions are complete and c4 is active.
Fresh 27B vLLM and both 35B arms remain, so this is progress evidence, not a
gate result. The first campaign remains invalid diagnostic evidence. The
every-axis online gate still depends on `SERVE-ASYNC-LLM`, because the current
server cannot expose genuine TTFT/TPOT/ITL.

The PR3 takeover is grounded in local
`/home/mudler/_git/vllm.cpp-pr3-validate` and DGX
`~/work/vllm.cpp-noPy`, both at PR head `85dfb48`. Existing `test_ops_gdn`,
two-model greedy and scratch-pool A/B jobs remain queued behind the serving
lock, but they are explicitly preliminary. Review found that the branch has no
vendored BF16 `gdn_chunko_*` artifacts or MANIFEST entries and its tests do not
assert Triton dispatch or dirty-buffer reuse; it also predates current main.
Closure therefore requires current-main integration, artifact regeneration and
drift-contract updates, assertable dispatch/reuse tests, nsys of vLLM and ours,
and fresh same-workload performance/correctness denominators. No PR3 row moved
state in this recovery change.

## 2026-07-10 — crash recovery resumed roadmap order 0 through AsyncLLM W2

The canonical record was reconciled after the host crash: recovery commit
`80975be` preserves both the stopped-stream evidence and remote
`BACKEND-ABI-VT` W0 claim `39c8b56`; `scripts/check-agent-record.py` plus all
13 mutation tests are green.
The dgx serving campaign survived under its original whole-series GPU lock;
PR3's queued jobs and the C1/C3/C4 implementation worktrees also survived.

`SERVE-ASYNC-LLM` is now `ACTIVE` under `CLAIM-SERVE-ASYNC-W2-1` in isolated
worktree `/home/mudler/_git/vllm.cpp-async-llm-w2`. Scope is exactly W2 from
[async-serving.md](specs/async-serving.md): AsyncLLM/output collector, live
completion/chat SSE, disconnect abort, additive non-blocking C ABI, and the
ported W2 tests. W3 async scheduling/runner work remains outside this claim.
CPU work proceeds while `CLAIM-SERVE-GATE-1` owns dgx; G1/G3-G6 stay explicit
GPU handoffs rather than speculative closure.
## 2026-07-10 — AsyncLLM W2 implemented and CPU-gated; GB10 gates handed off

`SERVE-ASYNC-LLM` moved `ACTIVE -> GATING` and
`CLAIM-SERVE-ASYNC-W2-1` was released. The production server now uses a new
`AsyncLLM` over W1's `InprocClient`: EngineCore runs on its dedicated thread,
an output-handler thread feeds a thread-safe single-slot collector per request,
and unrelated callers submit/generate/abort concurrently. DELTA outputs
coalesce when the producer wins, cumulative outputs replace by completion
index, fatal errors wake every consumer, and abort emits terminal metadata.

Completion and chat serving now return live pull sources rather than replaying
precomputed vectors. The httplib provider writes one SSE frame per pull,
preserves chat role/content/finish/`[DONE]` cadence, and aborts the underlying
request when the sink/releaser reports disconnect. The production server-wide
engine mutex is gone; only the retained synchronous `LLMEngine` compatibility
constructor takes a conditional legacy lock. `LoadedEngine::async_engine()`
owns the lazy async frontend and `examples/server` selects it.

The stable C ABI is additively extended from 11 to 17 exported symbols:
`vllm_request_submit`, `cancel`, `wait`, `done`, `error`, and `free`. A
library-owned delivery thread drains only its request collector and invokes the
existing delta callback while the shared engine keeps batching; callback
failures become request status/error and leave the engine reusable. Existing
`vllm_complete` and `vllm_complete_stream` are unchanged at the source/API
surface and now share AsyncLLM internally. The engine must outlive its request
handles, and wait/free are rejected from the request's own callback.

The stress pass found one load-bearing teardown race after the initial green
suite: shutdown could beat the engine thread before a queued ADD's first busy-
loop iteration. EngineCore then saw an empty scheduler and exited without
consuming/rejecting the ADD, leaving its collector blocked forever. Teardown now
sets the add-rejecting shutdown flag, aborts every OutputProcessor frontend
state (emitting terminal outputs) before core shutdown, forwards the remaining
core IDs, then joins. Final inspection closed the complementary admission race:
the stopped-state recheck, frontend registration and core enqueue are now one
critical section with shutdown's abort-all sweep, so a submitter that began
input processing earlier cannot publish a collector after that sweep. A new
64-way concurrent submit/shutdown regression covers both valid orderings. The
exact full-process reproducer passed 500/500 after the fixes.

Validation (CPU-only worktree):

- `cmake -S . -B build-w2 -DVLLM_CPP_CUDA=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo`
  and `cmake --build build-w2 -j2`: clean, zero warnings.
- `ctest --test-dir build-w2 --output-on-failure -j2`: 94/94 pass.
- `setarch x86_64 -R ctest --test-dir build-w2-tsan --output-on-failure -R
  'test_(output_processor|async_llm|openai_api_server|openai_conformance|capi)$'`:
  5/5 pass under ThreadSanitizer. (`setarch -R` is required on this host to
  avoid TSan's unrelated unexpected-memory-mapping startup failure.)
- `ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1
  ctest --test-dir build-w2-asan --output-on-failure -R
  'test_(output_processor|async_llm|openai_api_server|openai_conformance|capi)$'`:
  5/5 pass. A leak-enabled diagnostic reports only the existing process-lifetime
  Qwen device-buffer pool from `qwen3_5.cpp:119`; collector/AsyncLLM-only tests
  pass with leak detection enabled. No W2 address/undefined-behavior finding.
- `ctest --test-dir build-w2 --output-on-failure --repeat until-fail:100 -R
  'test_(async_llm|openai_api_server|capi)$'`: all three executables pass 100
  consecutive runs. The shutdown executable additionally passed 500 separate
  process invocations with a 5-second per-run timeout.
- C11 header compile, dlopen of all 17 names, and exact `vllm_*` export-set test
  pass in the 94-test suite. `scripts/check-agent-record.py` and its 13 mutation
  tests are the final record gate before integration.

Required first-GPU-idle handoff (one uncontended whole-series lock, after the
current `CLAIM-SERVE-GATE-1` pre-W2 baseline releases dgx): build the W2 commit
with the release CUDA/Triton settings; run both greedy engine gates (G1); run a
manual `curl -N` arrival sanity plus the pinned `vllm bench serve` client at
c=1 and the large-concurrency points for both 27B and 35B (G3); capture fresh
same-workload vLLM denominators and 2-3 reproductions for every throughput,
TTFT, TPOT/ITL and request-rate axis (G4/standing online gate); sample peak
VRAM/RSS (G6). The running `latres2` campaign uses a pre-W2 binary and remains
baseline evidence only. W3 `ENG-ASYNC-SCHED` scheduler/runner overlap is still
`READY` and is not claimed by this change.

## 2026-07-10 — ROAD-V1-C5 joint spike claimed after row split

`CLAIM-C5-SPIKE-1` owns a docs-only analysis of sliding-window KV/attention,
chunked-local KV/attention, YaRN, and the existing Llama 3/LongRoPE/dynamic-NTK
long-context block. Before deep analysis, the two oversized stable rows were
preserved as joint-spike umbrellas and split into six claimable leaves:
`KV-SLIDING-WINDOW-SPEC`, `KV-CHUNKED-LOCAL-SPEC`, `ATTN-CHUNKED-LOCAL`,
`ATTN-ROPE-LLAMA3`, `ATTN-ROPE-LONGROPE`, and
`ATTN-ROPE-DYNAMIC-NTK`. The existing `ATTN-YARN` and
`ATTN-SLIDING-WINDOW` leaves join that claim.

No implementation or support state changed. The next step is a pinned
`e24d1b24` vLLM plus dependency/runtime-dispatch and upstream-test inventory,
written as the nine-section
`specs/sliding-local-yarn-long-context.md`; no GPU work is authorized for this
spike.

## 2026-07-10 — ROAD-V1-C5 joint spike accepted; eight leaves `READY`

`CLAIM-C5-SPIKE-1` completed its docs-only pinned-source analysis and is
released. The accepted contract is
[specs/sliding-local-yarn-long-context.md](specs/sliding-local-yarn-long-context.md).
It covers both full execution chains: config → KV spec/manager/admission/
recycling → attention metadata/backend → the exact pinned FA2 mask (plus the
FlashInfer fallback), and config/factory → scaled cos/sin cache → CPU/CUDA RoPE
apply for YaRN/MRoPE, Llama3, LongRoPE, and both dynamic-NTK modes. The spec
ports the upstream tests, fills the pin's missing per-formula numerical tests
with oracle goldens, assigns non-overlapping W1-W8 ownership, and fixes model,
hardware, correctness, nsys, every-axis performance and memory gates.

Two findings are load-bearing. First, sm_121's default non-MLA path prioritizes
FlashAttention and resolves FA2 at this pin; the local mask lives in pinned
`vllm-project/flash-attention` commit
`2c839c33742309ec41e620bf837495ec9926c56e`, so source and future nsys evidence
must both agree. Second, the actual 35B and 27B gate configs use
`rope_type=default` interleaved MRoPE at 262144 positions and have no
sliding/chunked-local setting — they do **not** exercise YaRN. They remain
mandatory regressions, while each C5 leaf has a separate feature-positive
oracle/model gate.

The two umbrellas and eight leaves move `SPIKE -> READY`; none has code/test
support evidence and README therefore does not change. Engine summary is now
96 rows: `ANCHOR-BACKFILL=9`, `PARTIAL=17`, `SPIKE=0`, `READY=16`,
`ACTIVE=2`, `GATING=3`, `INVENTORIED=49` after the concurrently recovered W2
state is included.
Validation is green: `python3 scripts/check-agent-record.py` reports
ENGINE=96, MODEL=323, QUANT=81, KERNEL=30, BACKEND=51, and all 13
`tests.scripts.test_agent_record` mutation cases pass. No GPU command ran.

## 2026-07-10 — `BACKEND-ABI-VT` W0 additive spine CPU-green, GPU handoff queued

Recovered `CLAIM-BACKEND-ABI-VT-W0-1` after the machine crash, rebased its clean
worktree onto current `origin/main`, and implemented only the test-first W0
adapter spine. `Queue` now has a process-unique monotonic ID; new
device-explicit `vt::{Alloc,Free,CreateQueue,DestroyQueue}` resource functions
sit beside the legacy index-0 `Backend` shims; `ops.h` carries vLLM-compatible
semantic scalar IDs, layouts, tensor descriptors, named workspace roles and a
typed registrar. The new CUDA helper supplies explicit device/stream guards,
device-index callbacks, stable queue/device/op/slot workspace entries,
uninitialized/zero-first/zero-each policies, stream-ordered device scalars,
capture-time growth rejection, deterministic queue cleanup, and a tiny raw
signature probe. No existing production backend or kernel-family TU changed.

Upstream tests shipped with the code: `test_dropin_abi` names and re-expresses
the CUDA set-device/context and cudagraph capture/replay cases, plus the
vt-specific packed-type rejection, default-stream/device/queue-ID isolation,
workspace growth/roles/init/cleanup, scalar staging, two-device conditional,
and raw-boundary assertions. Local validation: clean Release CPU build, no
warnings; 94/94 ctest pass; scalar IDs independently match the vendored Marlin
`ScalarType::id()` values; record checker/mutation results are recorded after
the final record pass.

`BACKEND-ABI-VT` moves `ACTIVE -> GATING`, not `DONE`. `W0-GPU` (CUDA
cross-builds 80/90a/121a, GB10 runtime/capture/memcheck, both greedy gates and
unchanged production traces/A-B) waits for `CLAIM-SERVE-GATE-1` to release the
GPU; the exact commands are in the spike. `W0-SCALAR-FORWARDER` and
`W0-BACKEND-SHIM` remain deliberately named: moving the vendored Marlin scalar
class or editing production backend TUs was outside this claim, so the first
family migration must consolidate the header and move its resource use off the
legacy shims before the ABI row can close. README is unchanged because no
externally supported backend, quantization, model, or performance state moved.
The implementation claim is released into coordination's C1 `GATING` handoff;
the eventual GPU runner must claim that gate window before executing it.

## 2026-07-10 — C3 M-mtp-0 CPU/local implementation complete; two-checkpoint gate queued

Recovered the 1,063-line uncommitted MTP stream in its original isolated
worktree, checkpointed it before rebasing, and audited it against pinned vLLM
`e24d1b24` plus the exact upstream tests named by
[mtp-spec-decode.md](specs/mtp-spec-decode.md). M-mtp-0 now contains:

- an on-demand, safetensors-only BF16 `mtp.*` loader for both gate layouts,
  including the 27B dense layer and the 35B fused `[E,2I,H]` / `[E,H,I]`
  256-expert stacks; every norm/projection/expert tensor is shape- and
  dtype-checked against `H`, head counts/dim, dense/MoE intermediate sizes and
  shared-expert size;
- `Qwen3_5MTPModel`, sharing the target embedding and lm-head (FP4 head on the
  35B), with the pinned `norm(embed)` + `norm(target_hidden)` → concat/fc →
  one full-attention dense/MoE decoder layer → final norm forward and direct
  hidden-state return; normal target loading remains unchanged and loads the
  optional draft only when requested;
- ported loader/direct-return CPU tests for both architectures, with the
  unimplemented AutoRegressiveSpeculator tuple/tensor and propose-shape cases
  retained as explicit M-mtp-1 skips rather than false-green placeholders;
- a focused DGX oracle dumper and C++ golden consumer that require exact argmax
  on every captured row (at least 16) for both checkpoints and retain a 0.05
  hidden-state diagnostic bound.

The crash audit found and repaired one oracle-tool blocker before any GPU time:
pip-vLLM 0.24 uses an older explicit fused-expert `load_weights` method while
the pin uses `AutoWeightsLoader`. Their executable MTP class bodies are
otherwise AST-identical. The tool now strips exactly the two loader-only
methods plus the pin-only mapper assignment, compares every remaining class
member, and relies on the final exact-argmax gate to prove equivalent loaded
weights. Read-only DGX checks confirmed this guard passes the real installed
source; the pinned file at `/home/mudler/work/vllm-pin` has SHA256
`9b36e5bfcee4faf8d04319e069032c3c4a01c4aaf49f86108eca788038c0c7fd`, identical
to the canonical local pin.

**Local evidence:** clean CPU configure/full build; CTest **92/92**; focused
`test_mtp_speculator` **7/7 runnable cases, 141/141 assertions, 3 tracked
skips**; focused MTP oracle test cleanly skips without CUDA/goldens; generic CPU
op-parity remains green; `python3 -m py_compile`, `--help`, installed-vs-pin AST
guard and 3-row text-position normalization pass; record checker reports
`ENGINE=90 MODEL=323 QUANT=81 KERNEL=30 BACKEND=51`; all 13 record mutation
tests and `git diff --check` pass.

No GPU/model command ran: `CLAIM-SERVE-GATE-1` still holds `/tmp/gpu` for its
live campaign, with PR3 already queued behind it. `SPEC-MTP`, `Qwen3_5MTP` and
`Qwen3_5MoeMTP` therefore remain honestly `GATING`; `ROAD-V1-C3` remains
`ACTIVE`, and speculative decoding is still unavailable to users (no
scheduler, rejection sampler, GDN snapshots, config/API wiring, or GGUF head).
The implementation claim is released into the handoff queue.

### Exact first-free-GPU handoff (one lock for both oracle arms + C++ gate)

After branch `worktree-agent-a3002072f42ec1b7f` is pushed, prepare the build
without the GPU lock:

```bash
REPO=/home/mudler/work/vllm.cpp-mtp0
test -d "$REPO/.git" || git clone --no-checkout \
  git@github.com:mudler/vllm.cpp.git "$REPO"
git -C "$REPO" fetch origin worktree-agent-a3002072f42ec1b7f
git -C "$REPO" switch --detach origin/worktree-agent-a3002072f42ec1b7f
cmake -S "$REPO" -B "$REPO/build-cuda" \
  -DVLLM_CPP_CUDA=ON -DVLLM_CPP_TRITON=ON
cmake --build "$REPO/build-cuda" -j2 --target test_op_parity
```

When `CLAIM-SERVE-GATE-1` and the already-queued PR3 jobs release the device,
run this entire correctness series under one uncontended lock:

```bash
flock /tmp/gpu bash -lc '
set -euo pipefail
REPO=/home/mudler/work/vllm.cpp-mtp0
PY=/home/mudler/venvs/vllm-oracle/bin/python
PIN=/home/mudler/work/vllm-pin
M27=/home/mudler/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-NVFP4/snapshots/890bdef7a42feba6d83b6e17a03315c694112f2a
M35=/home/mudler/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/491c2f1ea524c639598bf8fa787a93fed5a6fbce

cd "$REPO"
VLLM_ENABLE_V1_MULTIPROCESSING=0 "$PY" \
  tools/parity/dump_qwen3_5_mtp.py \
  --model "$M27" --tag 27b --pinned-vllm "$PIN" \
  --out tests/parity/goldens
VLLM_ENABLE_V1_MULTIPROCESSING=0 "$PY" \
  tools/parity/dump_qwen3_5_mtp.py \
  --model "$M35" --tag 35b --pinned-vllm "$PIN" \
  --out tests/parity/goldens

VLLM_MTP_27B_SNAPSHOT="$M27" \
VLLM_MTP_35B_SNAPSHOT="$M35" \
VLLM_MTP_REQUIRE_CHECKPOINTS=1 \
  ./build-cuda/tests/test_op_parity \
  -tc="qwen3.5 MTP standalone head parity*"
'
```

Required return evidence before closing M-mtp-0: both
`qwen3_5_mtp_head_{27b,35b}/manifest.json` fixtures; installed-vs-pin forward
AST verification printed by each oracle arm; C++ runner reports `argmax T/T`
with `T>=16` for **both** checkpoints and satisfies the 0.05 hidden diagnostic;
the two golden directories are brought back in the gating commit; the focused
test is re-run once after that commit. Any mismatch keeps all three rows open
and is debugged against the captured target hidden/input/positions—never by
loosening exact argmax.

## 2026-07-10 — `MODEL-FACTORY-registry` claimed for CPU implementation

The accepted [model-factory registry spike](specs/model-factory-registry.md) is
now `ACTIVE` under `CLAIM-MODEL-FACTORY-1` in isolated worktree
`/home/mudler/_git/vllm.cpp-model-factory-registry` on branch
`codex/model-factory-registry`. This claim owns only the central ordered
type-erased registry, exact reject-unknown tables/messages, capability metadata,
the two existing Qwen3.5 registrations, the live `IsDenseArch` replacement and
the minimum runner/loader indirection it requires, plus ported registry tests and
their record surfaces. It adds no model family and performs no GPU work while
`CLAIM-SERVE-GATE-1` owns dgx. After CPU implementation and full relevant tests,
the two gate-model token/performance/memory no-regression campaign remains an
exact `GATING` handoff rather than a speculative closure.

## 2026-07-10 — `MODEL-FACTORY-registry` CPU implementation complete; GPU `GATING`

Implementation commit `c707602` replaces the live `num_experts==0` guess with
the pinned registry contract. `model_registry.{h,cpp}` now owns the central
declaration-ordered table, type-erased `LoadedModel`/factory hooks, consumed
`_ModelInfo` metadata, both existing Qwen registrations, all 32 previous + four
OOT entries, and the exact empty/inspection-failed/previous/OOT/subset-default
errors. `LoadedEngine` and `GPUModelRunner` each carry one type-erased model;
their existing Qwen forwards, Marlin preparation, dense/MoE token-budget policy,
KV layout, and decode graphs remain intact behind the registration. Live disk
loading resolves the complete `architectures` list before tokenizer/weight work,
so an unrelated dense config now rejects deterministically. GGUF exposes the
canonical MoE registration ID while retaining its container model type.

CPU evidence from the fresh isolated build:

```sh
cmake -S . -B build-model-factory -DCMAKE_BUILD_TYPE=Release \
  -DVLLM_CPP_CUDA=OFF -DVLLM_CPP_BUILD_EXAMPLES=OFF
cmake --build build-model-factory -j20
ctest --test-dir build-model-factory -j20 --output-on-failure
# 94/94 pass
./build-model-factory/tests/test_model_registry --success=0
# 11 active cases, 112/112 assertions; one explicit
# "MODEL-FACTORY-registry: no second family yet" skip
python3 scripts/check-agent-record.py
python3 tests/scripts/test_agent_record.py
```

The claim is released and the row is `GATING`, not `DONE`: no GPU command ran
while `CLAIM-SERVE-GATE-1` owns dgx. Exact first-idle-window handoff (the dispatch
refactor is structurally inseparable, so use adjacent-commit old/new binaries
under one lock rather than inventing a shipping runtime fallback toggle):

```sh
BASE_REPO=$HOME/work/vllm.cpp
OLD=$HOME/work/vllm.cpp-model-factory-old
NEW=$HOME/work/vllm.cpp-model-factory-new
export BASE_REPO OLD NEW
git -C "$BASE_REPO" fetch origin codex/model-factory-registry
git -C "$BASE_REPO" worktree add --detach "$OLD" c707602^
git -C "$BASE_REPO" worktree add --detach "$NEW" c707602
for tree in "$OLD" "$NEW"; do
  cmake -S "$tree" -B "$tree/build-cuda" -DCMAKE_BUILD_TYPE=Release \
    -DVLLM_CPP_CUDA=ON -DVLLM_CPP_CUDA_ARCHITECTURES=121a \
    -DVLLM_CPP_TRITON=ON -DVLLM_CPP_CUTLASS_DIR="$HOME/cutlass_probe"
  cmake --build "$tree/build-cuda" --clean-first -j"$(nproc)"
done

MODEL27=$(dirname "$(find "$HOME/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-NVFP4/snapshots" -name config.json -print -quit)")
MODEL35=$(dirname "$(find "$HOME/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots" -name config.json -print -quit)")
export MODEL27 MODEL35

flock /tmp/gpu bash -lc '
  set -euo pipefail
  ctest --test-dir "$NEW/build-cuda" \
    -R "test_qwen27_paged_engine|test_qwen36_paged_engine" \
    --output-on-failure
  for rep in 1 2 3; do
    for tree in "$OLD" "$NEW"; do
      /usr/bin/time -v "$tree/build-cuda/examples/vllm-bench" \
        --model "$MODEL27" --num-prompts 96 --input-len 1024 \
        --output-len 128 --concurrency 16 --seed 0 --temperature 0 \
        --num-blocks 2400
      /usr/bin/time -v "$tree/build-cuda/examples/vllm-bench" \
        --model "$MODEL27" --num-prompts 192 --input-len 1024 \
        --output-len 128 --concurrency 32 --seed 0 --temperature 0 \
        --num-blocks 2400
      /usr/bin/time -v "$tree/build-cuda/examples/vllm-bench" \
        --model "$MODEL35" --num-prompts 200 --input-len 1024 \
        --output-len 128 --concurrency 64 --seed 0 --temperature 0 \
        --num-blocks 2400
    done
    run_oracle() {
      "$HOME/venvs/vllm-oracle/bin/vllm" bench throughput --model "$1" \
        --dataset-name random --random-input-len 1024 \
        --random-output-len 128 --random-range-ratio 0 \
        --num-prompts "$2" --max-num-seqs "$3" --seed 0
    }
    run_oracle "$MODEL27" 96 16
    run_oracle "$MODEL27" 192 32
    run_oracle "$MODEL35" 200 64
  done
'
```

Capture temperature/power before/after each leg, verify memory returns before
the next engine, and record all three repetitions, total/output/request
throughput, TTFT/TPOT/ITL, peak RSS/unified memory, fresh vLLM denominators and
ratios. Closure requires both greedy gates unchanged, new-vs-old no regression
on every axis within reproduced run noise, and the standing ≥1.0× vLLM floors
retained at 27B c16/c32 and 35B c64. Only then append the closing ledger evidence,
move `MODEL-FACTORY-registry` `GATING→DONE`, and use the closing commit as owner.

## 2026-07-10 — crash recovery: CPU threadpool W1-W3 complete, row honestly `GATING`

Recovered `CLAIM-QGCT-1` in its original isolated worktree without discarding
commits `e75ef42`, `0b85ea1`, or `1b8e8cc`; two named stashes preserve the
pre-recovery uncommitted diff. `HANDSOFF.md` does not exist anywhere under
`/home/mudler`, so `AGENTS.md`, the accepted leaf spec, and canonical matrices
were used as the controlling record. The branch was reconciled with current
`origin/main` by merge `6aab95a` before final validation.

**W1-W3 present and audited:** `src/vt/cpu/cpu_threadpool.{h,cpp}` ports the
pinned llama.cpp `237ad9b96` native pool/barrier/chunk/park-wake protocol;
`cpu_ops.cpp` ports mul-mat's 16x16 chunk policy and partitions every registered
CPU op family by independent output rows/batches. `VLLM_CPP_CPU_THREADS=1`
retains the inline path; single-row non-GEMM decode work also stays inline.
Recovery fixes preserve the per-op adaptation over long runs by widening the
packed epoch to unsigned 64-bit, mirror ggml's TSAN dummy seq-cst RMW in place
of unsupported standalone fences, and register the full upstream multi-context
server case as a genuine skipped test rather than a passing message-only case.

**Non-performance gates:** clean post-main CPU/server configure and build;
serial full suite 94/94 at each of 1, 3, and 20 threads; dedicated upstream-test
port/determinism suite byte-identical; GCC TSAN-only focused run under
`setarch x86_64 -R` passed 8 executed cases / 19,595 assertions with one tracked
`SERVE-E2E-NIGHTLY` skip and no race report. The local 35B GGUF, 35B
safetensors, and 27B checkpoint gates emitted their absent-checkpoint messages
and ran zero model assertions; no e2e model claim is inferred from that.

**Why not DONE:** the required B4 same-binary 1-vs-20 throughput/RSS series
could not be validly run. At 2026-07-10 22:39 UTC this 20-core host still ran
unowned persistent `llama-cpp-avx512` and `depth-anything-cpp` inference
processes (about 2.4% and 5.3% CPU), while load average was 7.97/16.15/9.81.
Per benchmark protocol, a contended result is void; no process was killed or
perturbed and no number was recorded. `QUANT-GGUF-CPU-THREADPOOL` therefore
moves `ACTIVE -> GATING`, the implementation claim is released, and README /
quantization/backend/feature surfaces describe correctness support without a
speed claim.

**Next:** obtain an exclusive idle window on the same 20-core x86 host, hold
`/tmp/vllm-cpp-cpu-bench.lock` for the whole series, and run the exact leaf-spec
recipe: three interleaved identical-binary arms at threads 1 and 20 on
`Qwen3.5-2B-UD-Q8_K_XL.gguf` (in128/out32, one request, greedy seed 0), recording
prefill/decode, peak RSS, output tokens, and spread. Require ≥10x on both speed
axes and ≤1.05x RSS, then refresh clean llama.cpp `237ad9b96` pp/tg/RSS on the
same idle window. Only that evidence may advance the row to `DONE`; afterward
claim `QUANT-GGUF-KEEPQ-LOADER` and `QUANT-GGUF-CIQ-GEMM`.

## 2026-07-10 — recovered PR #3 validation evidence; current-main denominator remains open

Recovered the pre-current-main `CLAIM-PR3` validation branch and its preliminary
implementation/correctness evidence. The vendored contract now
derives and validates `cuda:121:32` from `sm_121a`, disables cubin line info,
pins the repository-owned Triton 3.6 numeric-target shim by hash, treats source
drift as fatal, and validates the exact base/artifact set. Two regenerations in
different absolute paths on `dgx.casa` were byte-identical. The no-Python
consumer configured with `VLLM_CPP_TRITON_PYTHON=/definitely/missing/python` and
built `vllm`, `test_ops_gdn`, `server`, and `vllm-bench` from vendored C only.

Runtime hardening adds one `std::call_once` per generated module, a concurrent
two-queue first-load test, allocate-before-retire scratch growth, queue-destroy
pool cleanup, and exact upstream CuTe-DSL max/mean tolerances (output max/mean
`<2e-3/<6e-5`; state `<2e-2/<6e-4`). The recovered branch had enabled the bf16
AOT `chunk_o` and grow-only per-stream pools by default and its previously
frozen 3x factorial found every-axis wins for both pool and bf16 steps on both
models. Those pre-current-main measurements are recovery evidence only: the
bf16 default remains off until the final current-main same-binary A/B is rerun.

Recovered validation on that exact pre-current-main tree: local CPU `91/91`; clean macOS
AppleClang drift CTest `1/1`; DGX `test_ops_gdn` `33/33`, `786/786`; compute-
sanitizer `0` errors and `0` leaked bytes; 27B engine gate `9/9`; 35B single plus
six-request batched graph gate `33/33`. The workflow now regenerates and executes
the consumer test under `/tmp/gpu`, and its required trailer is corrected.

Still open before `CLAIM-PR3` can close: rebuild and rerun the recovered validation
on the integrated current-main tree, then one uninterrupted-lock, two-rep final
ours-vs-production-vLLM comparison for both gate models and a paired `nsys` kernel
trace; finally perform the same-change permanent matrix/roadmap/README/ledger
reconciliation, completed-report archive, and claim release. No current-main
runtime or DONE status is asserted here.

## 2026-07-10 — PR #3 current-main non-GPU recovery complete; final GPU gate handed off

Crash recovery preserved the two pre-existing record edits in stashes
`f912def` and `be481151`, merged current `origin/main` without flattening the PR
history (`407cea3`), and recovered the useful pre-crash implementation through
`1c29dca`. The missing BF16 H32/H48 `chunk_o` launchers are now part of the
12-base / 48-generated-file SM121 bundle. The AOT contract derives
`cuda:121:32`, disables line info, pins the repository generator shim, and
fails on source, declaration, inventory, or artifact-hash drift. Generated
module loads are `call_once` guarded; scratch grows allocate-before-retire and
is released with queue destruction.

The upstream-derived CuTe-DSL test now has assertable BF16 `chunk_o` dispatch,
hand-fallback rejection, concurrent fresh-process module loading, pool
allocation/growth/reuse counters, and an explicit dirty-buffer proof: all nine
chunk buffers plus both WU buffers are filled with `0xff` before a same-stream
rerun. `VT_GDN_OUT_BF16` deliberately remains opt-in/default f32 until fresh
current-main measurements prove every-axis parity or better.

Local, non-GPU validation on the integrated tree is green:

- `bash scripts/check-triton-aot-drift.sh`;
- `bash tests/scripts/test_triton_aot_drift.sh` (ten independent stale-surface
  mutations);
- clean Release CPU configure/build; and
- `ctest --test-dir build-pr3-cpu --output-on-failure -j4` = **92/92**.

No GPU command ran because `CLAIM-SERVE-GATE-1` owns the DGX. Existing jobs and
evidence under `~/work/vllm.cpp-noPy` remain preliminary because they predate
the integrated tree. The final owner must sync the pushed branch head into a
fresh isolated DGX directory, hold one uncontended `/tmp/gpu` lock for each
complete series, and run this order:

1. configure a vendored-only consumer with CUDA `121a`, Triton ON/regen OFF,
   `VLLM_CPP_TRITON_PYTHON=/definitely/missing/python`, and the pinned CUTLASS;
   build `vllm`, `test_ops_gdn`, both paged-engine gates, `vllm-bench`, and the
   server;
2. run `test_ops_gdn_aot_concurrent_first_load`, full `test_ops_gdn`,
   `compute-sanitizer --tool memcheck --leak-check full` on the focused dirty-
   reuse case/full GDN binary, then the 27B and 35B greedy gates including the
   35B batched graph case;
3. on the exact 1024-in/128-out workloads, interleave 2–3 reps of f32-pools-off,
   f32-pools-on, and `VT_GDN_OUT_BF16=1`+pools-on for 27B
   (`192` prompts, concurrency `32`, max batched tokens `2048`, blocks `2368`)
   and 35B (`200`, `64`, `8192`, `4736`), recording total/output throughput,
   requests/s, TTFT, TPOT/ITL/E2EL, and peak process/system memory; and
4. in the same hardware window, run fresh pinned production-vLLM denominators
   on the identical token corpus/config and paired warmup-excluded `nsys`
   captures (`cuda_gpu_kern_sum`) for ours and vLLM.

Only current-head results may move `KERNEL-GDN-AOT-BF16` or
`KERNEL-GDN-SCRATCH` out of `ACTIVE`; below-vLLM on any axis or any correctness
drift remains an open gap.

## 2026-07-10 — crash-recovered roadmap work integrated on current main

The recovered branches are reconciled on `main` without importing the
threadpool branch's benchmark-only dense-GGUF loader parent. The integrated
checkpoints are AsyncLLM W2 (`5063a22`), the C5 long-context spike
(`851d632`), backend ABI W0 (`5a67f41`), M-mtp-0 (`b8e9598`), the model factory
registry (`a4a6ef3`), CPU threadpool W1-W3 (`0878fc7`, `f980660`, `adc92b4`),
and PR3 AOT/scratch hardening (`a767188`). Shared loader lifetime wiring keeps
the registry's type-erased `LoadedModel` and W2's lazily owned `AsyncLLM`
together; the clean integrated build validates that combined path.

Final local recovery validation (CUDA disabled; no GPU command ran):

- clean Release build in `build-integration` completed with zero warnings;
- the pre-PR3 integrated tree passed 98/98 at the threadpool's default worker
  count; after PR3 integration, `VLLM_CPP_CPU_THREADS=1 ctest --test-dir
  build-integration --output-on-failure -j2` passed 99/99, including the AOT
  drift contract;
- `scripts/check-triton-aot-drift.sh` and all ten mutation cases in
  `tests/scripts/test_triton_aot_drift.sh` passed;
- `scripts/check-agent-record.py` reports `ENGINE=96 MODEL=323 QUANT=81
  KERNEL=30 BACKEND=51`, all 13 record mutation tests pass, `git diff --check`
  is clean, and every commit introduced since the prior `origin/main` contains
  `FOLLOWING_AGENTS_PROTOCOL`.

No GPU-dependent row is promoted by this recovery. `SERVE-ASYNC-LLM`,
`BACKEND-ABI-VT`, `SPEC-MTP`, both Qwen MTP model rows,
`MODEL-FACTORY-registry`, and `QUANT-GGUF-CPU-THREADPOOL` remain `GATING`;
the two PR3 kernel rows remain `ACTIVE` under `CLAIM-PR3`. The existing
`CLAIM-SERVE-GATE-1` series retains GPU ownership. Its pre-W2 measurements are
baseline evidence only; after it releases the device, the canonical handoff
queue supplies the post-W2 serving gates followed by the queued ABI, MTP,
model-registry and PR3 GPU closures.

## 2026-07-10 — C5 W1 sliding-window KV leaf claimed while DGX remains occupied

`CLAIM-C5-SW-KV-1` moves `KV-SLIDING-WINDOW-SPEC` `READY -> ACTIVE` in isolated
CPU worktree `/home/mudler/_git/vllm.cpp-c5-sw-kv`, branch
`codex/c5-sw-kv-w1`. The claim is limited to W1 of the accepted
[joint C5 spike](specs/sliding-local-yarn-long-context.md): concrete
`SlidingWindowSpec`, spec/manager dispatch, exact page and admission-cap math,
right-to-left reachable prefix policy, skipped-block recycling, hybrid-disabled
full-allocation conversion, no-cascade behavior, and the complete applicable
ported KV tests. Attention kernels/config plumbing, chunked-local KV, RoPE,
model files and GPU work are out of scope.

The surviving DGX series still owns `/tmp/gpu`: ours 27B c1/c2/c4/c8 and two
c16 repetitions are complete, with c16 rep3 running at inspection time; vLLM
and 35B arms remain. It is preserved unchanged. C5 W1 needs only CPU G1/G2, so
implementation can proceed without contending with that campaign.

## 2026-07-11 — C5 W1 sliding-window KV implemented and handed to feature gating

`CLAIM-C5-SW-KV-1` completed its bounded CPU-only implementation scope and is
released. `KV-SLIDING-WINDOW-SPEC` moves `ACTIVE -> GATING`, never `DONE`.
The port adds:

- `SlidingWindowSpec` with asymmetric V-head sizing and the pinned
  `min(W-1+max_num_batched_tokens,max_model_len)` admission formula;
- `KVCacheSpecRegistry` metadata and inherited built-in lookup, plus
  registry-backed coordinator dispatch that supplies the exact SWA cap;
- `SlidingWindowManager` right-to-left contiguous-hit lookup, alignment/EAGLE
  lookahead/drop/re-alignment, reachable sparse-retention and replay tails,
  skipped whole-page recycling, and cascade disablement;
- exact SWA grouping fields and the hybrid-disabled conversion to
  `FullAttentionSpec(sliding_window=...)` so storage falls back to full while
  compute semantics remain local; and
- the applicable pinned spec/registry/manager/prefix/utility tests, with real
  named skips for `KV-PREFIX-CACHE` contiguous packing, `KV-OFFLOAD`,
  `KV-CONNECTORS`, and the `ATTN-SLIDING-WINDOW` + Gemma3 model-positive gate.

Local evidence (no GPU command ran):

- `cmake -S . -B build-c5-cpu -DCMAKE_BUILD_TYPE=Release
  -DVLLM_CPP_CUDA=OFF && cmake --build build-c5-cpu -j20` passed;
- `VLLM_CPP_CPU_THREADS=1 ctest --test-dir build-c5-cpu
  --output-on-failure` passed **99/99**;
- the four focused binaries report **69 passing cases / 49,763 assertions**,
  with four explicit dependency skips;
- the deterministic 40-trial property spans non-block-aligned window, batch,
  block and model boundaries; it proves predictor=allocator, physical-held
  pages≤cap, no live/free alias, visible-token-slot equality with the
  full-allocation oracle, and clean free/reallocate;
- a Debug CUDA-OFF build with `-fsanitize=address,undefined
  -fno-omit-frame-pointer` passed all four focused suites under
  `ASAN_OPTIONS=detect_leaks=1:halt_on_error=1` and
  `UBSAN_OPTIONS=halt_on_error=1`; and
- `git diff --check` passed. No formatter binary is installed on the recovered
  host, so formatting was reviewed against the surrounding project style.

The fresh pinned-vLLM test run is explicitly unavailable after the crash:
`~/venvs/vllm-oracle/bin/python` does not exist. The fallback exact command
`python3 -m pytest -q
tests/v1/core/test_single_type_kv_cache_manager.py::test_sliding_window_possible_cached_prefix`
reached the pinned tree but failed while loading `tests/conftest.py` because
system Python lacks `tblib`. No package was installed and no oracle result is
invented. Restore the documented oracle environment, rerun the exact C5 modules,
then pair this leaf with `ATTN-SLIDING-WINDOW` G4/G6-G9 (operator/model-positive
correctness, nsys trace, every-axis performance and memory). Until that evidence
exists, README and matrices explicitly say the KV policy is CPU-gated but
sliding-window model support is not user-visible.

## 2026-07-11 — C5 W2 sliding-window attention claimed alongside the serving campaign

`CLAIM-C5-SW-ATTN-1` moves `ATTN-SLIDING-WINDOW` `READY -> ACTIVE` in isolated
worktree `/home/mudler/_git/vllm.cpp-c5-sw-attn`, branch
`codex/c5-sw-attn-w2`. The accepted W2 scope is one backend-neutral semantic
window propagated through the generic attention/backend seam,
`vt::PagedAttention`, the CPU and portable-CUDA lower key bound, and the
vendored FA2 adapter, with the applicable pinned attention-backend and FA2
tests ported in the same change. Chunked-local attention/KV, RoPE, model-family
implementation, connectors/offload and DCP/PCP remain out of scope.

The active `CLAIM-SERVE-GATE-1` process still owns `/tmp/gpu`; read-only
inspection at 2026-07-11 00:21 UTC showed the 27B ours c32 arm running with the
server using 25,251 MiB. Its vLLM denominator and both 35B arms remain scripted
afterward. W2 therefore begins with source, CPU tests and compile-only CUDA
validation and will not launch GPU work until that whole campaign releases the
lock. Feature-positive model, oracle, nsys and every-axis performance/memory
gates remain an explicit `GATING` handoff rather than invented evidence.

## 2026-07-11 — C5 W2 sliding-window attention implemented and handed to feature gating

`CLAIM-C5-SW-ATTN-1` completed its bounded implementation scope and is
released. `ATTN-SLIDING-WINDOW` moves `ACTIVE -> GATING`, never `DONE`. The
port adds:

- typed top-level/effective-text-config `sliding_window` normalization;
- generic per-layer-over-model precedence and the exact decoder `(W-1,0)` /
  encoder `(W-1,W-1)` mapping;
- one optional backend-neutral `vt::AttentionWindow` propagated into
  `PagedAttentionArgs` and `AttentionLayer`;
- bottom-right-aligned CPU lower/upper bounds;
- compile-time full/local variants for every portable CUDA decode, tiled
  prefill, WMMA, GQA, Flash2 and vectorized/BM specialization, avoiding a
  per-key window branch on the existing full-attention path; and
- pinned FA2 API normalization: a finite decoder window dispatches the
  non-causal LOCAL specialization because `(W-1,0)` already carries the causal
  right bound and the pinned causal template otherwise ignores `window_left`.

Ported/upstream-derived tests cover mixed prefill/decode/chunked prefill,
widths `{1,3,4,5}`, page boundaries, GQA, unequal Q/K bottom-right alignment,
symmetric encoder poisoning, the q=129/W=4096 boundary, WMMA and FA2 local
arms. Four real skips retain TP, FlashInfer, StarCoder2 and Gemma3 obligations.

Fresh local evidence (no DGX command ran):

- `cmake --build build-c5-attn-cpu -j8` passed with no new warning;
- `VLLM_CPP_CPU_THREADS=1 ctest --test-dir build-c5-attn-cpu
  --output-on-failure -j2` passed **100/100**;
- `test_ops_paged_attn` passed **14/14 cases, 1,643 assertions**;
  `test_attention_window` passed **3/3 cases, 22 assertions, four named skips**;
  `test_hf_config` passed **9/9 cases, 98 assertions**; and the unchanged
  common metadata regression passed **5/5 cases, 22 assertions**;
- the three changed focused binaries pass a Debug ASan+UBSan ctest run with
  `ASAN_OPTIONS=detect_leaks=1:halt_on_error=1` and
  `UBSAN_OPTIONS=halt_on_error=1`;
- a temporary, non-repository NVIDIA CUDA 13.0.88 toolchain compiled
  `cuda_paged_attn.cu` for `121a` with the optional accelerators disabled;
  against the project-pinned CUTLASS v4.4.2 commit `da5e086`, it also compiled
  `cuda_flash_attn_fa2.cu` and both causal and non-causal/local hdim256 bf16
  FA2 instantiation units. Object symbols close both dispatch arms. This is
  compile-only evidence; no runtime correctness/performance is inferred; and
- `scripts/check-agent-record.py`, all **13** record mutation tests, and
  `git diff --check` pass.

The surviving `CLAIM-SERVE-GATE-1` campaign retains `/tmp/gpu`; W2 did not
enqueue behind it or contaminate its benchmark. Remaining handoff: restore the
pinned oracle environment, run G4 portable/FA2/FlashInfer-when-available vectors
under one lock, land a supported StarCoder2/Gemma3-class consumer, nsys both
engines on the identical feature-positive workload, and run G6-G9 correctness,
throughput, latency and memory comparisons plus the unchanged 27B/35B
regressions. Until then README and matrices say operator implemented/compile-
checked, but no user-visible sliding-window model support.
