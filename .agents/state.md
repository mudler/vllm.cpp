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
