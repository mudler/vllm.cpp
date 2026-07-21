# Breadth sweep plan — CUDA-arch additivity re-audit + recent-models sweep

Status: **SPIKE (draft)**. Owner: unassigned. Roadmap wiring: this spec is the
active execution plan for **`ROAD-V1-C2`** (model families, the actionable half)
and **`ROAD-V1-D1`** (NVIDIA target fan-out, the HW-blocked half). It answers the
user breadth-sweep directive recorded in
[[extensibility-first-additive-hw-models]] steps 1–2: *(1) re-audit the expansion
framework for CUDA copy-paste additivity; (2) sweep models + devices, recent
architectures first.*

Pins: vLLM `/home/mudler/_git/vllm` @ `e24d1b24`; local tree at HEAD `a875397`.
dgx `~/venvs/vllm-oracle` = vLLM **0.25.0** (confirmed runnable oracle).
The PR-#4 standard: the RTX 5070 sm_120 bring-up that SCATTERED +286 into
`qwen3_5.cpp`, +92 into `cuda_sample.cu`, +59 into `runner.cpp` — the
anti-pattern this audit measures against.

---

## DELIVERABLE A — CUDA-arch additivity audit

### A.0 Verdict (headline)

The framework has **two layers with opposite additivity**:

- **Policy/selection layer** (`src/vllm/platforms/*`, `src/vllm/v1/attention/registry.*`,
  the model registry, CMake arch passthrough): **clean and additive**, and already
  anticipates other archs. The PR-#4 debt (scattered model/runner/sampler edits) is
  **paid** here.
- **Compute-kernel layer** (`src/vt/cuda/*.cu`): a **single-arch monolith**. Arch is
  fixed at compile time to `sm_121a`; the fast quant/attention math is arch-specific
  PTX / CUTLASS `ArchTag` behind `#if` guards; there is **zero runtime SM dispatch**
  in any first-party kernel. A *different-family* arch (Hopper sm_90, Blackwell-DC
  sm_100, Ampere sm_80) needs new kernel bodies + a selector that does not exist —
  **not** copy-paste.

Consequence for the user's question ("does adding CUDA copy-paste?"):
- **Same family as GB10 (sm_120 consumer Blackwell): YES, near-additive now.** The
  fp4 PTX is already `sm_120a/121a`; the CMake guards already match `12[01]a`; no
  kernel/model/runner edit is required — the PR-#4 test PASSES.
- **Different family (sm_70…sm_100): NO.** Multi-file kernel + build work, AND
  **untestable** on our hardware (only GB10 sm_121 exists on dgx). This is the
  HW-blocked device-fan-out, correctly sequenced behind the model sweep.

### A.1 The seams, arch-by-arch

**Platform seam — ADDITIVE (data, not edits).**
`include/vllm/platforms/interface.h:82-125` (`class Platform`) +
`src/vllm/platforms/cuda.cpp:15-104` (`CudaPlatform`). Compute capability is probed
once at static registration (`cuda.cpp:88-91`, `cudaDevAttrComputeCapabilityMajor/Minor`)
and is the **single source of truth** via `get_device_capability()` (`cuda.cpp:24`)
/ `has_device_capability()` (`src/vllm/platforms/platform.cpp:17-18`). A new GPU is a
new `DeviceCapability` value — **no new Platform subclass is even required** for a
CUDA GPU; `CudaPlatform` already parameterizes on `(major,minor)`.

**Attention-backend priority — ADDITIVE (already has the cross-arch branch).**
`src/vllm/platforms/cuda.cpp:64-71` is the **one clean runtime arch branch in the
whole codebase**: `if (cap_.major == 10)` returns a FLASHINFER-first list, else
FLASH_ATTN-first — a faithful port of `vllm/platforms/cuda.py::_get_backend_priorities`.
The selector `src/vllm/v1/attention/registry.cpp` (`SelectAttentionBackendName`)
walks the list and returns the first *registered* backend. Adding an arch's backend
= one self-registering TU + one priority-list slot, zero selector/model/runner edit
(`BACKEND-ATTN-REGISTRY`, landed).

**Residency policy — ADDITIVE (folds the PR-#4 discrete-memory debt into data).**
`interface.h:43-75` + `cuda.cpp:45-51`. A discrete GPU sets different
`ResidencyPolicy` values (host-free, pool cap) and **zero model code changes**
(`BACKEND-PLATFORM` item 2, landed; consumed in `qwen3_5.cpp`).

**Model registry — ADDITIVE.** `MODEL-FACTORY-registry` (self-registration
`REGISTER_VLLM_MODEL`, `model_registry.h:167-189`) — orthogonal to arch, but it is
the seam that made the *model* sweep additive (Deliverable B).

**CMake gencode passthrough — PARTLY ADDITIVE.** `CMakeLists.txt:37` is a single
CACHE STRING `"121a"`; `:67` feeds it straight to `CMAKE_CUDA_ARCHITECTURES`. Adding
a numeric target to that list is a **one-line, fat-binary-correct** change — CMake
emits SASS/PTX per listed arch. This part is additive.

**Triton AOT per-arch layout — DESIGNED additive, one slot populated.**
`cmake/TritonAOT.cmake:63-74` already lays out `triton_aot_vendored/<arch>/` subdirs
with the explicit comment "sm_90/sm_80/gfx* slots in later". A new arch = a new
`triton_aot_vendored/sm_XX/` dir (additive) — but the cubins must be **regenerated on
that HW** (maintainer path, `VLLM_CPP_TRITON_REGEN`), so it is additive-but-blocked
without the device.

### A.2 SEAM-GAP list (shared code a new *different-family* arch must TOUCH but shouldn't)

1. **CMake arch-family compile guards are hardcoded `MATCHES "12[01]a"`** — 4 sites:
   `CMakeLists.txt:71` (`VT_FP4_MMA_SM120A`), `:88`/`:97` (CUTLASS NVFP4/FP8 TUs),
   `:137` (vendored Marlin MoE). A build that adds e.g. `90a` to the arch list
   **silently disables** fp4/fp8-cutlass/Marlin for *all* archs unless these guards
   are generalized to a capability→feature map. **FIX (cheap, do before the device
   fan-out): replace the 4 literal `12[01]a` regexes with a per-arch feature table.**
2. **No runtime SM dispatch in `src/vt/cuda/`** — the deepest gap. The quant/attention
   math is single-arch: NVFP4 uses `mma.sync…kind::mxf4nvf4` PTX
   (`cuda_matmul_nvfp4.cu:2119`, sm120a-only, behind `#if VT_FP4_MMA_SM120A`
   `:1505,1802,2112`); FP8 bakes `ArchTag=cutlass::arch::Sm120`
   (`cuda_matmul_fp8_cutlass.cu:164`); FA2 is "all instantiations compiled for
   sm_121a" (`cuda_flash_attn_fa2.cu:428`). Host launchers
   (`LaunchFp4Fp4` `cuda_matmul_nvfp4.cu:2112`, the fp8/flash launchers) take **no
   capability argument** and have **no `if(sm==)` selector**. A second-family arch
   needs new kernel bodies (Hopper wgmma / sm_100 tcgen05 — note `cuda_gdn.cu:3355`
   "sm_121 has no tcgen05/tmem") **plus** a new runtime selector that does not exist.
3. **`cuda_paged_attn.cu` WMMA ladder hardcodes GB10 assumptions** — the tensor-core
   path is gated to `d==256` (`:2529`) and the 100 KiB opt-in shared-memory ceiling
   is a hardcoded comment-assumption (`:870,:2401`), not a queried
   `cudaDevAttrMaxSharedMemoryPerBlockOptin`. A different arch's smem/MMA shape needs
   new branches in `LaunchPaged` (`:2513-2704`).
4. **The kernel layer can't see the capability source of truth.** `cuda_backend.cu`
   carries no SM state (it probes only `cudaDevAttrPageableMemoryAccess`,
   `:255-266`); the authoritative `(major,minor)` lives on `CudaPlatform`. Any runtime
   kernel selector (gap #2) first needs the capability *threaded to the kernel
   layer* — today it is not.

`__CUDA_ARCH__` scatter is **19 hits, all in vendored trees** (marlin/*, flash_attn/*)
— upstream Turing/Ampere guards, inert on our single-arch build; **zero** in
first-party `vt/cuda/*.cu`. So the scatter risk is not stray `#ifdef`s in our code;
it is the *absence* of a runtime dispatch layer (gaps #1–#4).

### A.3 The ADDITIVE-CUDA-ARCH bring-up template

For a **same-family** target (sm_120 consumer Blackwell — the realistic next arch):
1. Add the numeric target to `CMakeLists.txt:37` (`"120a;121a"`) — one line, fat binary.
2. `triton_aot_vendored/sm_120a/` — regenerate cubins on the device (maintainer path).
3. Nothing else: Platform auto-probes `(major,minor)`; attn-priority/residency are
   data; the fp4 PTX + CMake `12[01]a` guards already cover 120; model/runner/sampler
   untouched. **Zero kernel/model edits.**

For a **different-family** target (sm_90/sm_100/sm_80 — the HW-blocked case), the
bring-up is NOT copy-paste; it ADDS, per arch:
- a gencode target (`CMakeLists.txt:37`) — additive;
- **generalized CMake feature guards** (fix seam-gap #1 once, for all archs);
- **per-arch kernel tactic bodies** — new `VT_FP4_MMA_SM100A`/`_SM90` NVFP4 path,
  new `ArchTag` FP8/CUTLASS configs, new FA2 per-arch instantiations,
  new WMMA/smem branch — this is the real, multi-file work (seam-gaps #2, #3);
- **a runtime SM selector** in the host launchers + capability threaded to the
  kernel layer (seam-gaps #2, #4) — build this once, then each arch registers a tactic;
- a Platform entry: **none needed** (CudaPlatform is capability-parameterized);
- an attention-priority slot only if the arch prefers a different backend order
  (`cuda.cpp:64-71`, data).

### A.4 PR-#4 test judgment + testability

**Does the next arch touch fewer shared files than PR #4?**
- **sm_120 (same family): YES, decisively.** PR #4's +286 `qwen3_5.cpp` / +92
  `cuda_sample.cu` / +59 `runner.cpp` scatter is **eliminated** — the model/runner/
  sampler are now model-shape-agnostic and Platform-driven (landed seams). sm_120 is
  ~2 additive touches (arch string + triton dir), zero shared-code edits. PR-#4 test
  passes.
- **Different family: PARTIAL.** The policy/model/runner scatter is gone, but the
  kernel-layer scatter is *replaced by an absence* — you must build the runtime
  selector + per-arch tactics. Fewer files than PR #4 in the model/engine layer;
  MORE net work in the kernel layer (which PR #4 never even attempted for a new
  family). Honest: cross-family is a kernel campaign, not an additive drop-in.

**Testability of the 13 vLLM CUDA targets on our HW:**
- **Testable (executable):** GB10 **sm_121** only (dgx). This is the sole arch with
  execution evidence; every other target in `backend-matrix.md` is `INVENTORIED`
  (build availability ≠ runtime support, backend-matrix line 9).
- **Additive-but-near-free, untestable:** **sm_120** (same family; builds + should run
  on any sm_120 board, none on dgx).
- **Additive-but-untestable (build/gencode-only, needs the kernel campaign):**
  sm_70,75,80,86,87,89 (Volta→Ada), sm_90 (Hopper), sm_100,101,103,110 (Blackwell-DC).
  These are the `ROAD-V1-D1` device fan-out — **HW-blocked for validation**; a green
  fatbinary link is *not* execution evidence.

**Recommendation:** the CUDA *device* sweep is HW-blocked; do the two cheap seam-gap
fixes opportunistically (generalize the `12[01]a` CMake guards → capability feature
map; add a stub capability-threaded kernel-selector shape so future tactics register
additively), but **the actionable breadth on GB10 is the MODEL sweep** (Deliverable
B). This matches [[extensibility-first-additive-hw-models]] "HW-prioritized".

---

## DELIVERABLE B — recent-models sweep plan (the actionable breadth)

### B.0 What is already proven on GB10

- **Qwen3 dense** (`Qwen3ForCausalLM`) — `MODEL-TEXT-qwen3-qwen3-for-causal-lm`
  `ACTIVE`: correctness COMPLETE (0.6B + 4B near-tie gate 16/16); c1 effective
  every-axis parity, c8 decode residual open. This is the **template** for every
  dense bring-up: the runner is now model-shape-agnostic
  (`runner.cpp:458-470,651-680`), d128 FA2 prefill+decode landed, BF16 attention
  numerics mirror vLLM, and the fusion catalog + model registry are additive.
- **Qwen3.6 27B/35B** (`Qwen3_5*`) — the NVFP4 + GDN + MoE machinery
  (`qwen3_5_common`, Marlin NVFP4 W4A16 MoE, GDN linear-attention) already exists.

So the additive substrate for the sweep is: dense-BF16 (done), NVFP4 dense/MoE
(done for the gate), GDN-hybrid (done for 35B).

### B.1 Checkpoint + oracle inventory on dgx (verified 2026-07-21)

Runnable oracle: `~/venvs/vllm-oracle` = **vLLM 0.25.0** (loads every arch below).

| Checkpoint (present) | `architectures` | Shape | Quant | Maps to registry |
|---|---|---|---|---|
| `Qwen--Qwen3-0.6B` | `Qwen3ForCausalLM` | 28L dense | none/bf16 | `qwen3` — **DONE** |
| `Qwen--Qwen3-4B` | `Qwen3ForCausalLM` | 36L dense GQA 32/8 | none/bf16 | `qwen3` — **DONE** |
| `RedHatAI--Qwen3-32B-NVFP4A16` | `Qwen3ForCausalLM` | 64L dense | compressed-tensors NVFP4A16 (W4A16) | `qwen3` + new quant loader — **RE-VERIFIED 2026-07-21: this row's "present" claim is CORRECT** (5 real shards / 20.6 GB / 1603 tensors / index / tokenizer; oracle loads and generates). Bring-up landed W0-W3, strict gate GATING 4/6 — see [sweep-qwen3-32b-nvfp4a16](sweep-qwen3-32b-nvfp4a16.md) |
| `Qwen--Qwen3-Coder-30B-A3B-Instruct` | `Qwen3MoeForCausalLM` | 48L, 128 experts | none/bf16 | `qwen3_moe` (`registry.py:192`) |
| `facebook--opt-125m` | `OPTForCausalLM` | 12L dense | none | `opt` (`registry.py:176`) |
| `unsloth--Qwen3.6-27B-NVFP4` | `Qwen3_5ForConditionalGeneration` | dense+GDN | compressed-tensors NVFP4 | gate — **DONE** |
| `nvidia--Qwen3.6-35B-A3B-NVFP4` | `Qwen3_5MoeForConditionalGeneration` | 40L, 256 experts | modelopt NVFP4 | gate — **DONE** |

> **VERIFICATION NOTE (2026-07-21).** The "present" column of this table is NOT
> self-certifying and must be re-checked before any row is started. It was WRONG
> for OPT — listed as present when only a 36 KB `config.json` existed — which cost
> a materialization detour on that row. Each row now records its own re-verification
> when it is picked up: `RedHatAI--Qwen3-32B-NVFP4A16` was re-verified complete on
> 2026-07-21 (rank 3) and the claim held. Verify weight FILES, not just a config.

Also on disk but **metadata-only (no snapshot `config.json` resolved → not fully
downloaded / unusable as-is):** `RedHatAI/Qwen3-32B-NVFP4`, `Ornith-1.0-35B`,
`Jackrong/Qwopus3.6-35B-A3B-Coder`, `Qwen-AgentWorld-35B-A3B`, `nex-agi/Nex-N2-mini`,
`InternScience/Agents-A1`. Treat as absent until a config is confirmed; several are
likely fine-tunes of known archs (verify `architectures` before queuing).

### B.2 Additivity classification of the sweep candidates

Given the seams NOW (dense runner-generalization + d128 FA2 + NVFP4/GDN/MoE gate
machinery + fusion catalog + model registry all landed):

- **Near new-files-only (dense, RoPE, GQA):** `Qwen3` (done), `Llama`, `Mistral`,
  `Qwen2`, `GLM4`, `Olmo2/3`, `Gemma2/3` (adds gemma-RMSNorm — already have it — +
  logit softcap + sliding-window). These reuse the Qwen3 dense forward almost verbatim.
- **MoE (adds FusedMoE grouped GEMM routing):** `Qwen3Moe` — the `qwen3_5_moe` path
  exists; the delta is a BF16 (un-quantized) grouped-GEMM expert path vs the gate's
  NVFP4-Marlin path, config parse (128 experts, top-k), and dense (non-GDN) attention.
- **Quant-on-dense (adds a loader/compute scheme to a done forward):**
  `Qwen3-32B-NVFP4A16` — same dense forward, adds compressed-tensors **NVFP4A16
  (W4A16, bf16 activations)**; we have NVFP4 kernels but must wire the A16
  (bf16-activation) variant + compressed-tensors loader.
- **Cross-family canary (genuinely different, no RoPE/GQA):** `OPT` — learned absolute
  position embeddings, attention bias, no qk-norm. Tiny + fast oracle; the truest test
  that "a new *family* is new-files-only". Not recent, but the cheapest cross-family
  additivity proof.
- **Hybrid / more shared work (defer):** `Qwen3Next` (GDN+MoE hybrid — additive given
  35B GDN, but download-needed), `NemotronH`/`Jamba`/`Lfm2` (Mamba/SSM state — not yet
  built), `DeepseekV3/V4`/`MiniMaxM3`/`Sarvam*` (**MLA** — new attention, new campaign),
  `GptOss`/`MiniMaxM2`/`Glm4Moe` (MoE + sliding-window).

### B.3 RANKED SWEEP QUEUE

Bar for every row (the Qwen3 standard): near-tie-robust token-exact vs the vLLM 0.25.0
teacher-forced oracle ([[near-tie-distributional-gate]]) **AND** vLLM-speed on every
axis; regression-preserve 27B 235/235 + 35B 315/315; `-Werror` 0-warn + memcheck 0.

**Tier 1 — checkpoint-present, most-additive, do first (all runnable on GB10 today):**

| Rank | Model / arch | Checkpoint | Why first | Rough effort |
|---|---|---|---|---|
| 1 | **Qwen3-4B / 32B dense speed-close** (`Qwen3ForCausalLM`) | present, DONE-correctness | Finish the OPEN c8 decode-GEMM residual on the *already-correct* model before adding new archs — it de-risks every dense row's speed bar. Also promote 0.6B→32B (64L) dense as a size-scale check. | small (in-flight sub-campaign) |
| 2 | **Qwen3-Coder-30B-A3B** (`Qwen3MoeForCausalLM`) | present (48L/128E, bf16) | Most *recent* + present arch not yet ported; MoE path (`qwen3_5_moe`) exists → mostly config-parse + BF16 grouped-GEMM expert wiring + dense attention (reuse Qwen3). The flagship recent-MoE additive proof. | medium |
| 3 | **Qwen3-32B-NVFP4A16** (`Qwen3ForCausalLM` + W4A16) | present (64L, compressed-tensors) | Same *done* dense forward; adds the compressed-tensors NVFP4A16 loader + bf16-activation W4A16 compute → proves quant-scheme additivity on a new-size dense with zero forward risk. | medium |
| 4 | **OPT-125m** (`OPTForCausalLM`) | present (12L) | Cross-*family* additivity canary: no RoPE, learned pos-emb, attn bias. Tiny → fast oracle, cheap to gate. Proves "new family = new files" honestly. | medium |

**Tier 2 — download-needed, recent dense/hybrid, genuine cross-family breadth:**

| Rank | Model / arch | Download | Additivity | Effort |
|---|---|---|---|---|
| 5 | **Llama-3.x** (`LlamaForCausalLM`) | Llama-3.2-1B/3B | near new-files-only (dense RoPE GQA, no qk-norm) — the roadmap's explicit "Llama-first W-next" | small–medium |
| 6 | **Mistral** (`MistralForCausalLM`) | Mistral-7B-v0.3 | dense, optional sliding-window | small–medium |
| 7 | **Gemma3** (`Gemma3ForCausalLM`) | gemma-3-1B/4B | dense + gemma-RMSNorm (have it) + logit softcap + sliding-window | medium |
| 8 | **GLM4 / Olmo2-3** (`Glm4/Olmo2ForCausalLM`) | small variants | dense, recent | small–medium |
| 9 | **Qwen3-Next** (`Qwen3NextForCausalLM`) | Qwen3-Next-80B-A3B (large) | GDN+MoE hybrid — additive given 35B GDN, but large + hybrid | medium–large |

**Tier 3 — architecture campaigns (shared work, sequence after Tier 1–2):**
MLA family (`DeepseekV3/V4`, `MiniMaxM3`, `Sarvam*`) = new attention; Mamba/SSM
hybrids (`NemotronH`, `Jamba`, `Lfm2`, `Plamo2`) = new state kernel; large MoE +
sliding-window (`GptOss`, `MiniMaxM2`, `Glm4Moe`). Each needs its own leaf spike.

### B.4 Per-row protocol (mirror the Qwen3 bring-up)

W0 config+registry stub (`REGISTER_VLLM_MODEL` TU) → W1 any runner/KV-shape
generalization forced out (should be none for dense now) → W2 weight loader (mirror
vLLM `packed_modules_mapping`) → W3 forward from vt:: ops + fusion catalog → W4
near-tie-robust correctness gate vs the 0.25.0 oracle → W5 speed close vs graphed
vLLM. Port the matching upstream test module each step (test-porting protocol).

---

## Sequencing recommendation (for roadmap_v1)

1. **Actionable now (GB10, `ROAD-V1-C2`):** Tier-1 model sweep, in rank order.
   Start by closing Qwen3's open c8 decode residual (de-risks the speed bar), then
   Qwen3-Coder-30B MoE, then NVFP4A16 dense, then the OPT cross-family canary.
2. **Cheap seam-gap fixes (fold into the first quant/MoE row):** generalize the 4
   `CMakeLists.txt` `12[01]a` guards to a capability→feature table so a future
   multi-arch build does not silently drop fp4/fp8/Marlin.
3. **HW-blocked (`ROAD-V1-D1`, do not gate the sweep on it):** the CUDA device
   fan-out beyond sm_121 is untestable on dgx. sm_120 is near-free additive; every
   other target needs the kernel-tactic campaign (runtime SM selector + per-arch
   bodies) AND hardware to validate. Record as inventory, not as blocked model work.
