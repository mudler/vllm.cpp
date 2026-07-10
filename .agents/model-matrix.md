# Model parity matrix

**Pinned oracle:** vLLM `e24d1b24` (`vllm/model_executor/models/registry.py`).
**Detailed inventory:**
[specs/model-family-inventory.md](specs/model-family-inventory.md).

This is the top-level model-family status surface. It deliberately stays
small enough to coordinate parallel work. The detailed inventory owns the
complete one-row-per-upstream-target tables, stable `MODEL-*` work IDs,
architecture aliases, source anchors, spike state, implementation state, and
owner. Do not add a model claim here without adding or updating its detailed
row in the same change.

## Coordination contract

Every model target follows this state machine:

1. **Inventory:** the target exists as a stable `MODEL-*` row grounded in the
   pinned upstream registry.
2. **Spike:** coverage work is claimed and committed in the row's spec. A
   completed spike identifies the whole upstream
   execution chain, reusable vllm.cpp layers, missing kernels/backends, weight
   mapping, quantization interactions, and the exact upstream tests to port.
3. **Implementation:** `INVENTORIED -> SPIKE -> READY -> ACTIVE -> GATING -> DONE`.
   An agent may start implementation only after the row is `READY`; it records
   its agent/branch in `Owner` before code
   work begins.
4. **Evidence:** implementation becomes `DONE` only after merge plus correctness,
   performance, and applicable backend gates. The row must cite exact local
   code and test anchors. A family name, compatible-looking config, or shared
   layer stack is not support evidence.
5. **Completion:** when every row in a category is `DONE`, move that category's
   detailed closed record and superseded per-target specs to `.agents/completed/`
   in the same change. Keep this summary row as `✅` with a link to the archived
   evidence. No completed record remains load-bearing for live work.

Only one owner may claim a `MODEL-*` ID at a time. Parallel work should split
on IDs or dependency blocks, not edit the same target. An ownership claim or
state transition updates this table, the detailed inventory, the roadmap, and
the feature matrix in the same change under the project protocol.

## Top-level table

`IDs` counts category memberships; architectures with multiple tasks appear in
multiple categories. `Targets` groups aliases that resolve to the exact same
upstream `(module, class)` implementation.

| Block | Category | IDs | Targets | Upstream | Principal dependency fronts | State | Detailed section |
|---|---|---:|---:|---|---|---|---|
| `MODEL-TEXT` | Text generation | 130 | 114 | `registry.py:71-208` | dense attention; FusedMoE; Mamba/GDN; MLA/DSA; model-specific kernels | `INVENTORIED` (0/130 static IDs) | [Text generation](specs/model-family-inventory.md#model-text---text-generation) |
| `MODEL-EMBED` | Embedding | 40 | 28 | `registry.py:210-262` | encoder/bidirectional attention; sequence and token poolers; MM encoders | `INVENTORIED` | [Embedding](specs/model-family-inventory.md#model-embed---embedding) |
| `MODEL-LATE` | Late interaction | 11 | 9 | `registry.py:264-278` | token embeddings; ColBERT projection/pooling; MM encoders | `INVENTORIED` | [Late interaction](specs/model-family-inventory.md#model-late---late-interaction) |
| `MODEL-REWARD` | Reward | 3 | 3 | `registry.py:280-284` | sequence/process reward heads and pooling | `INVENTORIED` | [Reward](specs/model-family-inventory.md#model-reward---reward) |
| `MODEL-TOKCLS` | Token classification | 4 | 4 | `registry.py:286-300` | encoder-only attention; per-token heads/pooling | `INVENTORIED` | [Token classification](specs/model-family-inventory.md#model-tokcls---token-classification) |
| `MODEL-SEQCLS` | Sequence classification | 10 | 9 | `registry.py:302-329` | backbone; sequence pooler/classifier; MM encoders | `INVENTORIED` | [Sequence classification](specs/model-family-inventory.md#model-seqcls---sequence-classification) |
| `MODEL-MM` | Multimodal | 114 | 106 | `registry.py:331-580` | image/audio/video processors; encoder attention/cache; merger/MRoPE; ASR | `PARTIAL` (2 wrappers text-only; 0 full MM) | [Multimodal](specs/model-family-inventory.md#model-mm---multimodal) |
| `MODEL-SPEC` | Speculative draft models | 44 | 36 | `registry.py:582-633` | MTP; EAGLE/EAGLE3; DFlash; DSpark; Medusa; target-family kernels | `READY` for Qwen3.5 MTP/DFlash; others inventoried | [Speculative](specs/model-family-inventory.md#model-spec---speculative-decoding-models) |
| `MODEL-HFALIAS` | Static Transformers aliases | 4 | 2 | `registry.py:635-645` | generic Transformers causal/MM wrappers | `INVENTORIED` | [Transformers aliases](specs/model-family-inventory.md#model-hfalias---static-transformers-aliases) |
| `MODEL-HFBACKEND` | Generic Transformers backend | 10 | 10 | `registry.py:647-680` | dense/MoE causal; MM; embedding; sequence classification adapters | `INVENTORIED` | [Transformers backend](specs/model-family-inventory.md#model-hfbackend---generic-transformers-backend) |
| `MODEL-HFDYNAMIC` | Dynamic Transformers compatibility | unbounded | capability-driven | `registry.py:1096-1164` | compatible Transformers or remote `auto_map` classes; C++ policy/factory | `INVENTORIED`; factory policy spike required | [Dynamic backend](specs/model-family-inventory.md#dynamic-transformers-compatibility) |

## Count invariants

The static tables must retain these pin-specific invariants:

| Measure | Expected |
|---|---:|
| Category memberships | 370 |
| Unique registered architecture IDs after `_VLLM_MODELS` merge | 353 |
| Category/implementation target rows | 321 |
| Unique `(module, class)` targets | 307 |
| Unique implementation modules | 258 |
| Architecture IDs present in more than one task category | 17 |

Any upstream-sync change that moves these values must regenerate the detailed
inventory, explain the delta, add or retire stable rows, and update this table
in the same commit. The generic Transformers compatibility surface is dynamic
and intentionally excluded from the finite static counts.

## Execution order

This preserves the accepted roadmap sequence while exposing independent work:

1. Spike the generic architecture-to-factory contract and reject unsupported
   architectures deterministically.
2. Port dense shared-stack families: Llama-compatible, Qwen2/3, Mistral,
   Gemma, and Phi.
3. Port MoE shared-stack families: Mixtral, Qwen2/3-MoE, GLM4-MoE, and OLMoE.
4. Port Qwen3-Next while Qwen3.5 MTP and DFlash proceed in parallel.
5. Port remaining Mamba/GDN hybrids, then MLA/DSA families.
6. Port embedding, late-interaction, reward, and classification fronts.
7. Complete the Qwen3.5 gate-model vision wrappers, then broader vision,
   audio, video, OCR, and encoder-decoder/ASR families.
8. Define and gate generic/dynamic Transformers compatibility.
