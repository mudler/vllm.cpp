# `Qwen4ExpForConditionalGeneration` (Qwen3.8-Flash-Next)

**Campaign row:** `MODEL-MM-QWEN4-EXP` (the ID carried by the branch, the issue and
the append-only index row)
**Model-matrix target row:** `MODEL-MM-qwen4-exp-qwen4-exp-for-conditional-generation`,
the deterministic ID the row contract requires. Both name the same work; the index
row is append-only and cannot be re-keyed, so both are recorded rather than one
silently replaced.
**Issue:** [#1978](https://github.com/mudler/vllm.cpp/issues/1978)
**State:** `READY` (spec only; no product code lands under this row's first pull request)
**Motivating checkpoint:** `Qwen/Qwen3.8-Flash-Next`, released 2026-08-24, read live 2026-08-26

## Scope

Port `Qwen4ExpForConditionalGeneration` / `model_type: qwen4_exp`. The card calls it
"this experimental preview of the architecture that will underpin Qwen4"; the
`Qwen3.8` in the name is marketing continuity and not a shape relationship. It is a
180B-total / 6B-activated multimodal (image-text-to-text) hybrid: 48 layers in a
repeating `3 x linear_attention -> 1 x qwen_sparse_attention` pattern, 512-expert MoE
at top-10 plus one shared expert, a 20M-entry n-gram embedding table injected at
layer 2, a 4-branch gated residual stream, and a 1-layer MTP head.

In scope: text generation and the image/video path, every published quantized arm,
and the GGUF k-quant arms this repository requires of any model port.

Out of scope for the first implementation wave, each named under `## Owed` rather
than dropped: MTP depth > 1, the 1M-token RoPE extension the card advertises above
the native 262144, and any throughput claim.

## Why this needs a spec before code

Three of this row's decisions are expensive to reverse and cheap to get wrong, and
all three have already been made incorrectly once by an agent reading a related
record. They are settled here so a fresh implementer does not re-derive them.

1. **This is not a Qwen3.8 row.** `.agents/specs/qwen38-27b-bf16-gate.md` records
   `Qwen/Qwen3.8-27B` as the Qwen3.6-27B shape retrained, differing in exactly one
   config key. That precedent does not extend here. `qwen4_exp` shares an ancestor
   with `qwen3_5` and diverges in four load-bearing places.
2. **QSA's twin in vLLM is MiniMax-M3, not DeepSeek-V4.** See `## Design`. Building
   it on the DSA/MLA path is the wrong port, and DSA is the path an agent reaches
   for first because this tree already has it.
3. **The oracle split is a direction, not a default.** See below.

## Oracles

**vLLM implements nothing here.** Read live at `origin/main` = `6a5e8f5979`,
2026-08-26: no `qwen4*` path, no registry entry, and a repository-wide GitHub search
for `qwen4` returns zero results. `vllm-omni` likewise. This is absence from vLLM
`main`, not staleness in our pin (`555967922`), so a pin advance does not reach it.

**Developer direction, 2026-08-26: transformers is the oracle for the ALGORITHM,
vLLM supplies the OPS.** Recorded verbatim because it is the axis the whole row
hangs on: "use transformers as oracle for algorithmic side. but use ops from vllm so
we account for optimized path."

This is the correct reading of what each upstream is, and not a split of
convenience. transformers [#48337](https://github.com/huggingface/transformers/pull/48337)
(MERGED 2026-08-26, 5211 lines) is a semantics reference that says so in its own
code: `Qwen4ExpTextQSAIndexer.forward` loops in Python over `(batch_idx, query_idx)`
and carries the comment "we only allow eager and sdpa". Ported as written it yields
a correct model at an indefensible speed. AGENTS.md's "Mirror vLLM" polarity
continues to bind every primitive vLLM implements, even though vLLM has never
assembled this particular model from them.

Therefore: **every component resolves against exactly one oracle, named in the
`## Design` table. An implementer who cannot name the oracle for the line they are
writing has found a gap in this spec and returns `NEEDS_CONTEXT`.**

SGLang [#36497](https://github.com/sgl-project/sglang/pull/36497) is OPEN and is not
admissible while it stays open. Re-check it at each wave; if it merges it becomes a
second op source under the `sglang` registry id, still ranked below vLLM.

### The transformers lane pin (ACCEPTED 2026-08-26)

`.agents/oracles/transformers.md` pins transformers to **5.14.1**, deliberately tied
to what the pinned vLLM environment resolves, on the stated ground that an
independent pin "would let the oracle environment hold two different `transformers`
at once, which is the drift this registry exists to stop".

**5.14.1 does not contain `Qwen4Exp`**, so this row cannot run its algorithmic
oracle under the existing pin.

The exception argued here is narrow: the invariant guards against a vLLM environment
and its transformers drifting apart, and for `qwen4_exp` there is no vLLM
implementation to drift from. A lane-scoped second pin therefore cannot create the
inconsistency the rule exists to prevent. It is recorded in the oracle file as a
lane exception naming this row and this issue, and it expires the moment vLLM
registers `qwen4_exp`, at which point the row reconciles onto vLLM and transformers
demotes to the preprocessing role it holds everywhere else.

**Accepted by the developer on 2026-08-26**, having been put as an explicit
accept-or-reject rather than passed as housekeeping, because it changes the
semantics of a registry invariant.

**The lane pin is `transformers` 5.16.0, and it is a real release, not a branch
SHA.** That was not the expected outcome and it is better than one. `Qwen4Exp`
merged to `main` at 12:03:40Z on 2026-08-26 and `v5.16.0` was published at
12:35:15Z, 32 minutes later. Bounded rather than assumed, by fetching the model
source at each tag on 2026-08-26: `v5.16.0` returns HTTP 200 and `v5.15.0` returns
HTTP 404, so 5.16.0 is the FIRST release containing the architecture, which is the
tightest pin available. Its `auto_mappings.py` carries 5 `qwen4_exp` occurrences, so
the registration landed with the model rather than trailing it.

The version string is **unmeasured**: it is the release that provably contains the
model, not a `transformers.__version__` read off a running oracle. Resolving the
runtime string is owed to the first wave that stands one up. Full record and the
`oracle-pin-lane` block: [`../oracles/transformers.md`](../oracles/transformers.md).

### Gateability

`gateable = no` at the time of writing, and the reason is memory rather than
software: see `## Hardware`. The oracle must demonstrably build **and run the
model**, and no published artifact fits any fleet device. The first wave's real
deliverable is the arm that makes an oracle run possible at all.

## Upstream chain

| Source | Revision | Role |
|---|---|---|
| `huggingface/transformers` | **`v5.16.0`** (lane pin; first release containing `qwen4_exp`, landed by `#48337` merged 2026-08-26) | algorithm; `models/qwen4_exp/modular_qwen4_exp.py` is the authored delta, `modeling_qwen4_exp.py` the generated expansion |
| `vllm-project/vllm` | `origin/main` `6a5e8f5979` (survey only; the parity pin stays `555967922`) | ops |
| `Qwen/Qwen3.8-Flash-Next` | HF `main`, read 2026-08-26 | config and weights |

Read the **modular** file, not the generated one. It is 1186 lines against 2707 and
it is the file that states what is inherited unchanged, which is most of the model.

## Our baseline

What this tree already has, and therefore what the port does NOT rebuild. Stated
first because the delta only means something against it, and because the size of
this list is the reason the row is tractable at all.

- **The Qwen3.5 family end to end.** `src/vllm/model_executor/models/qwen3_5*.cpp`
  carries the dense and MoE backbones, the GGUF weights path, the MTP draft
  (`Qwen3_5MTPModel`) and the runner integration. `Qwen4ExpTextModel` inherits from
  `Qwen3_5MoeTextModel`, so this is the base the upstream delta is written against.
- **GDN linear attention with a Triton-AOT fast path.** `src/vt/cuda/cuda_gdn.cu`.
  The AOT specializations are pinned to `K=V=128, Hg=16, H in {48,32}` and this
  model's `linear_key_head_dim` / `linear_value_head_dim` / `linear_num_key_heads` /
  `linear_num_value_heads` are `128 / 128 / 16 / 48`. An exact hit, not a near miss.
- **A working sparse-attention indexer**, `deepseek_v4_dsa.cpp` +
  `deepseek_v4_compressor.h` + `src/vt/cuda/cuda_deepseek_v4.cu`. Useful for its
  compressor and its cache plumbing; **not** the right base for QSA's selection
  path, see `## Port map`.
- **Hyper-connection residual streams**, `deepseek_v4_mhc.cpp`, ported 1:1 from
  vLLM's `kernels/mhc/`. Different math from Gated Residual, same fused shape.
- **Interleaved mRoPE**, `layers/rotary_embedding/mrope.cpp`, which this model needs
  (`mrope_section [11, 11, 10]`, `partial_rotary_factor` 0.25 over `head_dim` 256).
- **The Qwen3.5-Moe vision tower**, which upstream reuses here **unchanged**.
- MoE with grouped GEMM, and the GGUF k-quant loader stack.

## Design

`Qwen4ExpTextModel` inherits from `Qwen3_5MoeTextModel` and leaves the rotary
embedding, MLP, experts, TopK router and the **entire vision tower** unchanged
(`class Qwen4ExpVisionModel(Qwen3_5MoeVisionModel): pass`). This tree already has all
of that. The port is the delta below.

## Port map

| Component | Algorithm oracle | Op oracle (vLLM) | This tree |
|---|---|---|---|
| GDN linear attention | `Qwen4ExpTextGatedDeltaNet` | `layers/mamba/gdn/qwen_gdn_linear_attn.py` | **HAVE.** `K=V=128, Hg=16, Hv=48` is an exact match for the AOT gate in `TryTritonPackedDecode` / the delta_h dispatch (`src/vt/cuda/cuda_gdn.cu`, pinned to `K=V=128, Hg=16, H in {48,32}`) |
| Grouped RMSNorm | `Qwen4ExpTextRMSNorm(group_size=)` | `layers/layernorm.py` `group_size` | new, small; mirror vLLM's form |
| QSA block scoring + top-k | `Qwen4ExpTextQSAIndexer` | `models/minimax_m3/common/indexer.py`, `common/ops/index_topk.py`, `common/sparse_attention.py` | new |
| QSA pooled-key build | indexer forward | `models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py` (`SparseAttnCompressNormRopeStoreC4Kernel`, carries `compress_ratio`) | partial: `deepseek_v4_compressor.h` |
| Indexer side cache | `Cache.update_indexer` | `MiniMaxM3IndexerCache`, `v1/attention/backends/mla/indexer.py` | new KV spec |
| Gated Residual | `Qwen4ExpTextGatedResidual` | `layers/mhc.py`, `kernels/mhc/*` (**different math**, same fused shape) | partial: `deepseek_v4_mhc.cpp` |
| MoE 512 / top-10 + 1 shared / intermediate 640 | `Qwen4ExpTextSparseMoeBlock` | FusedMoE, grouped GEMM | HAVE, shape change only |
| MTP, 1 layer, `hybrid: true` | config `mtp` | `qwen3_5_mtp.py` | HAVE, needs extension |
| PLE dilated depthwise conv | `Qwen4ExpTextPLELayer._short_conv` | **NONE** | new, no vLLM op |
| N-gram hashed embedding | `Qwen4ExpTextNGramEmbedding` | **NONE** | new, no vLLM op |
| Vision tower | `Qwen4ExpVisionModel` = `Qwen3_5MoeVisionModel` | qwen3_5 vision | HAVE. `deepstack_visual_indexes: []`, so no deepstack |

**Exactly two components have no vLLM op**, and they are the two where transformers
is the sole source and we author the kernel ourselves. Everything else has an
optimized vLLM form to mirror, and mirroring it is mandatory rather than optional.

### QSA maps to MiniMax-M3

DSA is an **MLA** indexer. QSA is not MLA: plain GQA, 24 Q heads, 2 KV heads,
`head_dim` 256, `partial_rotary_factor` 0.25 giving 64 rotary dims. MiniMax-M3's
lightning indexer is vLLM's **non-MLA** block-sparse case, and its docstring
describes QSA's shape exactly: it "scores KV blocks with the index heads and selects
the top-k blocks ... that the main block-sparse attention then attends to", owning
"its own side cache (one index-key vector per token)". `common/ops/index_topk.py`
supplies `_index_block_score_kernel` and a bitonic `_topk_index_kernel`.

Reconciliation the implementer owes, not hand-waved here: M3's sparse block size is
128 where QSA's `indexer_compress_ratio` is 4 with `block_topk = indexer_budget /
compress_ratio = 512`; and QSA's per-block key is a **mean pool over the block**,
then `k_layernorm`, then RoPE at the block-start position, scored as
`relu(q . k).sum(over 4 index heads) / sqrt(128)`. The pooled-key construction is
what `SparseAttnCompressNormRopeStoreC4Kernel` already does under a different name.

`indexer_kv_heads` must be 1; the upstream config validator rejects anything else.

### Two structural consequences beyond the module list

- **The residual stream is `hc_count * hidden_size` = 4 x 2560 = 10240 wide through
  the whole stack.** `Qwen4ExpTextGatedResidual` reads it through a grouped RMSNorm
  and a low-rank (`hc_lowrank` = 320) SiLU-then-sigmoid gate, collapses to 2560 for
  the block, and writes back with a per-branch scalar gate
  (`2 * sigmoid(block_inject_weight(x) / hc_count)`). This is a change to the
  per-layer loop and to every residual buffer, not a drop-in module. The
  `Qwen4ExpTextModel` also holds one `use_combine=False` mixer that collapses the
  stream at the end.
- **`number_of_conv_states = 3` on a PLE layer** (GDN conv, PLE conv, and the n-gram
  token history, which upstream stores as a third conv state precisely because the
  manipulations are identical). The KV-cache spec grows a third conv stream plus the
  indexer side cache. Adjacent to [#1963](https://github.com/mudler/vllm.cpp/issues/1963)
  and [#1966](https://github.com/mudler/vllm.cpp/issues/1966).

### The n-gram embedding is integer-exact or it is silently wrong

Head vocab sizes are successive **primes** found after `ngram_vocab_size_base - 1`
(20,000,000), indexed by a global head index; IDs are built by XOR-mixing
splitmix64-seeded per-position multipliers over shifted token windows, then reduced
modulo each head's prime. An off-by-one in the prime search, or a 32-bit truncation
anywhere in the mix, yields a valid-looking lookup into the wrong row. Nothing
downstream detects it, and a token gate against a model this size will not localise
it. Gate the ID construction directly, against transformers, before any weight is
loaded.

## Dependencies

Shared seams this row must route through rather than around, per AGENTS.md
"Shared seams". Each is named so a reviewer can check the routing instead of
inferring it.

- `ModelRegistry::Forward` and `dense_attn::AttnBlock` for decode.
- `vt::FusedChain` for model fusion; `layers::MlpGateUpMethodBase` and
  `vt::MergedGemmGroup` for the mergeable MLP projections.
- `include/vllm.h` for every shipped capability. Examples and servers stay thin
  ABI clients and never include an internal header.
- `vllm::HfConfigFromGguf` and the `qwen3_5` GGUF builder, which currently
  hard-asserts its own architecture and will refuse `qwen4_exp` by name until this
  row extends it.
- `src/vllm/v1/kv_cache_interface.*` for the third conv stream and the indexer side
  cache. This is the seam [#1963](https://github.com/mudler/vllm.cpp/issues/1963)
  and [#1966](https://github.com/mudler/vllm.cpp/issues/1966) are moving; coordinate
  rather than fork.

New files go beside their vLLM counterparts and mirror the upstream file structure,
per AGENTS.md. Where the upstream counterpart is MiniMax-M3 rather than a Qwen file,
mirror the op's home and say so in the file header.

## Work breakdown

Waves are separable and each is independently reviewable. Every wave lands reachable
from a production entry point, or names what is unreached with its owning row and
issue per AGENTS.md "Nothing lands dead".

- **W0, this pull request.** Spec, records, oracle exception proposal. No product
  code.
- **W1, config and registration.** `qwen4_exp` config resolution including the
  `full_attention` -> `qwen_sparse_attention` rewrite upstream performs in
  `__post_init__`, every `validate_architecture` rejection, and a refusal naming any
  unimplemented arm. Reachable through the loader.
- **W2, the two components with no vLLM op.** N-gram hashed embedding and the PLE
  layer with its dilated depthwise conv. Gated against transformers goldens on
  integer equality for the ID construction. First because they are the highest
  silent-wrongness risk and because they are independent of the attention work.
- **W3, gated residual.** The 10240-wide stream through the per-layer loop, both
  `use_combine` arms, and the final mixer.
- **W4, QSA.** Indexer side cache and KV spec, pooled-key build, block scoring and
  top-k, block-sparse consumer. Mirrors MiniMax-M3's op shape.
- **W5, assembly and the load plan.** Full model forward, vision path, MTP.
- **W6, the first runnable arm** and the row's real unblock: a Q4_K_M backbone with
  the n-gram table non-resident, per the developer decision in `## Hardware`. Two
  separable halves. **W6a** authors the `qwen4_exp` GGUF architecture on our side,
  because llama.cpp has none, and states in its result that these arms therefore
  have no llama.cpp oracle. **W6b** makes the 51 GB table non-resident, which on
  unified memory cannot be the existing host-pinned offload and needs its mechanism
  established first. W6b is the one with unknown cost and should be spiked before it
  is scheduled.

Waves W2 through W4 have no ordering dependency on each other and can be dispatched
in parallel to separate worktrees. W5 is a barrier.

## Hardware

Usable budget on GB10 is about 119 GB. Read live from the HF API, 2026-08-26:

| Artifact | On disk | Verdict |
|---|---|---|
| `Qwen/Qwen3.8-Flash-Next` BF16 | ~360 GB (`BF16 = 179,999,981,424` params) | no |
| `Qwen/Qwen3.8-Flash-Next-FP8` (official) | ~180 GB | no |
| `RadixArk/Qwen3.8-Flash-Next-NVFP4` | ~128 GB; NVFP4 backbone with the n-gram table kept at **FP8, 51.2 GB** | no, over budget before KV |
| `unsloth/Qwen3.8-Flash-Next-GGUF` | **README only, zero weight files** | does not exist |

No GGUF exists and no existing tool can produce one, because llama.cpp has no
`qwen4_exp` architecture either. Per AGENTS.md the quantized arms are a standing
requirement, so this row owes them and owes authoring the arch on our side.

**The architecture hands us the lever.** Its card argues n-gram embedding is "more
amenable to offloading than MoE", and the arithmetic agrees: the per-token cost is
`(ngram_size - 1) * heads_per_ngram` = 16 lookups of `ple_embed_dim / ngram_heads` =
160 dims. **51 GB of the 180 GB, 28% of the model, is a table touched 16 times per
token.** Making it non-resident is the intended design point. RadixArk reached the
same split independently.

| Arm | Backbone (125B) | N-gram (51B) | Resident | Fits |
|---|---|---|---|---|
| Q8_0 throughout | ~133 GB | ~54 GB | ~191 GB | no |
| Q4_K_M throughout | ~76 GB | ~31 GB | ~109 GB | yes, ~10 GB for KV and activations |
| **Q4_K_M backbone, n-gram table non-resident** | ~76 GB | 0 | **~76 GB** | yes, with room |

**Developer decision, 2026-08-26: the first runnable arm is the third row — a
Q4_K_M backbone with the n-gram table non-resident.** Q8_0 was raised and does not
fit: at ~191 GB it exceeds the budget by a wider margin than BF16 exceeds it on a
box half this size, and no partial-Q8 split reaches 119 GB while keeping the
backbone at 8 bits. Q4_K_M-throughout fits on paper at ~109 GB but leaves about
10 GB for KV and activations on a model whose native context is 262144, which is
not a margin. The chosen arm is also the only one that matches what the
architecture was built for, so the offload is a design point rather than a
concession.

This promotes the non-resident table from a note to a **first-class deliverable**
of W6. It is not free: GB10 is unified memory, so the existing host-pinned offload
seam (`ENG-WEIGHT-OFFLOAD`, mirroring vLLM's `cpu_offload_gb`) does not by itself
solve this there, and the mechanism has to be disk-backed or genuinely unloaded.
Establish that before designing around it.

These are sizing estimates from published parameter counts, not measurements. They
decide which arm to attempt first and nothing else. GB10 is **unified** memory, so
"offload to host" is not a move there; non-resident means disk-backed and page-cached,
and its cost is unmeasured. Establish it before it is designed around.

## Risks

- **Porting the eager reference as written.** The stated risk of the oracle split.
  The QSA indexer and the n-gram ID construction are both written as scalar Python
  in transformers. A reviewer should mutate for this: an implementation whose QSA
  path has no block-level kernel is a correctness result, not a port.
- **Reaching for DSA.** This tree has a working DSA indexer, and it is the wrong
  base. See above.
- **Silent n-gram mis-indexing.** See above.
- **Sizing estimates hardening into measurements.** The `## Hardware` table is
  arithmetic on published counts. `.agents/` already records this failure mode
  (a quoted number becoming a measured one); do not let the 76 GB row be cited as
  an observation.
- **A GGUF arm with no oracle.** llama.cpp does not implement `qwen4_exp`, so the
  usual quant-arm cross-check against a quant-matched llama.cpp does not exist. A
  k-quant arm here can only be gated against our own higher-precision path, which
  is a weaker gate, and the spec must say so rather than imply parity.
- **`transformers_version: 5.8.0.dev0`** in the published config is older than both
  our pin and the branch that merged `Qwen4Exp`. It records the branch the config
  was authored on and is not a usable pin. Do not resolve the oracle from it.

## Tests to port

AGENTS.md requires the upstream tests in the same change, preserving parameters,
modes, fixtures, tolerances, failure cases and the revision anchor. The upstream
suite is `tests/models/qwen4_exp/test_modeling_qwen4_exp.py` at transformers #48337,
707 lines, two classes. Inventory, read live 2026-08-26:

**`Qwen4ExpTextModelTest`** (`Qwen4ExpTextModelTester`, a `CausalLMModelTester`):

| Upstream case | Ports to | Note |
|---|---|---|
| `test_ple_layers_must_use_linear_attention` | W1 | a config invariant; cheap and load-bearing |
| `test_ple_padding_and_static_cache_match_unpadded_sequence` | W2 | the padding/EOS-segment semantics of the n-gram history |
| `test_all_layer_types_cached_forward_match_full_forward` | W4/W5 | cached vs full forward across BOTH layer types; this is the incremental-decode gate |
| `test_ple_beam_generation` | W5 | PLE under beam search, where the conv and n-gram states must follow the beam |
| `test_ple_sharded_checkpoint_loads_and_forwards` | W5 | 131 shards here, so sharded load is not optional |
| `test_generate_with_ple_and_inputs_embeds` | W5 | drives `reverse_embedding`, the inputs-embeds path |
| `test_reverse_loading_mapping` | W1 | weight-name mapping both directions |
| `test_attention_outputs`, `test_hidden_states_output` | W3/W4 | both are OVERRIDDEN upstream because the hyper-connection stream changes the shapes; port the override, not the base |
| `test_tp_plan_matches_params` | not ported | tensor-parallel plan; no TP surface in this row |
| `test_generate_compile_model_forward_fullgraph`, `test_generate_compilation_all_outputs`, `test_multi_gpu_data_parallel_forward`, `test_generate_with_quant_cache` | not ported | torch.compile / multi-GPU / torch quant-cache harness, no counterpart here |

**`Qwen4ExpCompositeModelTest`** (`Qwen4ExpVisionText2TextModelTester`, a
`VLMModelTester`): `test_mismatching_num_image_tokens`, `test_video_forward`,
`test_composite_checkpoint_loads_as_causal_lm`,
`test_base_model_checkpoint_loads_as_conditional_generation`,
`test_generate_with_ple_and_inputs_embeds`, plus its own `test_attention_outputs` /
`test_hidden_states_output` overrides. All port to W5. The remaining cases in that
class are the same harness-only skips as above.

Adaptations must be documented per AGENTS.md, and only where genuinely unavoidable.
"Our harness differs" is not one; "upstream asserts against a `torch.compile`
fullgraph we do not have" is.

### Local red-first tests

Red-first, smallest failing test per slice, each entering through a production entry
point per AGENTS.md "Nothing lands dead". A unit test that constructs the type by
hand does not discharge this.

1. N-gram ID construction against transformers goldens: prime head vocab sizes, the
   splitmix64 multipliers, the shift-and-XOR mix, EOS segment handling. Integer
   equality, no tolerance.
2. Grouped RMSNorm against vLLM's `group_size` form.
3. Gated Residual forward against transformers, both `use_combine` arms.
4. QSA block selection: selected token index sets equal to transformers on the same
   inputs, including the ragged tail beyond the last complete block.
5. PLE layer end to end, including the dilated depthwise conv and its state.
6. Config resolution: the `full_attention` -> `qwen_sparse_attention` rewrite that
   upstream `__post_init__` performs, and every rejection in `validate_architecture`.
7. Loader coverage against the published index, with the refusal path naming any
   unimplemented arm.
8. Inertness: existing Qwen3.5/3.6/3.8 goldens byte-identical.

## Gates

No token gate is claimable until an arm runs. In order:

1. **G0, component goldens.** Tests 1-6 above against transformers at the lane pin.
   This is the only gate reachable today, and it is reachable without the weights.
2. **G1, load plan.** Every published tensor accounted against a committed manifest,
   per arm, with refusals naming what is missing.
3. **G2, token-exact greedy** vs transformers at the lane pin, on whichever arm
   `## Hardware` makes runnable first. Strict token equality; the near-tie
   distributional doctrine applies only if the oracle's greedy decode is shown
   non-deterministic, which is not assumed here.
4. **G3, quantized arms.** Per arm, with the lower-bound requirement this repository
   places on quantized gates, and with the missing-llama.cpp-oracle limitation stated
   in the result rather than omitted.
5. **Speed: nothing.** No throughput, latency or memory number is admissible from
   this row until G2 passes. There is no vLLM denominator for this model, so when a
   speed axis does open, the spec must first say what the denominator is.

## Evidence required

Per gate: the exact build and run recipe, the lane transformers revision, the
checkpoint repo **and revision** plus sha256 for any quantized artifact, the device,
and the contention state. `docs/USAGE.md` gains the checkpoint pins in the same
change that makes any arm reachable, not later.

## Stop conditions

- vLLM registers `qwen4_exp`: **stop and reconcile onto vLLM** before continuing.
  This is the designed end of the transformers exception.
- SGLang #36497 merges: re-survey the op mapping; it does not displace vLLM.
- The transformers lane pin is rejected in review: the row holds at `READY` and the
  gate stays `PENDING`. Do not proceed on an unpinned oracle.
- No arm is made to fit any fleet device: the row holds with G0 passed and G1-G3
  `PENDING` on hardware, recorded as visible debt, and no token claim is made.

## Owed

- [#1978](https://github.com/mudler/vllm.cpp/issues/1978): this port. No product
  code lands under the spec pull request.
- GGUF k-quant arms, including authoring the `qwen4_exp` architecture on our side,
  and the statement that no llama.cpp oracle exists for them.
- MTP depth > 1.
- The 1M-token RoPE extension above the native 262144.
- The non-resident n-gram table: its mechanism, and a measurement of its cost.
- A speed denominator, once one exists.

## Now

`READY`. Spec committed, no implementation.

Both decisions this spec was blocked on are **settled** (developer, 2026-08-26) and
recorded in place rather than left as proposals: the transformers lane pin is
ACCEPTED at 5.16.0 (`## Oracles`), and the first runnable arm is the Q4_K_M backbone
with a non-resident n-gram table (`## Hardware`).

Next actions, in order: W0 lands this spec; W1 through W3 are reachable today
against the lane pin with tiny random configs and need neither a checkpoint nor a
GPU lease; W6b's mechanism is the unknown that decides whether the chosen arm is
schedulable, and it should be spiked before W6 is planned.
