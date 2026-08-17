# NemotronH end-to-end through the public ABI — the runner's recurrent half is Qwen3.5-shaped

**Issue:** [#810](https://github.com/mudler/vllm.cpp/issues/810).
**Parent row:** `MODEL-TEXT-nemotron-h-nemotron-hfor-causal-lm`
([#517](https://github.com/mudler/vllm.cpp/issues/517), parent spec
[nemotron-h-model.md](nemotron-h-model.md)) — this spec owns what that spec's
§6e carried forward as item 3 and explicitly put out of its own scope.
**Depends on an in-flight fix:** [#775](https://github.com/mudler/vllm.cpp/issues/775)
is already repaired by **open PR
[#868](https://github.com/mudler/vllm.cpp/pull/868)** — see §7 R2. A1 consumes it,
it does not redo it.
**Base:** `origin/main` @ `2e9d95e74c7aee133e21771182a6a587fe74c67b`.
**Pinned oracle:** `${VLLM_SOURCE}` = `/home/mudler/_git/vllm` @
`5559679229bc961848b121ccdeaa8fa5d79bec98` (vLLM 0.26.0.dev0), verified at HEAD
while writing this spec, per [upstream-sync.md](../upstream-sync.md).
**Weight loader:** landed at `bc570da0d`.
**Lifecycle at this commit:** the row stays `INVENTORIED`. A spec commit changes
no lifecycle state and therefore owes no `STATUS`/`BENCHMARKS`/`NOW` write; §8
records what the implementing changes owe.

**No product code is written by this spec.** It is committed before
implementation, per AGENTS.md §"Spec before code".

---

## 0. The established fact this spec starts from

An independent investigation proved, on `main` with the weight loader merged, on
the real 21 GiB checkpoint on GB10:

```
vllm-cli: model load failed (status 2): vllm_engine_load: vt: runner: Qwen3.5 MambaSpec
  shapes disagree with model config at src/vllm/v1/worker/gpu/runner.cpp:525
```

The weights load (17.7 GiB RSS) and then **engine construction** refuses.

`GPUModelRunner::initialize_kv_cache` (`src/vllm/v1/worker/gpu/runner.cpp:439`,
body to `:785`) has two halves with opposite polarity.

**The attention half is spec-driven, and its comment says why.**
`runner.cpp:539-560` records that *"upstream never reconstructs the KV shape from
the HF config: every layer's cache bytes come from its SPEC"*, and names the
defect it was written to remove — *"we previously hardcoded `num_blocks * 2 *
block * Hkv * Dh` with Hkv/Dh read from `config_`"*. The implementation at
`:566-607` takes `block_size`, `num_kv_heads`, `head_size`, `dtype` and
`page_size_bytes()` off the `AttentionSpec`.

**The recurrent half is config-driven and uses the spec only as an assertion
oracle.** `runner.cpp:494-507` rebuilds the geometry from HF-config fields
NemotronH does not ship:

```cpp
const int64_t Hk = config_.linear_num_key_heads;        // :498  -> 0
const int64_t Kw = config_.linear_conv_kernel_dim;      // :502  -> 0
const int64_t conv_dim = 2 * key_dim + value_dim;       // :505  -> 0
const int64_t conv_state_len = ... (Kw - 1) ...;        // :507  -> -1
```

and then, at `:523-527`, refuses when the model's own published spec disagrees
with that reconstruction:

```cpp
VT_CHECK(mamba_spec->shapes[0] == expected_conv_shape &&
             mamba_spec->shapes[1] == expected_ssm_shape,
         "runner: Qwen3.5 MambaSpec shapes disagree with model config");
```

`MakeNemotronHKVCache` (`src/vllm/model_executor/models/nemotron_h_registry.cpp:155-217`)
correctly published `{6144, 3}` and `{64, 64, 128}` at `:204-215`.

Note what the surrounding lines already get right: `:528-529` takes the state
**dtypes** off the spec, and the GDN allocation at `:663-679` sizes buffers with
**shapes from the config and dtypes from the spec**. So the file is already
half-converted; A1 finishes a conversion the attention half completed.

**The comment at `:494-497` claims the polarity the code does not have.** It
states that *"as in upstream, MambaSpec is the source of truth for the recurrent
tensors' order, shapes, dtypes, and page bytes"*. That is the intended design and
it is what §2 confirms upstream does. The code validates against config-derived
numbers instead. A1 makes the comment true.

### 0.1 Per-layer membership is a Qwen3.5 string

`runner.cpp:660-662`:

```cpp
const bool is_gdn =
    has_mamba_group && !config_.layer_types.empty() &&
    config_.layer_types[static_cast<size_t>(l)] == "linear_attention";
```

NemotronH's `layer_types` is empty, so all **52** layers are treated as full
attention: zero recurrent buffers, and ~8.7x the attention pages actually needed
(52 against the 6 real GQA layers at indices {5, 12, 19, 26, 33, 42}).

`KVCacheGroupSpec::layer_names` (`include/vllm/v1/kv_cache_interface.h:343`) is
already populated by `MakeNemotronHKVCache` with the real per-layer names
(`nemotron_h_registry.cpp:178`, `:206`). **`initialize_kv_cache` never reads
it** — the runner keys purely on `kind()` and layer index. Group membership is
the durable signal; a config spelling is not.

There are now **three** independent per-layer-hybrid signals in this tree that
must agree and are never cross-checked: `config_.layer_types[l]` in the runner,
`layer.is_linear_attention` — a *weights* flag — in the Qwen3.5 forward
(`src/vllm/model_executor/models/qwen3_5.cpp:7053-7065`), and
`KVCacheGroupSpec::layer_names` in the spec. A1 does not unify all three, but it
moves the runner onto the one the model publishes, which is the one that cannot
be absent.

### 0.2 Why Kimi-Linear works and NemotronH does not — the anti-pattern to refuse

Only three architectures produce a `MambaSpec` at all: Qwen3.5/3.6
(`qwen3_5_common.cpp:82`), Kimi-Linear (`kimi_linear_registry.cpp:153`) and
NemotronH (`nemotron_h_registry.cpp:210`). Qwen3.5 passes the `:525` check
because the check *is* Qwen3.5's geometry. Kimi-Linear passes only because
`src/vllm/transformers_utils/hf_config.cpp:484-528` **synthesizes** the
`linear_*` fields and fabricates a `layer_types` vector of `"linear_attention"` /
`"full_attention"` strings out of its nested `linear_attn_config`. Its comment at
`:484-488` states the purpose outright: so that *"the runner's MambaSpec
consistency check (`runner.cpp` `expected_conv_shape`/`expected_ssm_shape`) and
its per-layer linear-attention/full-attention allocation loop see the same
geometry `MakeKimiLinearKVCache` declares"*.

There is no equivalent synthesis for NemotronH, and **this spec forbids adding
one.** Its config ships `layers_block_type`, `mamba_num_heads`, `mamba_head_dim`,
`ssm_state_size` and `conv_kernel`, and no `layer_types` and no `linear_*`. A
third synthesis branch would work, and it is how this defect reaches model number
four. The model already published the truth; the runner should read it.

### ★ THE SAFETY CONSTRAINT — it shapes every scope decision below

Neutering **only** the `VT_CHECK` at `runner.cpp:525` makes `vllm_engine_load`
**succeed** and reach the forward. `ForwardNemotronHForCausalLM`
(`nemotron_h_registry.cpp:120-133`) consumes exactly **three** of
`ModelForwardInput`'s eighteen fields — `token_ids`, `logits_indices`, `queue`. It
ignores `positions`, `attn_meta`, `gdn_meta`, `attn_kv`, `gdn_state`,
`gdn_state_slots`, `num_reqs`, `pure_decode`, `num_speculative_tokens`,
`gather_logits` and the rest (struct at
`include/vllm/model_executor/models/model_registry.h:229-285`). A server past
that check would decode step 2 onward with **fresh recurrent state and no KV** —
plausible tokens, wrong tokens, no refusal.

**Fixing the allocation without the forward is strictly MORE DANGEROUS than
today's refusal.** The refusal is currently the only thing making the gap
visible.

This is a **hard gate on this spec, not advice**:

> **G-SAFE.** No change may land that removes or weakens the `runner.cpp:525`
> refusal unless the same commit either (a) lands the device/paged forward that
> consumes `attn_kv` / `gdn_state` / `gdn_meta` / `gdn_state_slots` / `num_reqs`,
> or (b) installs a by-name refusal in `ForwardNemotronHForCausalLM` for any
> step where `attn_kv` or `gdn_state` is non-empty, or `num_reqs > 1`. The
> refusal names the architecture and the missing piece, per AGENTS.md.
>
> A reviewer who cannot point at the interlock in the same diff returns FAIL.
> The interlock is **narrowed, never deleted**, until every clause it guards is
> implemented and gated.

**The interlock is not novel — mirror the in-tree idiom.**
`ForwardKimiLinearForCausalLM` (`kimi_linear_registry.cpp:88-113`) already gates
its paged fold on `!input.attn_kv.empty() && !input.gdn_state.empty()` at
`:99-102`, keeping the non-paged seam alive below it. That is structurally the
same predicate, inverted. Copy its shape rather than inventing one.

---

## 1. Scope decision — three work items, two landable units

**Decision: SEVERAL, not one.** The four candidate slices (S1 allocation, S2
per-layer routing, S3 the device/paged forward, S5 the ABI token gate) do not
split four ways, and they do not collapse into one either.

### A1 — the shared-runner refactor plus the safety interlock

**S1 and S2 are ONE unit, not two.** They are the same refactor of the same
function, and neither is separately testable on the driver model:

- S1 without S2 computes correct conv/SSM shapes for **zero** layers — with
  `layer_types` empty, `:660-662` classifies every layer as full attention, so
  nothing consumes the corrected shapes. The change would be green and inert.
- S2 without S1 still trips the `VT_CHECK` at `:525` before the routing loop at
  `:629-722` is reached. The change would be unreachable.

They fail together, they are proven by the same test pair, and splitting them
manufactures a commit that cannot be gated. **A1 = S1 + S2 + the G-SAFE
interlock, in one commit, on top of #868.**

| In A1 | Out of A1 |
|---|---|
| conv/SSM shapes, dtypes and per-slot bytes read from `mamba_spec->shapes` / `->dtypes`, mirroring `runner.cpp:539-607` | any change to `dense_attn`, the block table, the scheduler, or the KV manager |
| the config cross-check at `:523-527` **deleted**, not widened; the three `Qwen3.5`-named messages at `:516`, `:525`, `:535` renamed to name the architecture | any new HF-config field, and specifically **no** NemotronH branch in `hf_config.cpp` (§0.2) |
| per-layer recurrent membership derived from `KVCacheGroupSpec::layer_names` | the device forward (A2) |
| the G-SAFE by-name refusal in `ForwardNemotronHForCausalLM` | the MTP head (parent W5) and GGUF (parent W7) |
| the interlock opened through `vllm::ModelAs<>` from PR #868 (§7 R2) | re-fixing #775, and the tree-wide sweep of the cast class (#847) |
| a by-name refusal in `MakeKVCacheMaybeSpec` for a speculative non-Qwen3.5 config (§1.2) | making speculative NemotronH actually work |

**What A1 unblocks:** every hybrid whose config does not speak Qwen3.5's
`linear_*` / `layer_types` dialect can allocate, without a per-model synthesis
branch. It is a precondition for A2 and for any future Mamba2 architecture.

### A2 + A3 — the device/paged forward and its e2e proof, one PR

**S3 and S5 land together.** A device forward with no e2e gate is unfalsifiable,
and an e2e gate with no device forward has nothing to run. `vllm_complete_tokens`
already exists at `VLLM_ABI_VERSION 20` (`include/vllm.h`, added at v13), so S5
needs **no ABI growth** — there is no separable "grow the surface" task to carve
out. Keeping them in one PR also means the commit that deletes the G-SAFE
interlock is the commit that proves it is no longer needed.

**S3 closes four gaps at once**, which is why it is one item and not four:

1. **No paged attention.** `NemotronHAttentionMixer`
   (`src/vllm/model_executor/models/nemotron_h.cpp:585-630`) recomputes Q, K and
   V from scratch on every call — `:611-613` — then runs one dense causal
   `vt::Attention` over the whole `[T, ·]`. No `PagedKvCache`, no slot mapping,
   no `ReshapeAndCache`. `NemotronHGreedyDecode` (`:970-992`) therefore runs a
   full forward over the growing sequence per generated token. The fp8 KV scales
   the loader materialized are documented as unused for exactly this reason
   (`nemotron_h_forward.h:207-216`).
2. **No batching.** `token_ids` is the step's concatenated scheduled tokens
   **across requests**, and the forward treats it as ONE causal sequence. Two
   concurrent requests silently cross-contaminate — a correctness defect that
   produces fluent output.
3. **The CPU-queue-only `VT_CHECK`** at `nemotron_h.cpp:822-825`, inside
   `NemotronHForward`, which names this spec's predecessor as its owner.
4. **`PrepareNemotronHForCausalLM` is a pure no-op**
   (`nemotron_h_registry.cpp:113-118` — three `(void)` casts), so nothing warms,
   sizes or validates per-step state.

Each is a symptom of the same thing — the forward is a host reference that owns
its own K/V and its own state — so repairing one without the others produces a
half-paged forward harder to reason about than either end state.

**Permitted split, if a reviewer would be better served by parts** (AGENTS.md:
size is a review judgement, not a counter): A2 may split into **A2a**
(single-request paged decode; `num_reqs > 1` still refused) and **A2b**
(batching). The rule that makes this safe is the one already stated: at A2a the
interlock is **narrowed** to the `num_reqs > 1` clause, not removed. A2a with
the interlock deleted is not an acceptable split.

**WHAT A2 ACTUALLY SPLIT INTO — read this before using the `A2a`/`A2b` names
above.** A2 was scoped as one unit and came back as a campaign. The A2-R landing
commit (`598226e96`) renamed the units, because `A2a`/`A2b` collide with both the
paragraph above and Laguna's unrelated "Brick A2b": **A2-R** (dense weights on
the residency seam, landed), **A2-Q** (the quantized arms), **A2-P** (this
paragraph's `A2a`, single-request paged decode, and the unit that NARROWS the
interlock), **A2-B** (this paragraph's `A2b`, batching). A2-Q was then split
again on 2026-08-16, for a reason that is not size: its two halves have
DIFFERENT blockers, so bundling them makes the NVFP4 half wait on a build-file
fix it does not need.

| Unit | Spec | Blocker |
|---|---|---|
| A2-Q1 — FP8 W8A8 mamba | [`nemotron-h-a2q1-fp8-mamba.md`](nemotron-h-a2q1-fp8-mamba.md) | [#960](https://github.com/mudler/vllm.cpp/issues/960), declared as a base |
| A2-Q2 — NVFP4 MoE + `lm_head` | [`nemotron-h-a2q2-nvfp4-moe-lmhead.md`](nemotron-h-a2q2-nvfp4-moe-lmhead.md) | none for the build; the Thor leg is PENDING [#962](https://github.com/mudler/vllm.cpp/issues/962) |
| A2-P — paged KV, carried recurrent state, device logits | [`nemotron-h-a2p-paged-forward.md`](nemotron-h-a2p-paged-forward.md) | none; #496 W2 landed at `43a6c5518`, so R3 below is CLEARED |

Both A2-Q halves leave G-SAFE fully intact, and both inherit A2-R's unreached
posture — the device forward still has no production caller until A2-P.

**A2-P is the unit that narrows the interlock**, that removes or narrows
`scripts/runner-routing-allowlist.txt:26`, and that gives every device arm its
first production caller. It carries two corrections to this file, measured at
`10002648199cfbbaf1e423f7c80cacb2f4b56366` and argued in
[`nemotron-h-a2p-paged-forward.md`](nemotron-h-a2p-paged-forward.md):

- **§2's `dense_attn::AttnBlock` seam claim is WRONG for this architecture** and
  is superseded by A2-P §2.3. That block applies `vt::RopeNeox` unconditionally
  (`dense_attn_block.h:497` — the `:496` cited in-tree is a comment line) and
  reads a `cfg.rms_norm_eps` that `hf_config.cpp:551` defaults to `0.0` for a
  checkpoint shipping `layer_norm_epsilon` instead. NemotronH has no RoPE at all.
  Filed as [#941](https://github.com/mudler/vllm.cpp/issues/941); A2-R already
  took the model-local road, and A2-P extends that block rather than migrating.
- **§2.7's R4 named an unplanned kernel arm that already exists.**
  `vt::CausalConv1dFwd` admits a bf16 conv state wherever the backend answers
  `SupportsCompressedConvState()` (`src/vt/ops.cpp:1644-1650`), which CUDA
  (`cuda_backend.cu:117`), Vulkan (`vulkan_backend.cpp:142`) and ROCm
  (`rocm_backend.hip:333`) all do. It landed at `908bad0ac` on 2026-08-09, six
  days before this spec was written. The decision — a bf16 persistent conv page,
  never widened to f32 — is unchanged; only its cost was mis-stated.

### 1.1 Dependency order

```
A1 (runner refactor + interlock)  ──►  A2 (device/paged forward)  ──►  A3 (ABI e2e token gate)
        │                                        │                          └── same PR as A2
        │                                        ├── requires KERNEL-SSM-MAMBA (#496) W2, the CUDA arm
        │                                        └── requires the conv-state dtype mirror (§2.7)
        └── requires main + bc570da0d, and PR #868 for `vllm::ModelAs<>` (§7 R2)
```

A1 is independent of #496. Its only dependency is PR #868 (§7 R2), which is a
small in-flight fix rather than a campaign, so A1 is claimable now. A2 cannot: the parent spec's
§0 hard blocker holds — the CUDA Mamba2 SSD arm is W2 of
[#496](https://github.com/mudler/vllm.cpp/issues/496)
([spec](mamba2-ssd.md)). **Re-verify #496 W2's state against the current head and
the current issue before claiming A2; do not implement against a described
dependency** ([`porting.md`](../porting.md)).

### 1.2 `MakeKVCacheMaybeSpec` — decided: refusal in A1, repair owed to parent W5

`src/vllm/entrypoints/model_loader.cpp:835-846` hard-routes **any** speculative
config to `MakeQwen3_5KVCacheSpec` at `:842-843`, unconditionally on
architecture. All four speculative methods reach it (`mtp`, `dflash`, `ngram`,
`dspark`). Same defect class as #810 — a Qwen3.5-shaped assumption in shared
code — but not on A1's critical path, because no speculative NemotronH config
can exist until the parent spec's W5 (the MTP head) lands, and
`MakeNemotronHKVCache` deliberately passes no `num_speculative_blocks`
(`nemotron_h_registry.cpp:202-203`).

**Decision: in scope for A1 as a refusal only.** A1 refuses a speculative config
for an architecture that is not Qwen3.5, naming the architecture and the missing
piece, rather than silently handing it Qwen3.5 KV specs. That is cheap, it is
the AGENTS.md rule for an unimplemented arm, and it converts a silent-wrong-answer
path into visible debt. **The real repair — routing a speculative config through
the model's own KV-cache factory — is owed to parent W5** and is out of scope
here. Record it in the parent spec's W5 section when A1 lands.

### 1.3 Adjacent debt found while scoping — recorded, not fixed

`runner.cpp:483-491` recognizes only `kFullAttention`, `kMlaAttention` and
`kMamba`. A `SlidingWindowSpec` or `ChunkedLocalAttentionSpec` group would match
nothing and leave `full_attn_group_id_ == -1`. No registry KV builder produces
either today (`ChunkedLocalAttentionSpec` is constructed only at
`src/vllm/model_executor/layers/attention/chunked_local_attention.cpp:143`), so
this is latent, not live. **Out of scope. File an issue when A1 is claimed** —
AGENTS.md requires an issue for a bug found in flow, and this one is not
fixable in flow because no model exercises it.

### 1.4 Spec and implementation are SEPARATE pull requests (decided, do not re-litigate)

`porting-a-model.md` §0 says to ask the developer at row claim whether the spec
and the implementation share one pull request, and to record the answer.
**Asked and answered on 2026-08-15: separate. Spec first, implementation after.**

The reason is recorded here so the next row claim does not re-open it. It is not
a preference — it falls out of two AGENTS.md rules that compose only one way:
*"the spec is committed **before** implementation, never written up afterwards"*,
and *"a **fresh implementer** works from the committed spec"*. A fresh implementer
is not fresh if it also authored the spec, and a spec shipping in the same pull
request as its implementation is being reviewed alongside the code it was
supposed to constrain — which is the review this protocol exists to avoid. The
general recommendation in `porting-a-model.md` §0 covers the ordinary case where
a spec is a paragraph of scope; this row's spec is the artifact a different agent
works from.

So: **this PR carries the spec and no product code. A1 is a separate PR by a
different agent**, and its reviewer is a third.

### 1.5 Explicitly out of scope

Speed (no ratio is recorded by this row); the MTP head; the GGUF k-quant arm
(parent §5b/W7, still refused by name); `NemotronHPuzzleForCausalLM`; TP
sharding of `n_groups`; any change to the Qwen3.5 or Kimi-Linear model files.

---

## 2. Upstream anchors — `file:line` on both sides, at `555967922`

Verified against the checkout at the pin (`git rev-parse HEAD` →
`5559679229bc961848b121ccdeaa8fa5d79bec98`). **Nemotron-H is still in the old
layout at this pin**: `vllm/model_executor/models/nemotron_h.py`, registered at
`vllm/model_executor/models/registry.py:179`. The newer `vllm/models/<name>/`
package holds only `deepseek_v32`, `deepseek_v4`, `inkling` and `minimax_m3` — so
an `ls` of either directory alone would have been misleading, and both were
searched.

### 2.1 The polarity we have inverted — upstream derives the spec from the MODULES

This is the load-bearing anchor of the whole spec.

| Question | Upstream | Ours |
|---|---|---|
| Where does a hybrid's KV spec come from? | `gpu_model_runner.py:7774` `get_kv_cache_spec()` — docstring: *"parsing the kv cache format from each Attention module in the static forward context"* | `ModelRegistry::MakeKVCache` (`model_registry.cpp:329-334`), per-model factory |
| How are the layers enumerated? | `gpu_model_runner.py:7785-7787`: `get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase)`, then `for layer_name, attn_module in attn_layers.items()` | `config_.layer_types[l]` string compare (`runner.cpp:660-662`) |
| Who produces the shape? | the **layer instance**: `gpu_model_runner.py:7801` `attn_module.get_kv_cache_spec(...)`; Mamba's is `mamba/abstract.py:63-79`, built from `self.get_state_shape()` / `self.get_state_dtype()` | the runner, from `config_` (`runner.cpp:494-507`) |
| Is the HF config ever re-read for shapes? | **only** for page-size padding, on the platform, not in the runner: `vllm/platforms/interface.py:859-860` → `cache_config.mamba_page_size_padded`, consumed back at `mamba/abstract.py:66` | yes, and it is the refusal |

The layer registers *itself* by prefix at
`vllm/model_executor/layers/mamba/mamba_mixer2.py:496-499`
(`compilation_config.static_forward_context[prefix] = self`), and the registry
is filtered by `isinstance` at `vllm/config/vllm.py:2442-2449`. `MambaBase` is an
`AttentionLayerBase` (`mamba/abstract.py:18`) with abstract `get_state_shape`
(`:46`), `get_state_dtype` (`:60`) and `mamba_type` (`:56`).

**No `layer_types` or `hybrid_override_pattern` string is parsed anywhere in
`gpu_model_runner.py`.** The only reads of `hybrid_override_pattern` are inside
`nemotron_h.py` at `:275`, `:572`, `:578`, `:595` — module-construction time
only, to choose a layer class from `ALL_DECODER_LAYER_TYPES` (`:531-536`,
`M`/`-`/`*`/`E`). By the time the runner asks, the pattern has already become
instantiated objects.

### 2.2 The runner allocates BYTES; the LAYER creates typed views

`vllm/v1/worker/gpu_model_runner.py:7311-7313`:

```python
tensor = torch.zeros(kv_cache_tensor.size, dtype=torch.int8, device=self.device)
```

and the Mamba branch of the reshape, `:7429-7441`, hands the layer an untyped
page view:

```python
elif isinstance(kv_cache_spec, MambaSpec):
    page_size_bytes = kv_cache_spec.page_size_bytes
    kv_caches[layer_name] = raw_tensor[: num_blocks * page_size_bytes].view(
        num_blocks, 1, 1, page_size_bytes)
```

The unpack into typed conv/SSM views happens in the layer,
`vllm/model_executor/layers/mamba/abstract.py:38-43` (method from `:29`):

```python
for shape, dtype in zip(self.get_state_shape(), self.get_state_dtype()):
    nbytes = prod(shape) * get_dtype_size(dtype)
    state = pages[:, offset : offset + nbytes].view(dtype)
    states.append(state.view(-1, *shape))
```

**The upstream runner literally cannot perform our cross-check** — it never
holds a conv shape. **That is the argument for DELETING the check, not for
relocating it.** A cross-check that upstream is structurally incapable of
expressing is not a safety net we happen to have and they happen to lack; it is
a second, independent derivation of a number the model already published, and a
second derivation is exactly the thing that can disagree. Widening it to accept
NemotronH's geometry, or moving it behind a per-architecture switch, would keep
the defect and add a maintenance surface. It goes.

That is also why `MambaSpec` carries `shapes` and `dtypes`
(`vllm/v1/kv_cache_interface.py:690-696`) and `page_size_bytes` is their sum
(`:699-703`), and why our `MambaSpec`
(`include/vllm/v1/kv_cache_interface.h:302-323`) mirrors those fields and
deliberately carries **no** `num_heads` / `head_dim` / `conv_dim` — the geometry
is implicit in `shapes`, exactly as upstream. A1 stops re-deriving what the
mirror deliberately did not store.

Hybrid layout reconciliation (attention blocks-first vs K/V-first when a mamba
group coexists) is `gpu_model_runner.py:7443-7449` and
`_update_hybrid_attention_mamba_layout` at `:7477-7509` — read before changing
our tensor views.

### 2.3 Group membership and per-layer routing

- Grouping key is the **spec object**, `vllm/v1/core/kv_cache_utils.py:1209-1211`:
  `same_type_layers[layer_spec].append(layer_name)`; entry point
  `get_kv_cache_groups` at `:1760`, hybrid path
  `_get_kv_cache_groups_uniform_page_size` at `:1140`.
- `KVCacheGroupSpec` is `layer_names: list[str]` + `kv_cache_spec`,
  `vllm/v1/kv_cache_interface.py:938-947` — the exact structure our `:343`
  mirrors and does not read.
- Backend groups are built from the modules **again**, not the config:
  `gpu_model_runner.py:7024-7027`; `AttentionGroup` carries
  `layer_names` + `kv_cache_group_id` (`vllm/v1/worker/utils.py:211-215`).
- Per-group metadata is built once and **fanned out by layer name**,
  `gpu_model_runner.py:2548-2549`:
  ```python
  for layer_name in attn_group.layer_names:
      attn_metadata_dict[layer_name] = attn_metadata_i
  ```
  and each layer looks itself up by its own prefix,
  `mamba_mixer2.py:712-713`: `attn_metadata = attn_metadata_raw[self.prefix]`.
- Mamba groups get **no token→slot mapping**, `gpu_model_runner.py:7238-7240`:
  `if kv_cache_spec_kind == KVCacheSpecKind.MAMBA:
  slot_mapping_modes.append(SlotMappingMode.NONE)`.

**Mirror this:** per-layer membership comes from `layer_names`, which is what A1
switches to.

### 2.4 State across decode steps, and across requests — what A2 must mirror

- **Batch reorder.** Decodes first: `gpu_model_runner.py:1126-1130`
  `reorder_batch_to_split_decodes_and_prefills(...)`; the function is
  `vllm/v1/attention/backends/utils.py:665`, order documented at `:574-575`.
  Mamba builders declare `reorder_batch_threshold: int = 1`
  (`vllm/v1/attention/backends/mamba_attn.py:87`).
- **Counts.** `mamba_attn.py:463-469` `split_decodes_and_prefills(...)`
  (`utils.py:564`).
- **Per-request state index comes from the BLOCK TABLE**, not a slot map:
  `mamba_attn.py:513-518` `mamba_get_block_table_tensor(...)`
  (`utils.py:927-965`), then split by request class at `:523-532`:
  ```python
  state_indices_tensor_d, state_indices_tensor_p = torch.split(
      state_indices_tensor, [num_decodes, num_prefills], dim=0)
  ```
- **Continuation detection.** `mamba_attn.py:554-556`
  `has_initial_states_p = (num_computed_tokens[num_reqs - num_prefills : num_reqs] > 0)`.
- **Metadata fields** on the base: `mamba_attn.py:29-81` —
  `has_initial_states_p:39`, `state_indices_tensor_p:42`,
  `state_indices_tensor_d:47`, `cu_chunk_seqlen_p:66`, `last_chunk_indices_p:69`.
  Mamba2 adds `prep_initial_states`, `chunk_size`, `seq_idx_p`
  (`mamba2_attn.py:105-111`); the varlen chunk metadata builder is
  `mamba2_attn.py:31-88`.
- **The mixer's forward**, `mamba_mixer2.py:151` / body from `:695`:
  token split `:757-760`, output split `:808-812`; prefill conv `:832-847`
  (`causal_conv1d_fn(..., has_initial_state=has_initial_states_p,
  cache_indices=state_indices_tensor_p)`); per-request initial SSM states
  gathered `:855-866`; varlen scan `:870-892`
  (`mamba_chunk_scan_combined_varlen(..., seq_idx=seq_idx_p,
  cu_chunk_seqlens=cu_chunk_seqlen_p, state_dtype=ssm_state.dtype)`); state
  write-back `:977-978` `ssm_state[state_indices_tensor_p] = varlen_states`;
  decode in-place slots `:1006-1010`; decode conv `:1013-1025`; decode SSM
  `:737-742` with `state_batch_indices` / `dst_state_batch_indices`.

**This is what "batched recurrent state" means, and none of it exists locally
yet.** A2 mirrors it; A2 does not invent it.

### 2.5 Nemotron-H's own shapes and dtypes at the pin

- `nemotron_h.py:761-798` `get_mamba_state_shape_from_config`:
  `intermediate_size = mamba_num_heads * mamba_head_dim` (`:779`), then
  `MambaStateShapeCalculator.mamba2_state_shape(...)` (`:781-790`).
- `nemotron_h.py:744-758` `get_mamba_state_dtype_from_config` →
  `MambaStateDtypeCalculator.mamba2_state_dtype(model_dtype, mamba_cache_dtype,
  mamba_ssm_cache_dtype)`.
- The shape expressions, `vllm/model_executor/layers/mamba/mamba_utils.py:174-199`:
  ```python
  conv_dim = intermediate_size + 2 * n_groups * state_size
  conv_state_shape = cls._orient_conv_shape(divide(conv_dim, tp), conv_kernel - 1 + num_spec)
  temporal_state_shape = (divide(num_heads, tp), head_dim, state_size)
  ```
- **Both classmethods are consumed only for page padding** (`interface.py:859-860`)
  and by `nano_nemotron_vl.py`. The runtime shapes come from the layer instance:
  `mamba_mixer2.py:1119-1139` `get_state_shape()`, `:1105-1117`
  `get_state_dtype()`.

### 2.6 The conv-state layout question — ANSWERED, and our orientation is a supported upstream mode

`mamba_utils.py:23` defines `ConvStateLayoutType = Literal["SD", "DS"]`;
`:27-43` `get_conv_state_layout()` returns **`"SD"` by default** (the `return "SD"`
is `:43`; `is_conv_state_dim_first()` is `:46`), overridable by
`VLLM_SSM_CONV_STATE_LAYOUT` (`vllm/envs.py:227`, parsed `:1698-1700`). The
decisive expression is `:152-157`:

```python
def _orient_conv_shape(dim, state_len):
    """Return (dim, state_len) for DS layout, (state_len, dim) for SD."""
    if is_conv_state_dim_first():
        return (dim, state_len)
    return (state_len, dim)
```

and the kernels want dim-major regardless, so SD is transposed on the way in —
`mamba_mixer2.py:714-721`:

```python
conv_state = (self.kv_cache[0] if is_conv_state_dim_first()
              else self.kv_cache[0].transpose(-1, -2))
```

**So our local `(dim, state_len)` is upstream's `DS`** — a first-class,
upstream-supported layout, not a divergence. The note at
`nemotron_h_registry.cpp:194-200` is correct that the bytes are the same product,
and it can now be sharpened from "our local convention" to "upstream's `DS`
layout, `VLLM_SSM_CONV_STATE_LAYOUT=DS`, `mamba_utils.py:152-157`". Upstream even
has the byte-equality test to port: **`tests/v1/worker/test_mamba_utils.py:2136`
`test_ds_conv_layout_bias_gt_0_byte_equal_to_sd`**.

**Owed by A2:** port that test, preserving its parameters and its revision
anchor, per AGENTS.md §"Port its tests in the same change".

### 2.7 The conv-state DTYPE question — ANSWERED, and it exposes a live local conflict

`mamba_utils.py:96-107` `_mamba_state_dtype`, reached from `mamba2_state_dtype`
at `:73-81`:

```python
conv_state_dtype = get_kv_cache_torch_dtype(mamba_cache_dtype, model_dtype)
if mamba_ssm_cache_dtype == "auto":
    temporal_state_dtype = conv_state_dtype
else:
    temporal_state_dtype = STR_DTYPE_TO_TORCH_DTYPE[mamba_ssm_cache_dtype]
```

So **conv state is NOT f32 upstream**: it carries `--mamba-cache-dtype`
(default `auto` → the model dtype, bf16 here). Only the SSM/temporal state takes
`mamba_ssm_cache_dtype`, which this checkpoint sets to `float32`. The only
hard-coded `torch.float32` states at the pin are the ReplaySSM `dt_cache` ring
(`mamba_utils.py:93`) and KDA (`:137`) — neither applies.

That makes `MakeNemotronHKVCache`'s `conv_dtype = kBF16`
(`nemotron_h_registry.cpp:167`) **a correct mirror**, and it makes the local
conflict explicit rather than latent:

> `nemotron_h_forward.h:284-296` records that the host reference's conv state is
> **f32 "because `vt::CausalConv1dFwd` validates the conv state as f32"**, while
> the persistent page the registry declares is bf16 — and it says outright that
> reconciling the two belongs to the paged-decode work, i.e. **A2**.

**Decision (developer-directed 2026-08-15): mirror upstream. A2 gives
`vt::CausalConv1dFwd` a bf16 conv-state arm and stores the conv state at the
cache dtype.** This is a correctness obligation under AGENTS.md, **not a
scheduling call**, and it is not to be re-litigated as one when the kernel work
turns out to be real.

The reasoning follows from the rules rather than from taste:

- AGENTS.md: *"vLLM resolves ONE model dtype and every layer inherits it"*, and
  `f32` is a rare, annotated escape. The upstream answer here is measured, not
  inferred — `mamba_utils.py:96-107`, conv state carries `mamba_cache_dtype`,
  default `auto` → the model dtype → bf16 for this checkpoint.
- AGENTS.md again, and this is the whole point: *"a token gate cannot catch a
  dtype that is too WIDE — it is still numerically correct, so tokens match,
  goldens pass, and we move twice the bytes."* An f32 conv state where upstream
  uses bf16 is **precisely** that defect. Every gate this row owns is blind to
  it, which is why it is settled in the spec rather than during implementation.
- **The registry is right and the precondition is the bug.**
  `nemotron_h_registry.cpp:167` declares `kBF16` and mirrors upstream correctly.
  What forces f32 is a *local* precondition — `vt::CausalConv1dFwd` validating
  its conv state as f32 — and a local precondition contradicting a correct
  mirror is a defect in the precondition, not a reason to widen the page to
  satisfy it.

**The f32 fallback survives only as an annotated escape**, never as the default:
if it is taken it owes a one-line reason at the buffer and its cost **stated in
bytes per token** in §10. A silent widening is not one of the options.

---

## 3. RED-first, and why the existing gate cannot be trusted

`runner.cpp` serves **every** model — every architecture served by `LoadedEngine`
flows through this one function (`model_loader.cpp:1007-1023` constructs the
single `GPUModelRunner`). A refactor of it owes a proof that it changes nothing
for the models that already work, and that proof must be armed.

### 3.1 The existing gate asserts this row's claim and its shape half is INERT

`tests/vllm/v1/worker/test_runner.cpp:505` is titled **"runner: MambaSpec is the
allocation source of truth"**, and it passes today — on a tree where the
MambaSpec is demonstrably *not* the source of truth for shapes. It is not lying;
its shape half is a **self-consistency gate**
([[gate-comparing-shared-helper-proves-consistency-not-correctness]]):

- `MakeConfig()` (`test_runner.cpp:98-120`) populates every `linear_*` field and
  a full `layer_types` vector;
- `MakeKvConfig(c)` derives the `MambaSpec` shapes from **that same `c`**;
- so `shapes == expected_*_shape` is two derivations of one config agreeing with
  each other. It cannot fail, and the `VT_CHECK` at `:525` passes straight
  through it.

The **dtype** half of that same case *is* genuinely armed — `MakeKvConfig(c,
kBF16, kF16)` passes dtypes the config cannot produce, and `:528-529` honours
them.

**Its dtype half being armed while its shape half is inert is the cheapest
RED-first anchor in this row**, and it is worth saying in exactly those terms.
The two halves sit in one `TEST_CASE`, five lines apart, testing one function.
One of them already does the right thing. So the repair needs no new harness, no
new fixture and no new file — it needs the shape half to be given inputs the
config cannot produce, which is what the dtype half is already doing next to it.

**This is the RED-first anchor and it is cheap.** Add a subcase whose `MambaSpec`
shapes are **not derivable from the config**: `linear_*` zeroed, `layer_types`
empty, spec shapes `{6144, 3}` and `{64, 64, 128}` — the real NemotronH values.
On `main` it fails at `runner.cpp:525` with `Qwen3.5 MambaSpec shapes disagree
with model config`. **Capture that red and put the transcript in the PR body.**
After A1 it is green. A test never seen failing has proven nothing
([`verification.md`](../verification.md)).

Note what is *already* gated and needs no new coverage: the NemotronH **spec
side** is fully asserted at `tests/vllm/models/test_nemotron_h_scaffold.cpp:620-665`
— two groups, `{{6144,3},{64,64,128}}`, `{kBF16, kF32}`, all 6 + 23 real layer
names, `page_size_bytes()`, and `KVBytesPerBlock == attn page * 6`. **The gap is
that no test anywhere constructs a `GPUModelRunner` from a NemotronH spec.**
A1 closes exactly that gap and nothing else.

### 3.2 The Qwen3.5 byte-identity arm — required, not optional

A1 must carry a **byte-identity** arm proving the refactor changes nothing for
the existing hybrid. The precedent and the wording already exist in this tree:
`include/vllm/v1/kv_cache_interface.h:354-374` states a **BYTE-NEUTRALITY
CONTRACT** for the `per_layer_attn_specs` seam — "byte-identical allocation,
view, indexing and kernel dispatch to before this field existed". Mirror that
contract and that phrasing; a reviewer already knows how to read it.

The arm asserts an **explicit table**, not spot checks, and it must be able to
say **how many** things it examined ([[the-state-was-not-the-one-you-believed]]):

| Asserted, per Qwen3.5 arm | Why |
|---|---|
| `attn_kv().size()`, `gdn_state().size()`, `num_hidden_layers` | a changed count is the signal; a suite that cannot report its own N has not reported |
| the layer→group assignment for **every** layer index, as a literal vector | spot-checking layer 0 cannot see a routing inversion |
| every `conv_state` / `ssm_state` shape and dtype, per layer | that is the defect class |
| `full_attn_group_id()`, `gdn_group_id()` | group selection is what A1 rewires; the `fa_draft` case at `test_runner.cpp:468` must stay green |
| `fa_page_size_bytes()` and total allocated bytes | the byte-neutrality claim itself |

Expected literals are derived at the **pre-refactor base SHA** and committed as
literals. Deriving them from the config inside the test reproduces exactly the
defect §3.1 describes — **the expected values must not be computed by the code
under test, nor by a helper sharing its inputs.**

### 3.3 The NemotronH arm

Same table, driven from a synthetic 52-layer NemotronH-shaped `KVCacheConfig`
(no checkpoint, so it runs in CI):

- exactly **6** attention layers at indices **{5, 12, 19, 26, 33, 42}** and **23**
  recurrent layers, asserted as the full index vector, not as counts alone;
- `attn_kv().size() == 6`, not 52 — with the ~8.7x page ratio stated in the test
  so a regression reads as what it is;
- conv shape `{6144, 3}`, SSM shape `{64, 64, 128}`, taken from the spec;
- state dtypes taken from `spec->dtypes`, with §2.7's reconciliation applied.

### 3.4 Mutations — each arm must be proven armed

Applied **alone**, in a scratch copy, rebuilt, run, tree restored byte-for-byte.
**Print the compile exit code and the test-binary sha256 beside every result** —
a mutation that fails to build reads as a passing test, and that has happened
three times in one campaign here
([[mutation-build-failure-reads-as-a-passing-test]]).

| # | Mutation | Must RED |
|---|---|---|
| M1 | conv-state second dim `- 1` in the refactored allocation | Qwen3.5 byte-identity arm **and** NemotronH arm |
| M2 | recurrent state dtype forced to `kBF16`, ignoring `spec->dtypes` | both arms |
| M3 | membership reverted to `config_.layer_types[l] == "linear_attention"` | **NemotronH arm RED, Qwen3.5 arm GREEN** — the asymmetry is the proof the refactor is behaviour-preserving for Qwen3.5. Report it as a pair; a mutation that reds both means Qwen3.5 behaviour changed |
| M4 | membership taken from group **index** instead of `layer_names` | NemotronH arm |
| M5 | the G-SAFE refusal in `ForwardNemotronHForCausalLM` replaced by a fall-through | the interlock test |
| M6 | the `MakeKVCacheMaybeSpec` speculative refusal removed | the §1.2 refusal test |
| M7 | the interlock's `vllm::ModelAs<>` reverted to an unchecked `static_cast` | the interlock test, and the UBSan leg under `-DVLLM_CPP_SANITIZE='address,undefined'`. This mutation belongs to A1 only if #868 has landed; if A1 declares #868 its base, #868 owns its own equivalent and A1 does not restate it |
| M8 (A2) | the carried SSM state zeroed at the start of every step | the A3 multi-step token gate — see §5.2 |
| M9 (A2) | every request's state slot forced to 0 | the A3 batched token gate |
| M10 (A2) | the batch treated as one concatenated causal sequence | the A3 batched token gate |

---

## 4. Blast radius — enumerate before editing

`initialize_kv_cache` is declared at
`include/vllm/v1/worker/gpu/runner.h:386` and called from `runner.cpp:349` and
`:390` (the two real constructors; the two Qwen3.5 convenience constructors at
`:394-409` delegate). The engine builds exactly one runner
(`include/vllm/entrypoints/model_loader.h:468`, constructed
`model_loader.cpp:1007-1023`), so **every architecture flows through it.**

Spec kinds that reach it today, and which the byte-identity arm must not disturb:

| Spec | Producers |
|---|---|
| `FullAttentionSpec`, uniform | ~25 dense archs (`qwen3_5_common.cpp:72`, `llama_registry.cpp:129`, …) |
| `FullAttentionSpec`, per-layer via `per_layer_attn_specs` | Gemma-4 (`gemma4_registry.cpp:236-241`) |
| `MLAAttentionSpec` | DeepSeek-V2/V4, Kimi-K3, Kimi-Linear, MiniCPM3, GLM4-MoE-Lite |
| `MambaSpec` | **only 3**: Qwen3.5/3.6, Kimi-Linear, NemotronH |

This same function was already generalized once, when the first dense model
forced it to stop assuming the Qwen3.5 hybrid topology; `test_runner.cpp:1197-1199`
records that both of the resulting cases *crashed* pre-generalization — one on
the empty-`layer_types` out-of-bounds index that #810 is the second instance of.
**A1 is the third instance of that generalization.** Treat it as a known class.

---

## 5. Gates — and what each one cannot see

Correctness first, always. No throughput number is recorded by this row.

### 5.1 A1's gate — unit, deterministic, no checkpoint

`ctest -R '^test_runner$'` plus `ctest -R '^test_nemotron_h'`, plus the full
`scripts/agent-preflight.sh` — read §5.5 first and subtract the inherited
baseline. This gate runs anywhere, including the local development box; it needs
no GPU and no checkpoint.

It sees shapes, dtypes, counts, byte totals and layer→group routing. **It cannot
see whether the state is ever used** — that is why G-SAFE exists, and A1's gate
must never be reported as evidence that NemotronH runs.

### 5.2 A3's token gate — multi-step and multi-request, deliberately

Two standing lessons bind this gate's design, and both point the same way.

**A token gate cannot see a dtype that is too wide, or a dropped mechanism.** F32
where upstream uses a narrower cache dtype is *more* precise: tokens match and we
quietly move twice the bytes ([`porting.md`](../porting.md)). A mechanism can be
absent while the argmax is unchanged ([`porting-a-model.md`](../porting-a-model.md) §3).

**The SSM state dtype is unobservable with fresh single-leg state.**
`src/vllm/model_executor/models/nemotron_h_forward.h:298-301` records this, and
the converse: the dtype *"becomes observable exactly when a state is carried in,
which is what the two-leg gate exercises and what the W6 paged decode does on
every step"*. The corresponding two-leg unit gate already exists —
`tests/vllm/models/test_nemotron_h_forward.cpp:923-961`, "the SSM dtype is
independent, and it is CARRIED". So:

> **The gate must exercise multi-step decode and multi-request batching. A
> longer prompt is not a substitute.** A single-step or single-request gate is
> structurally blind to the entire class of defect A2 introduces.

A3 asserts:

1. **Multi-step:** all 32 golden tokens per prompt, token-exact — not the first
   token. The parent spec's §6d already matches 3/3 *first* tokens against a
   forward carrying no state at all, which is exactly how little a first token
   proves.
2. **Multi-request:** the three golden prompts submitted **concurrently and
   interleaved**, each required to reproduce its own single-request golden. This
   is the only assertion that can see gap 2 in §1 — a batch treated as one
   causal sequence produces fluent, plausible, wrong output.
3. **M8 must red arm 1; M9 and M10 must red arm 2.** If M8 does **not** red the
   32-token gate, the gate is blind to the carried-state dtype and the row then
   owes a **direct assertion** on the allocated state dtype and byte total
   instead. Record the outcome either way; a mutation that stays green is a
   coverage hole, not a pass.
4. **The memory format is checked explicitly against the oracle**, not inferred
   from matching tokens: conv and SSM state dtypes and total state bytes compared
   against what the pinned oracle reports for the same checkpoint, read from the
   **running** engine's resolved config, not from source
   ([`porting.md`](../porting.md)).
5. **Oracle identity asserted BY COMMIT, aborting on mismatch.** Use
   `vllm-oracle-next`, never `$HOME/venvs/vllm-oracle`, which symlinks to a
   0.25.0 rollback predating `NemotronHMoEDecoderLayer` and fails in a way that
   reads as "the model is unsupported" (parent spec §5a). **Assert the commit,
   not the version string:** `environment.md` records (2026-08-12) that the pin
   reports `0.23.1rc1.dev1511+g555967922`, a `setuptools_scm`
   nearest-ancestor-tag artefact — HEAD *is* `5559679229bc`, and the recorded
   `0.26.0.dev0` string will never match. A version-string assertion here fails
   with nothing wrong. A v1 driver script also needs
   `if __name__ == "__main__":` or EngineCore's spawn re-imports it and the
   failure names neither vLLM nor the caller.

### 5.3 Upstream tests owed in the same change

Per AGENTS.md, upstream's tests are ported in the same change, preserving
parameters, modes, fixtures, tolerances, failure cases and the revision anchor:

- `tests/v1/worker/test_mamba_utils.py:2136`
  `test_ds_conv_layout_bias_gt_0_byte_equal_to_sd` — the DS/SD byte-equality that
  makes §2.6's claim a gate rather than an assertion. **A2.**
- `tests/v1/worker/test_gpu_model_runner.py:1137`
  `test_hybrid_attention_mamba_tensor_shapes` — *"writing a mamba block will not
  corrupt an attention block and vice versa"*. This is the closest upstream
  analogue of A1's byte-identity arm; port its intent. **A1.**
- `tests/v1/attention/test_mamba_update_block_table.py:75`
  `test_update_block_table_copies_block_idx_to_persistent_buffers` and `:178`
  `test_state_indices_tensor_d_includes_num_speculative_blocks` — per-request
  state indexing. **A2.**
- Note for the record: `tests/models/language/generation/test_hybrid.py` does
  **not** list Nemotron-H at the pin (searched `SSM_MODELS`, `HYBRID_MODELS`,
  `HYBRID_MODELS_REQUIRING_CHUNKED_PREFILL` at `:28-58`). There is no upstream
  e2e hybrid case to port for this architecture; A3's gate is the committed
  oracle golden instead. That is a search result with the paths named, not an
  assumption.

### 5.4 The gate hosts (developer-directed 2026-08-15), and the discipline

**The two gate hosts are `dgx.casa` (GB10, sm_121a) and `192.168.68.23` (Jetson
Thor, sm_110).** Both arms of A3 run on both.

**The local x86_64 box is a development arm, not a gate host.** It has no GPU, so
it cannot see anything device-side, and an A3 result from it is not an A3 result.
Run A1's unit gate and the preflight there all you like; do not report a token
gate from it.

**GB10 still provides the CUDA==CPU equivalence check** — it runs both backends,
so nothing is lost by dropping the local box.

**Thor is not a second CUDA host, it is the PORTABLE path.** It builds with
`-DVLLM_CPP_CUDA_ARCHITECTURES=110 -DVLLM_CPP_CUDA=ON -DVLLM_CPP_TRITON=OFF` and
**no cutlass — every sm_110 fast-path cell resolves EMPTY**
(`environment.md`, verified 2026-08-11). So Thor exercises the fallback kernels
that GB10's fast paths hide, which is exactly why the loader row took its
evidence on both, and A3 does the same. A conv/SSM path that only works through
a fast path is a path that does not work.

Thor's operational facts, none of them optional:

- **The GPU is reachable only from inside a container.** No host CUDA toolkit, no
  `nvcc`, no `cmake`. Use `sudo docker` (the user is in group `admin`, not
  `docker`), **`--runtime=nvidia`** — `--gpus all` is refused outright — and
  **`-e NVIDIA_DISABLE_REQUIRE=1`**, because the image's `NVIDIA_REQUIRE_CUDA`
  tops out at `driver<576` and the box runs 595.78.
- **★ THIS BOX REBOOTS INSTEAD OF OOM-KILLING.** `vm.overcommit_memory=1` with
  zero swap: touching memory the kernel could not back takes the whole machine
  down, with `exit=255`, `OOMKilled=false` and no `dmesg` OOM line. Observed three
  times on 2026-08-11. The rule recorded there is **no >25 GB model plus a second
  checkpoint**. The driver checkpoint is 20.1 GiB and A3 loads it **alone** — the
  MTP drafter is parent W5 and out of scope here — so A3 fits, but a future
  spec-on run on this box does not. Size for it before pressing enter.
- No `local-ai-worker` and no `~/gpu.lock` on Thor in its current state; the
  worker-restore discipline below is dgx-only.
- Move code with `git archive`, never rsync ([[dgx-transfer-git-archive-not-rsync]]);
  weights move dgx→Thor with `tar -ch | ssh ... tar -x` into a FRESH directory.
- `nsys` in that image is 2024.2.3 and **cannot trace CUDA here**. Irrelevant to
  A3, which records no number — noted so nobody plans one.

**GPU discipline on dgx**, all mandatory: `flock $HOME/gpu.lock` — the lock is
`$HOME/gpu.lock`, **not** `/tmp/gpu.lock`; `local-ai-worker` parked and restored
at the end; one log per run; never a large oracle alongside `ctest`, because
`gpu_memory_utilization` reserves HOST RAM on GB10 and has OOM-rebooted that box
too; CUDA `ctest` with `-j 1`. `df -h` before and after every build on either
host — an ENOSPC leaves the previous binary in place and prints a green status
([[stale-binary-prints-green-status]],
[[enospc-makes-checkers-emit-false-policy-refusals]]).

### 5.5 The inherited baseline — subtract it, do not chase it

At the time this spec was written, `main` @ **`0141cb4b4`** is **RED on six
gates**, all of them consequences of the `#865` `ci.yml` rewrite and none of them
anything an implementer of this row can cause:

```
check-release-binary-contract     check-release-workflow     check-test-registration
test_check_release_binary_contract   test_release_manifest   test_release_pipeline
```

Reproduced in a detached worktree at `0141cb4b4` **with zero modifications**;
preflight was green on the immediately preceding base `e8f3b4a75` in the same
session on the same box. Filed as
[#873](https://github.com/mudler/vllm.cpp/issues/873), and a repair is being
dispatched separately.

**Subtract them as an inherited baseline; do not chase them, and do not report
them as this row's.** Equally: do not subtract them *blindly*. #775 is the exact
precedent — a lane stayed red long enough that people learned to discount it, and
by the time the cause had changed underneath, anyone subtracting "the known red"
was subtracting the wrong thing. Re-measure at your own base and name what you
actually saw.

**If the #873 repair lands before A1 is claimed, this section is historical
rather than operative** — the correct action is then a clean green preflight, and
a red on any of the six is yours.

---

## 6. The e2e acceptance — this row's stop condition

> **The row is done when `vllm_engine_load` followed by a generate call, both
> made through `include/vllm.h` and nothing else, reproduces the committed
> 32-token goldens at
> `tests/parity/goldens/nemotron_35_lightning_greedy/oracle.json` token-exact
> for all three prompts, with `VT_NEMOTRON35_SNAPSHOT` UNSET so the revision
> check binds, on **both gate hosts — `dgx.casa` (GB10, sm_121a) and
> `192.168.68.23` (Thor, sm_110, no CUTLASS, `TRITON=OFF`)**, per §5.4.**

`VT_NEMOTRON35_SNAPSHOT` is deliberately never revision-checked
(`tests/parity/test_hf_snapshot_pinning.cpp:62`), so setting it during the gate
silently unpins the checkpoint. Unset it and **record the resolved directory** as
evidence. The golden carries the three oracle version strings, the model path,
revision `29f2d174`, and `prompt_token_ids` per prompt — so the pre-tokenized ABI
path needs no tokenizer agreement established first.

### 6.1 The driver — `examples/nemotron_h_gen`, modelled on `kimi_linear_gen`

**Model it on `examples/kimi_linear_gen`** (`examples/kimi_linear_gen/main.cpp`;
`examples/CMakeLists.txt:29-31` links it against `vllm::shared` and nothing
else). That example is a thin public-ABI client — `#include "vllm.h"` only,
`vllm_engine_load` + `vllm_complete_tokens` — and it is **not** listed in
`scripts/example-abi-allowlist.txt`, so a new example built the same way needs
**no allowlist row**. Its header comment records that its former private harness
was deleted once the paged path became production; that is exactly the end state
A2 reaches.

**Do NOT copy `deepseek_v4_gen` or `laguna_gen`.** Both are on that allowlist,
both drive a bespoke forward through internal headers, and the allowlist's own
preamble names them as the transition state it exists to retire. A new example in
their shape would be a **new** internal-reaching example with no entry, which
reds `scripts/check-surface-coverage.py` — the gate doing its job. Satisfying it
by appending a line would be precisely the "CLI-only capability landed silently"
defect it was built to stop.

### 6.2 The routing allowlist entry is part of done

`scripts/runner-routing-allowlist.txt:26` carries a `nemotron_h` entry naming its
own removal condition: the forward is the host reference, it holds K and V for
the whole prompt, pages nothing, and returns `HostLogits`, so *"the device/paged
runner path is W6, which is what removes this entry"*. **A2 removes that line.**
A stale entry for a now-clean model reds the checker, so this is enforced, not
aspirational — and if A2 cannot remove it, A2 has not reached the seam it
claimed.

---

## 7. Risks, and the decisions they force

**R1 — G-SAFE is the whole risk.** An A1 that lands without the interlock
converts a loud refusal into a silently wrong server. §0 is the gate; a reviewer
who cannot point at the interlock returns FAIL.

**R2 — the #775 repair is ALREADY IN FLIGHT as PR
[#868](https://github.com/mudler/vllm.cpp/pull/868); A1 consumes it rather than
redoing it.** An earlier draft of this spec told A1 to fix #775 in flow. That was
wrong, and the correction is recorded rather than quietly applied, because the
way it was wrong is a standing trap: an issue-and-open-PR search that stops at
issues finds #775 OPEN and concludes the work is unclaimed
([[verify-gap-against-open-prs-not-just-issues]]).

The facts as measured. `ForwardNemotronHForCausalLM` does downcast
unconditionally at `nemotron_h_registry.cpp:122`, and `ModelRegistry::Forward`
(`model_registry.cpp:324-327`) performs no type check — §0 is accurate. But
branch `row/FIX-NEMOTRON-H-FORWARD-DOWNCAST-775` (`0923af27c`, merged forward to
`829bbd527`) already carries a complete, gated repair, and PR #868 is open on it.
What it introduces matters to A1 directly:

- **`vllm::ModelAs<Model>` in the shared registry header** — a `dynamic_cast`
  plus a branch, with the refusal authored once and out of line in
  `RaiseModelTypeMismatch`. It mirrors the local `dynamic_cast`-with-refusal
  idiom already used by `kv_cache_spec_registry.cpp`,
  `single_type_kv_cache_manager.cpp` and `attention/backend.cpp`.
- A test handing `reg.factory->forward` a complete `ForeignLoadedModel` and
  requiring a named refusal.
- The rejected alternative, written down: comparing
  `model.registration().architecture` against the expected name answers what the
  *registration claims*, not what the object *is*, and the realistic defect
  carries the right architecture string on the wrong object.

> **Decision: A1 depends on #868 and opens the interlock with
> `vllm::ModelAs<NemotronHLoadedModel>`.** The G-SAFE refusal is a guard placed
> before exactly the member calls #868 makes safe, so writing it on top of
> `ModelAs` is both correct and one line. **Confirm #868's state before claiming
> A1.** If it has landed, use the helper. If it is still open, A1 either waits or
> declares #868 its base and says so in the PR body — it must not reimplement the
> helper, and it must not reintroduce the cast.
>
> **The class is #847, and it is not mechanical.** #868's own message records why
> the 34 sibling sites across 32 model TUs were not swept with it:
> `llama_registry`, `qwen3_5_dense` and `gemma4_registry` each register three
> architectures against one forward, so those sites have no single architecture
> name to refuse under. #847 carries the three candidate answers. A1 references
> it and does not widen into it.

**R3 — A2 is blocked on #496 W2 and the block may still hold.** Re-verify
against the current head and issue state before claiming A2. If the CUDA Mamba2
arm has not landed, A2 stops and reports; a device forward gated only against a
host reference is not the gate this row claims.

**R4 — the conv-state dtype conflict is live, it is A2's to resolve, and the
answer is already decided** (§2.7). `nemotron_h_registry.cpp:167` declares the
page `kBF16`; `nemotron_h_forward.h:284-296` says the host path's conv state must
be f32 because `vt::CausalConv1dFwd` validates it as f32; the scaffold test
asserts bf16 at `test_nemotron_h_scaffold.cpp:652`. Upstream's answer is bf16
(`mamba_utils.py:96-107`), so **A2 gives the kernel a bf16 conv-state arm**. The
risk this entry records is not *which way* — that is settled — but that the
kernel arm is unplanned work an implementer may be tempted to trade away under
schedule pressure. It is not tradeable: a too-wide dtype passes every gate this
row owns. Any f32 fallback carries its one-line reason and its cost in bytes per
token, in §10.

**R5 — the fixture must be the checkpoint the changed path loads.** Pin revision
`29f2d174` explicitly. A repo silently re-quantized under the same name has cost
this project a full campaign ([[unsloth-27b-nvfp4-is-now-fp8]]).

**R6 — `origin/main` moves under a shared checkout.** Merge an immutable SHA,
and **re-run the full gate after merging rather than reading the diff** — a clean
merge is not a merge that builds the behaviour either side had. #818 is exactly
that failure, in this very model's tests, found by re-running the gate.
Never force-push, including `--force-with-lease`.

### 7.1 Stop conditions

- The interlock cannot be expressed — the forward cannot distinguish a
  state-carrying step from a stateless one → **`NEEDS_DECISION`**; do not land A1.
- Reading shapes from the spec changes Qwen3.5's allocated bytes → **stop.** The
  byte-neutrality contract *is* the claim; if it does not hold, the refactor is
  not behaviour-preserving and this spec is wrong, not the test.
- #496 W2 has not landed when A2 is claimed → **stop and report** (R3).
- Mutation M8 stays green → the gate is blind to the carried-state dtype; the
  row owes a direct dtype assertion, recorded as a finding rather than absorbed.
- The goldens cannot be reproduced and the divergence looks like a near-tie →
  **ask the oracle's own top-2 margin FIRST**
  ([[qwen35b-first-request-only-matches-vllm]]) before declaring a defect.
- A `quantized_layers` entry names an algorithm we do not implement → refuse by
  name, record as owed, never silently dequantize.
- Any temptation to add a NemotronH branch to `hf_config.cpp` → **stop.** That is
  §0.2, and taking it means the refactor was not done.

---

## 8. Records owed on landing

Per the AGENTS.md "Public documents" table. A spec commit owes none of these;
these are the obligations of the implementing changes.

| Change | Owes |
|---|---|
| **A1** | nothing in `docs/` — it edits `src/` and `tests/` only and changes no lifecycle state. It **does** owe: this spec's `## Outcome` opened with the byte-identity result and the §3.4 mutation table; a note in the parent spec's W5 section recording the §1.2 refusal and the repair still owed; the §1.3 issue filed; #775/#868 referenced as a dependency, never re-fixed (the class stays with #847) |
| **A2 + A3** | a lifecycle change, so: `docs/STATUS.md`, `docs/BENCHMARKS.md` (pending, failed or void is a result — silence is not), the row spec's `## Now`, and the row + checklist entry + rollup in `.agents/model-matrix.md:285` in the **same** change (`scripts/check-model-checklist.py` enforces the rollup). Plus `docs/FEATURES.md:141`, whose `NemotronHForCausalLM` row currently reads "CPU host forward returns logits … W6 owns the token gate"; the removal of `scripts/runner-routing-allowlist.txt:26`; and `docs/USAGE.md` if `scripts/check-doc-checkpoint.py` classifies the new example as a user-facing surface — note `kimi-linear-gen` is not currently documented there, so run the checker rather than assuming either way |
| **both** | `.agents/issue-index.md` row for #810 (appended by this spec's commit); #810 linked from the spec and the PR body; this spec's `## Outcome` recording what was measured, what was rejected and why, and why each default is what it is |

`.agents/NOW.md` is authored at operator cadence and is **not** a per-row
lifecycle write.

---

## 9. Now

**State at this commit:** spec only. No product code, no lifecycle change. Per
§1.4 the implementation is a **separate** pull request by a **different** agent;
this one carries the spec.

A1 is claimable now against `main` + `bc570da0d`, with PR #868 as its base
(§7 R2). A2 is blocked on #496 W2, and that block must be re-verified rather than
inherited from this sentence.

**Three things to read before the first edit**, because each has already cost
somebody a cycle: §5.5, so the six `#873` gates are subtracted rather than
chased; §5.4, so the token gate is planned for `dgx.casa` and Thor and not for
the local box, which cannot see anything device-side; and §7 R2, so #775 is
consumed rather than re-fixed.

**Next action:** a fresh implementer claims A1 from §1, captures the §3.1 red
first, and lands A1 with the G-SAFE interlock opened through
`vllm::ModelAs<NemotronHLoadedModel>`. A fresh reviewer — never the implementer —
runs the §3.4 mutations and reports M3 as a pair.

## 10. Outcome

Not yet written. Per AGENTS.md this section is filled when the row reaches
`DONE`: what was measured, what was rejected and why, and why each default is
set the way it is — including, explicitly, the conv-state dtype decision of §2.7
and its cost in bytes per token if it diverges from upstream.
