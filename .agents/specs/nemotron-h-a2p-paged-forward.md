# A2-P — NemotronH's forward becomes the runner's forward: paged KV, carried recurrent state, device logits

**Issue:** [#810](https://github.com/mudler/vllm.cpp/issues/810).
**Parent row:** `MODEL-TEXT-nemotron-h-nemotron-hfor-causal-lm`
([#517](https://github.com/mudler/vllm.cpp/issues/517)).
**Governing spec:** [`nemotron-h-abi-e2e.md`](nemotron-h-abi-e2e.md) — this file
owns what that spec's §1 called `A2a` and what the A2-R landing commit
(`598226e96`) renamed **A2-P**.
**Siblings:** [`nemotron-h-a2q1-fp8-mamba.md`](nemotron-h-a2q1-fp8-mamba.md),
[`nemotron-h-a2q2-nvfp4-moe-lmhead.md`](nemotron-h-a2q2-nvfp4-moe-lmhead.md).
**Base:** `origin/main` @ `10002648199cfbbaf1e423f7c80cacb2f4b56366`.
**Consumes:** [#941](https://github.com/mudler/vllm.cpp/issues/941) — the seam
correction this spec is the answer to (§2.3).
**Pinned oracle:** `${VLLM_SOURCE}` = `/home/mudler/_git/vllm` @
`5559679229bc961848b121ccdeaa8fa5d79bec98`, verified at HEAD while writing this
spec, per [`upstream-sync.md`](../upstream-sync.md).
**Lifecycle at this commit:** unchanged. A spec commit changes no lifecycle
state and owes no `STATUS`/`BENCHMARKS`/`NOW` write; §9 records what the
implementing change owes.

**No product code is written by this spec.** Per the governing spec's §1.4 the
implementation is a separate PR by a different agent, and the reviewer is a
third. `row/MODEL-NEMOTRON-H-ABI-A2Q2A` is in flight on `nemotron_h_device.cpp`
and `nemotron_h_weights.cpp` at the time of writing (local, `4b71c16af`,
unpushed), which is a second reason this lands as a spec first.

---

## 0. What A2-P is, in one paragraph

`ForwardNemotronHForCausalLM` still returns
`HostLogits(NemotronHForward(...))` — `nemotron_h_registry.cpp:185-187`. That
host reference recomputes Q/K/V over the whole sequence on every call
(`nemotron_h.cpp:657-659`), runs one dense causal `vt::Attention` over the whole
`[T,·]` (`:671`), rebuilds the recurrent state from scratch each step, and
treats `token_ids` as ONE causal sequence. A2-P replaces it, for the
single-request case, with a forward that reads and writes the runner's paged KV
and its persistent conv/SSM pages. That is the unit that **narrows G-SAFE for
the first time**, that removes `scripts/runner-routing-allowlist.txt:26`, and
that gives A2-R's and A2-Q's device arms their first production caller.

A2-P is the **wiring** unit. It is not the quantization unit: A2-Q1 and A2-Q2
own the FP8 and NVFP4 arms, and A2-P must not absorb either.

---

## 1. Scope — which G-SAFE clauses this unit consumes, and which stay

The interlock, verbatim at `src/vllm/model_executor/models/nemotron_h_registry.cpp:161-170`:

```cpp
VT_CHECK(
    input.attn_kv.empty() && input.gdn_state.empty() && input.num_reqs <= 1,
    "Model architecture NemotronHForCausalLM: the PAGED/BATCHED decode path ...
```

Its own comment at `:158-160` pre-committed the split, and A2-P executes exactly
that half:

| Clause | A2-P | Why |
|---|---|---|
| `input.attn_kv.empty()` | **CONSUMED — the clause is dropped** | A2-P writes K/V into the runner's pages and reads attention out of them |
| `input.gdn_state.empty()` | **CONSUMED — the clause is dropped** | A2-P gathers the conv/SSM rows at step start and scatters them back at step end |
| `input.num_reqs <= 1` | **STAYS. A2-B removes it, not A2-P** | nothing in this unit reorders a batch, splits decodes from prefills, or indexes per-request state by anything but slot 0 |

> **The interlock is NARROWED, never deleted.** After A2-P the refusal reads
> `input.num_reqs <= 1` alone, still names the architecture and still names the
> missing piece — batching, owned by A2-B. **A reviewer who finds the whole
> `VT_CHECK` gone returns FAIL**, and so does one who finds the `num_reqs`
> clause weakened to a warning, a log line or a `#ifdef`.

The narrowed refusal's message is rewritten in the same change. Leaving the
old text beside a two-thirds-smaller predicate is how a message stops describing
what it enforces, which AGENTS.md §"Changing the rules or a checker" is about.

### 1.1 The rest of the scope

| In A2-P | Out of A2-P |
|---|---|
| a paged `NemotronHAttnBlockPaged` over the 6 GQA layers: `ReshapeAndCache` write into `PagedKvCache`, paged attention read | **any** change to the FP8 mamba projections (A2-Q1) or the NVFP4 MoE/`lm_head` arms (A2-Q2) |
| carried conv + SSM state for the 23 Mamba2 layers, gathered from and scattered to `GdnStateCache`, indexed by the metadata's state indices | batching, decode/prefill reordering, `num_reqs > 1` — **A2-B** |
| the persistent conv page stored at the **cache dtype (bf16)**, resolving the governing spec's §2.7 (see §4.4 — the kernel arm it feared already exists) | the MTP head (parent W5), the GGUF k-quant arm (parent W7), `NemotronHPuzzleForCausalLM` |
| `ForwardNemotronHForCausalLM` returning **device-resident** `ForwardLogits` on the `gather_logits` path | `n_groups` TP sharding; any change to the Qwen3.5, Kimi-Linear or Laguna model files |
| removing `scripts/runner-routing-allowlist.txt:26` | any throughput, latency or memory number, on any axis |
| a thin public-ABI example, `examples/nemotron_h_gen`, modelled on `kimi_linear_gen` | prefix caching for the mamba state (upstream's `mamba_cache_mode == "all"`) |
| the A3 end-to-end token gate through `include/vllm.h` (governing spec §6) | speculative decode widening the conv row (`num_spec > 0`) |

**A2-P does not depend on A2-Q1 or A2-Q2.** The host arms for the mamba and MoE
blocks exist and compute today (`nemotron_h_device.cpp:441-454` bounces to them),
so the paged wiring can land against them and each quantized arm then swaps its
block underneath. The reverse is not true, which is why A2-P is the unit that
owns the reachability debt both A2-Q specs list under their `## Owed`.

---

## 2. Upstream anchors — `file:line` on both sides, at `555967922`

Verified against the checkout at the pin (`git rev-parse HEAD` →
`5559679229bc961848b121ccdeaa8fa5d79bec98`). Nemotron-H is still in the old
layout at this pin: `vllm/model_executor/models/nemotron_h.py`.

### 2.1 The attention half

| Behaviour | Upstream | Ours |
|---|---|---|
| the attention class | `nemotron_h.py:409` `class NemotronHAttention(nn.Module):` | `nemotron_h_device.cpp:231` `NemotronHAttnBlock` |
| fused QKV, `bias=False` | `nemotron_h.py:443-451` `QKVParallelLinear(...)` | three `vt::MatmulBT` at `nemotron_h_device.cpp:279-283` (unfused — the checkpoint ships q/k/v separately) |
| `scaling = head_dim ** -0.5` | `nemotron_h.py:441` | `nemotron_h_device.cpp:293`, `nemotron_h.cpp:669` |
| `Attention(...)` construction | `nemotron_h.py:463-472` — **no `rotary_emb`, no `q_norm`/`k_norm`, no `logits_soft_cap`** | `nemotron_h_device.cpp:289-295` |
| the whole forward, four lines | `nemotron_h.py:474-483`: `qkv_proj` → `split` → `self.attn(q, k, v)` → `o_proj` | `nemotron_h_device.cpp:279-301` |
| K/V written to the paged cache **before** attention runs, as a separate op | `layers/attention/attention.py:544-552` `unified_kv_cache_update(key, value, self.layer_name)`, then `:554-561` `unified_attention_with_output(...)` | **the port** — `vt::ReshapeAndCache` into `PagedKvCache`, then paged attention |
| the concrete cache write | `v1/attention/backends/flash_attn.py:1122-1131` `reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype, layer._k_scale, layer._v_scale)` | the shared `dense_attn::KvSlice` (`dense_attn_block.h:233`) builds the two rank-4 K/V views the write and read need |
| slot mapping per token | `gpu_model_runner.py:2190-2194` `block_table.compute_slot_mapping(...)` | `v1::CommonAttentionMetadata::slot_mapping` (`include/vllm/v1/attention/backend.h:111`) |

**★ NO RoPE, AND THAT IS MEASURED.** A case-insensitive grep for
`rotary|rope|q_norm|k_norm|rms_norm_eps` over the whole of `nemotron_h.py` at the
pin returns **exactly one** hit — `:383`, `rms_norm_eps=config.layer_norm_epsilon`,
which is the `MambaMixer2` gated-RMSNorm argument and not attention. `q` and `k`
go straight into `self.attn` at `:481` with zero positional transform.
`NemotronHAttentionDecoderLayer.forward` accepts `positions` (`:516`) and never
uses it.

### 2.2 The recurrent half — what "carried state" means concretely

The single most important structural fact for the port, and the one a naive
implementation gets backwards: **decode rows come FIRST in the token dimension**,
prefill rows second. Every split in the mixer is
`[num_decode_tokens, num_prefill_tokens]`.

| Behaviour | Upstream | Notes for the port |
|---|---|---|
| the real body is `conv_ssm_forward`, not `forward` | `mamba_mixer2.py:687` | `forward` (`:548-586`) only projects, calls the custom op, gates and out-projects |
| per-layer metadata by prefix | `mamba_mixer2.py:703` + `:712` `attn_metadata = attn_metadata_raw[self.prefix]` | ours is per-group, handed in as `input.gdn_meta` |
| conv-state SD/DS orientation | `mamba_mixer2.py:717-721` `conv_state = (self.kv_cache[0] if is_conv_state_dim_first() else self.kv_cache[0].transpose(-1, -2))` | ours is `(dim, state_len)` = upstream's **DS** (`mamba_utils.py:46-48`); no transpose needed |
| token split, decodes first | `mamba_mixer2.py:758-767` | |
| output split, decodes first | `mamba_mixer2.py:808-812` | kernels write in place into the two views; there is no merge step |
| PREFILL conv | `mamba_mixer2.py:829-847` `causal_conv1d_fn(..., conv_states=conv_state, has_initial_state=has_initial_states_p, cache_indices=state_indices_tensor_p, ...)` | ours: `vt::CausalConv1dFwd` |
| PREFILL scan | `mamba_mixer2.py:870-892` `mamba_chunk_scan_combined_varlen(..., dt_softplus=True, dt_limit=(0.0, inf), z=None, state_dtype=ssm_state.dtype)` | ours: `vt::Mamba2ChunkScan`, already used at `nemotron_h.cpp` |
| PREFILL state write-back | `mamba_mixer2.py:977-978` `ssm_state[state_indices_tensor_p] = varlen_states` | the simple, non-prefix-cache branch — **the one A2-P ports** |
| DECODE index selection | `mamba_mixer2.py:1007-1010` — read and write the SAME slots, in place | again the non-cache branch |
| DECODE conv | `mamba_mixer2.py:1013-1025` `causal_conv1d_update(..., conv_state, ..., conv_state_indices=state_indices_tensor_d, ...)` | ours: `vt::CausalConv1dUpdate` |
| DECODE SSM | `mamba_mixer2.py:1087-1103` `selective_state_update(ssm_state, ..., state_batch_indices=..., dst_state_batch_indices=..., out=preallocated_ssm_out_d)` | ours: `vt::Mamba2StateUpdate` |
| continuation detection | `mamba_attn.py:554-556` `has_initial_states_p = (num_computed_tokens[num_reqs - num_prefills : num_reqs] > 0)` | |
| per-request state index comes from the BLOCK TABLE | `mamba_attn.py:513-518` `mamba_get_block_table_tensor(...)`, split at `:523-532` | **not** from a slot map |
| mamba groups get NO token→slot mapping | `gpu_model_runner.py:7239-7242` `SlotMappingMode.NONE`; the early return is `block_table.py:160-163` | the attention group still gets one |
| decode threshold | `mamba_attn.py:87` `reorder_batch_threshold: int = 1` | A2-B's concern, not A2-P's |

The metadata fields, for the mapping onto `v1::GDNAttentionMetadata`:
`has_initial_states_p` (`mamba_attn.py:39`), `state_indices_tensor_p` (`:42`),
`state_indices_tensor_d` (`:47`), `query_start_loc_p` (`:40`), `seq_lens` (`:62`),
`cu_chunk_seqlen_p` (`:68`), `last_chunk_indices_p` (`:71`);
Mamba2 adds `prep_initial_states`, `chunk_size`, `seq_idx_p` (`mamba2_attn.py:105-111`).
The varlen chunk metadata builder is `mamba2_attn.py:22-88`, and its pure-CPU
loop at `:56-68` is the cleanest port oracle in the file.

### 2.3 ★ NOT `dense_attn::AttnBlock` — the trap is filed, measured, and already avoided

The governing spec's §2 named `dense_attn::AttnBlock` as NemotronH's device
attention seam. **It cannot serve this architecture**, and
[#941](https://github.com/mudler/vllm.cpp/issues/941) filed the reason. Measured
again at this base, because a filed root cause is a hypothesis until re-read:

| Fact | Anchor |
|---|---|
| the block is Qwen3-shaped and config-driven | `include/vllm/model_executor/models/dense_attn_block.h:335-338` |
| **it applies RoPE unconditionally** | `dense_attn_block.h:497` `vt::RopeNeox(d.q, q3, k3, si.positions.t(), MakeRopeArgs(cfg));` — the `:490-493` branch above it only selects *which* rope implementation |
| **it reads an eps this checkpoint does not ship** | `dense_attn_block.h:345` `const float eps = static_cast<float>(cfg.rms_norm_eps);` ← `src/vllm/transformers_utils/hf_config.cpp:551` `cfg.rms_norm_eps = GetDouble(text, "rms_norm_eps", 0.0);`, and Nemotron-3.5-Lightning ships `layer_norm_epsilon`/`norm_eps` and no `rms_norm_eps` |
| **there is no rope-free entry point** | the header's entire public surface is `FusedChainAdoptEnabled:61`, `RopeCacheEnabled:89`, `MakeRopeArgs:105`, `Qwen3QkvMergeEnabled:150`, `MergedQkvEnabled:161`, `WeightF32:168`, `ResidentWeight:178`, `ResidentWeightF32:211`, `KvSlice:233`, `BuildStepInputs:266`, `AttnBlock:335`. No variant, no flag |
| **`rotary_dim == 0` aborts rather than bypassing** | `src/vt/ops.cpp:1427-1429` `VT_CHECK(args.rotary_dim > 0 && ...)` |

A2-R already took the right road: `NemotronHAttnBlock`
(`nemotron_h_device.cpp:231-233`) is model-local, in the established
`granite.cpp:84` / `gemma4.cpp:206` idiom, and its `:43-60` header records the
three reasons. **A2-P extends that block; it does not migrate to the shared
one.** It still reuses the shared residency seam
(`dense_attn::ResidentWeight`, `dense_attn_block.h:178`) and the shared K/V view
helper (`dense_attn::KvSlice`, `:233`) — the parts that are not rope-shaped.

> **Correction owed in the same change.** `nemotron_h_device.cpp:56` cites
> `dense_attn_block.h:496` for the `RopeNeox` call. It is **`:497`**; `:496` is a
> comment. `nemotron_h_registry.cpp:141` cites `nemotron_h.cpp:585-630` for
> `NemotronHAttentionMixer`; it actually begins at **`:631`**. Both anchors
> predate A2-P and both drift further with every edit to those files. Fix them
> where A2-P is already editing, and do not copy either into new prose.

---

## 3. Design

### 3.1 The forward's shape, and the branch predicate

Mirror `ForwardKimiLinearForCausalLM` (`kimi_linear_registry.cpp:88-112`), which
is the only in-tree instance of exactly this fold. Its predicate is at
**`:99-100`** and it has **three** clauses, not the two the NemotronH comment
quotes:

```cpp
if (input.gather_logits && weights.resident.resident && !input.attn_kv.empty() &&
    !input.gdn_state.empty()) {
  return KimiLinearModel::ForwardPaged(input, weights);
}
```

Two consequences bind A2-P:

1. **The paged branch passes `input` WHOLE** (`ForwardPaged(input, weights)`).
   Only the two non-paged branches decompose into arguments, and neither of them
   can see `gdn_meta`, `gdn_state`, `gdn_state_slots`, `num_reqs` or
   `pure_decode` at all. A2-P's `NemotronHModel::ForwardPaged` takes
   `const ModelForwardInput&`. Do not invent a 12-argument signature.
2. **The residency clause is part of the idiom.** A2-P's equivalent predicate is
   `input.gather_logits && <weights are resident> && !input.attn_kv.empty() &&
   !input.gdn_state.empty()`. Mirroring only the two-clause fragment quoted at
   `nemotron_h_registry.cpp:145-147` would diverge from the cited idiom and would
   route a non-resident model into a device path with nothing uploaded.

The host reference stays below the fold, unchanged, as Kimi-Linear's does. It is
the operand the numeric gate compares against, and deleting it deletes the gate.

### 3.2 The paged attention half

Per GQA layer `l ∈ {5, 12, 19, 26, 33, 42}`, with `PagedKvCache` from
`input.attn_kv[i]` (`qwen3_5.h:61-68`; the runner allocates and owns it —
`qwen3_5.h:59-60`, built at `runner.cpp:906-916`):

1. project Q/K/V (`nemotron_h_device.cpp:279-283`, unchanged);
2. **no rope, no qk-norm** — the projection output is what attention sees;
3. build the two rank-4 K/V views with `dense_attn::KvSlice`
   (`dense_attn_block.h:233`) — the raw buffer is rank 5
   `(num_blocks, 2, block_size, num_kv_heads, head_size)` and `vt::kMaxRank` is
   4, which is why the struct carries dims rather than a tensor
   (`qwen3_5.h:44-46`);
4. write this step's K/V at `input.attn_meta.slot_mapping`
   (`backend.h:111`), down-cast to the cache dtype before the write — the "auto"
   `ReshapeAndCache` copy requires `cache dtype == k/v dtype` (`qwen3_5.h:47-49`);
5. run paged attention over `block_table_tensor` / `seq_lens` / `query_start_loc`
   (`backend.h:107-108`, `:89-90`, `:81-82`), causal, scale `Dh^-0.5`.

The fp8 K/V scales the loader materialized are **still not selected**:
`FullAttentionSpec` is built with `v1::ResolveKvCacheDType()` at
`nemotron_h_registry.cpp:239`, and A2-P does not change that. An fp8 KV page is a
separate decision with its own gate; taking it here would be a too-narrow dtype
that the token gate also cannot see, in the opposite direction from §5.2's
too-wide case.

### 3.3 The recurrent half

Per Mamba2 layer, with `GdnStateCache` from `input.gdn_state[g]`
(`qwen3_5.h:76-82`; `ssm_state [num_state_blocks, ...]`,
`conv_state [num_state_blocks, conv_dim, K-1]`, both updated in place; built at
`runner.cpp:970-978` through the rank-general `slot_major_view` lambda at
`:953-969`, which is the mirror of `mamba/abstract.py:38-43`).

Mirror `GdnBlockPaged` (`qwen3_5.cpp:4345-4348`) statement for statement — it is
the same shape at a different recurrence:

| Step | Qwen3.5 anchor | A2-P |
|---|---|---|
| validate the cache layout before touching it | `qwen3_5.cpp:428-441 ValidateGdnStateCacheLayout` | a NemotronH twin; refuse by name on a rank or slot-count mismatch |
| decode: conv update IN PLACE, indexed | `qwen3_5.cpp:4556-4560` `vt::CausalConv1dUpdate(..., conv_cache, ..., &gidx)` | `vt::CausalConv1dUpdate` |
| decode: SSM update IN PLACE, indexed | `qwen3_5.cpp:4655-4659` `vt::GdnDecode(..., ssm_cache, ..., &gidx)` | `vt::Mamba2StateUpdate` |
| prefill: gather + zero fresh rows in one op | `qwen3_5.cpp:4684-4686` `vt::GdnStateGather(..., &has_initial)` | the same helper |
| prefill: scatter back | `qwen3_5.cpp:4713-4716` `vt::GdnStateScatter(...)` | the same helper |
| the per-step slot vector, uploaded once, shared by every recurrent layer | `qwen3_5.cpp:3897`, built `:3995-3997`, narrowed for decode at `:4557` | the same shape; at `num_reqs == 1` it is one element |

> **★ THE ZEROING OBLIGATION IS A CALLER'S, AND IT IS THE LOUDEST TRAP HERE.**
> `include/vllm/v1/attention/backends/gdn_attn.h:126-139` states it in the
> header, in a `⚠ CALLER OBLIGATION` block: the recurrence kernels read the state
> buffer **unconditionally**, with no `has_initial_state` gate, so a request whose
> `prefill_has_initial_state == 0` reads a **stale mamba block** and produces
> silent wrong output. Upstream does the same zeroing at
> `mamba_mixer2.py:854-866` (`torch.where(has_initial_states_p[...], ssm_state[...], 0)`).
> Mutation M4 exists for exactly this, and it is the mutation most likely to
> survive a gate that was not designed for it.

Note the local sentinel deviation, already recorded at `gdn_attn.h:65-72`: our
cache ABI uses a **negative** `kNullStateSlot = -1` (`:72`), not upstream's
block-0 reservation. A2-P inherits it and does not re-litigate it.

### 3.4 What the runner already hands over

Nothing new is owed on the runner side. A1 finished it, and this is measured
rather than assumed:

| Field | Set at |
|---|---|
| `.attn_kv = attn_kv_` | `runner.cpp:1371` |
| `.gdn_state = gdn_state_` | `runner.cpp:1372` |
| `.num_reqs` | `runner.cpp:1376` |
| `.gdn_state_slots` | `runner.cpp:1377`, sized `runner.cpp:513-515` |
| `.pure_decode` | `runner.cpp:1378`, computed `runner.cpp:1341-1342` as `attn_meta.num_actual_tokens == num_reqs && gdn_meta.num_prefill_tokens == 0` |
| the call | `runner.cpp:1465` `ModelRegistry::Forward(*model_, forward_input)` |
| per-layer recurrent membership, from `KVCacheGroupSpec::layer_names` | `runner.cpp:761-778` via `GroupLayerMask` (`:363-364`) |

**If a field A2-P needs is absent, that is a finding about A1, not a licence to
compute it locally from `config_`.** Re-deriving a per-layer signal from the HF
config is #810 itself, one layer up, and §0.2 of the governing spec forbids it by
name.

### 3.5 Device logits, and the allowlist entry

`scripts/runner-routing-allowlist.txt:26` names its own removal condition: *"The
device/paged runner path is W6, which is what removes this entry"*. Concretely,
`scripts/check-runner-routing-consistency.py` asks three things, and A2-P must
satisfy all three:

- **(a) on-device logits** — the default `gather_logits` forward returns a
  device-resident `ForwardLogits`, not `HostLogits`. `NemotronHDeviceForward`
  currently ends in the host `lm_head` (`nemotron_h_device.cpp:481-503`, refusal
  at `nemotron_h.cpp:1031-1034`), so **on `origin/main` A2-P cannot satisfy (a)
  alone**: `lm_head` is NVFP4 W4A16 and its device arm is A2-Q2's.
- **(b) no private host generate loop** — satisfied; there is none.
- **(c) bf16-resident activations** — A2-R's residual stream is already `DBuf`s.

> **★ ORDERING DECISION, and it is the one thing in this spec an implementer
> must check before starting.** A2-P removes the allowlist entry only when the
> forward returns device logits, which needs A2-Q2's `lm_head` arm. **Re-verify
> `row/MODEL-NEMOTRON-H-ABI-A2Q2A` against `origin/main` at claim time** (§8 R1).
> If it has landed, A2-P removes the entry in the same change and the checker
> proves it. If it has not, A2-P **narrows** the entry — stating that the forward
> is now paged and runner-routed and that only the host `lm_head` remains — and
> names A2-Q2 as what removes it. Narrowing a stale entry is the same discipline
> as narrowing G-SAFE. **Deleting the entry while still returning `HostLogits`
> reds the checker, and satisfying the checker by widening the allowlist is the
> defect it exists to stop.**

### 3.6 The example, and why it must be the thin one

`examples/nemotron_h_gen`, modelled on `examples/kimi_linear_gen/main.cpp` —
verified thin: its only project include is `#include "vllm.h"` (`:29`), it links
`vllm::shared` and nothing else (`examples/CMakeLists.txt:29-31`), and it does
not appear in `scripts/example-abi-allowlist.txt`. It drives
`vllm_engine_load` (`:133`) and `vllm_complete_tokens` (`:177`, `:196`).
`vllm_complete_tokens` exists at `VLLM_ABI_VERSION 21` (`include/vllm.h:669`,
added at v13, `:106`), so **A2-P grows no ABI**.

**Do NOT copy `deepseek_v4_gen` or `laguna_gen`.** Both are on the ABI allowlist
(`example-abi-allowlist.txt:25-26`), both reach internal headers, and the
allowlist's preamble names them as the transition state it exists to retire. A
new example in their shape reds `scripts/check-surface-coverage.py`, and
appending a line to satisfy it is the "CLI-only capability landed silently"
defect the checker was built to stop.

---

## 4. Four decisions this spec makes so implementation does not

### 4.1 Single request means slot 0, not "no slots"

At `num_reqs == 1` the state-index vector has exactly one element and the block
table exactly one row. **Index through them anyway.** A forward that hardcodes
slot 0 passes every gate A2-P owns and then fails silently under A2-B, and the
mutation that would have caught it (M5) cannot fire because there is nothing to
mutate. The indexing machinery lands here; only the *count* is one.

### 4.2 Prefill and decode are both A2-P's, and the split is by token count

`pure_decode` (`runner.cpp:1341-1342`) distinguishes them. A single request still
takes both paths — step 1 is a prefill of `T` tokens, steps 2..32 are decodes of
one — so A2-P ports both branches of `conv_ssm_forward` and both branches of the
attention path. What it does **not** port is the interleaving: at `num_reqs == 1`
one step is entirely prefill or entirely decode, never both. The decodes-first
ordering (§2.2) is nonetheless implemented as upstream states it, because
retrofitting an ordering convention under A2-B is how the two halves disagree.

### 4.3 The conv layout stays DS, and the upstream byte-equality test comes with it

Ours is `(dim, state_len)` at `nemotron_h_registry.cpp:268`, which is upstream's
**DS** (`mamba_utils.py:46-48`, selected by `VLLM_SSM_CONV_STATE_LAYOUT`,
default `SD`). The bytes are the same product either way. The comment at
`nemotron_h_registry.cpp:250-255` still calls DS "our local convention"; A2-P
sharpens it to name the upstream mode, and ports
`tests/v1/worker/test_mamba_utils.py:2136`
`test_ds_conv_layout_bias_gt_0_byte_equal_to_sd` — **verified at the pin as a
method of `class TestPostprocessMambaFusedKernel` (`test_mamba_utils.py:410`)**,
not a module-level function, which matters to whoever ports its fixture.

### 4.4 ★ The conv-state dtype: the kernel arm the governing spec feared ALREADY EXISTS

The governing spec's §2.7 and R4 decided the persistent conv page is **bf16** and
warned that giving `vt::CausalConv1dFwd` a bf16 conv-state arm was "unplanned
work an implementer may be tempted to trade away". **Measured at this base: that
work landed on 2026-08-09, six days before that spec was written.**

`src/vt/ops.cpp:1644-1650`:

```cpp
VT_CHECK(conv_state.dtype == DType::kF32 ||
             (conv_state.dtype == DType::kBF16 && conv_backend != nullptr &&
              conv_backend->SupportsCompressedConvState()),
         std::string(name) +
             ": conv_state must be f32, or bf16 on a backend whose conv kernels "
             "support a compressed state in place ...");
```

and the predicate is answered `true` by CUDA (`src/vt/cuda/cuda_backend.cu:117`),
Vulkan (`src/vt/vulkan/vulkan_backend.cpp:142`) and ROCm
(`src/vt/rocm/rocm_backend.hip:333`). It landed with `908bad0ac`
(*"perf(vulkan): GDN state I/O in place"*). CPU answers the
`include/vt/backend.h:186` default `false`.

> **Decision, and it is now cheap rather than a campaign.** The persistent conv
> page stays **bf16**, exactly as `MakeNemotronHKVCache` already declares it
> (`nemotron_h_registry.cpp:222`), and on both gate hosts it is passed to the
> conv kernels **directly, with no widening**. On the CPU backend the caller does
> the f32 gather/compute/scatter around a page that is still bf16 — which is
> precisely what `ops.cpp:1641-1642` says the alternative is, and what
> `qwen3_5.cpp:4564-4570` already implements as its non-indexed fallback.
>
> **The page is never widened to f32 to satisfy a precondition.** An f32
> persistent conv page is the too-wide dtype AGENTS.md names, every gate this
> row owns is blind to it, and there is now no kernel reason for it. If an
> implementer finds one anyway, that is a `NEEDS_DECISION` with the measurement,
> not a quiet widening.

The transient per-call f32 conv buffer inside the host reference
(`nemotron_h.cpp:493-498`) is untouched; it is not a page and A2-Q1 §1.1 already
says so.

This entry is also a record of the failure mode, not only its fix: the governing
spec asserted a kernel gap that a `grep` for `SupportsCompressedConvState` would
have refuted at the time it was written. **Re-measure a stated blocker at your
own base.**

---

## 5. Gates

Correctness first. **A2-P records no throughput, latency or memory number on any
axis.** The row's speed work does not begin until this gate is green.

### 5.1 The RED-first test enters through `ModelRegistry::Forward`

AGENTS.md §"Nothing lands dead" is explicit that a unit test constructing the
type by hand proves the class works and never that anything reaches it. Today
`NemotronHDeviceForward` has exactly **one** non-declaration call site in the
whole tree — `tests/vllm/models/test_nemotron_h_forward.cpp:1805` — which is the
test-only-driver shape `reachability.md` names.

> **The smallest failing test constructs a `GPUModelRunner` from a NemotronH
> `KVCacheConfig` and calls `ModelRegistry::Forward`.** Not
> `NemotronHDeviceForward`, not `NemotronHAttnBlock`, not a fabricated
> `ModelForwardInput`.

The red is cheap and it is already available: on `origin/main` that call reaches
`nemotron_h_registry.cpp:161` and refuses by name with *"the PAGED/BATCHED decode
path is not ported"*, because the runner hands it non-empty `attn_kv` and
`gdn_state`. **Capture that transcript and put it in the PR body.** A test never
seen failing has proven nothing.

Three arms, all through that entry point:

1. **multi-step, single request** — a synthetic small NemotronH-shaped config,
   prefill then ≥ 3 decode steps, asserting the paged result equals the host
   reference's result for the same token sequence. This is the arm that can see
   a dropped carry.
2. **per-block numeric equivalence** — `NemotronHTrace`
   (`nemotron_h_forward.h:389-397`) compared against the host arm at every one of
   the 6 attention and 23 mamba layers. Numeric, not token-only. Bands **measured
   in the case**, per §5.3.
3. **the A3 end-to-end token gate** — §5.4.

### 5.2 What each gate CANNOT see

State it in the spec so nobody reports a gate as evidence of something it cannot
observe:

| Gate | Blind to |
|---|---|
| the token gate | **a dtype that is too WIDE.** An f32 conv or SSM page is *more* precise: tokens match, goldens pass, and the path moves twice the bytes. §4.4 is settled here for exactly this reason |
| the token gate | **a dequant fallback.** A silently dequantized NVFP4 expert produces correct tokens; only the memory format and the load accounting can see it |
| the token gate | **a dropped mechanism whose argmax is unchanged.** `porting-a-model.md` §3; this is why arm 2 exists |
| any single-step gate | **everything A2-P adds.** With fresh state and one leg the SSM dtype is unobservable — `nemotron_h_forward.h:346-349` says so outright, and the two-leg unit gate at `test_nemotron_h_forward.cpp:923-961` exists because of it |
| any single-request gate | cross-request contamination, which is A2-B's to prove and which A2-P's surviving `num_reqs <= 1` clause refuses instead |
| the per-block numeric gate | whether anything **reaches** the block. That is M7 |
| `scripts/check-runner-routing-consistency.py` | whether the paged path is *correct* — it is a routing floor, not a proof |

**The memory format is checked explicitly against the oracle**, not inferred from
matching tokens: conv and SSM page dtypes and total state bytes compared against
what the pinned oracle reports for the same checkpoint, read from the **running**
engine's resolved config rather than from source.

### 5.3 Bands are MEASURED, and the guard is a PROPERTY

Two lessons this row has already paid for, and both bind here. A bf16 band of
`3e-2` once sat *above* a `2.11e-2` defect and accepted a fully rope'd answer. A
"safety factor" guard compared two compile-time constants and observed nothing
about the running system. `DevRelFor(dt)`
(`test_nemotron_h_forward.cpp:1596`) is the repaired model — `1e-5` f32,
`4e-3` bf16, each derived from what the two arms actually agree to with the band
driven to `1e-9`.

> **Derive every band from a measurement taken in the case. Make the guard a
> property: the perturbed answer, run through the same arithmetic that accepted
> the real one, must come out rejected. No stored twin, no invented safety
> factor.**

### 5.4 The A3 end-to-end token gate

The golden is verified present and its shape measured:
`tests/parity/goldens/nemotron_35_lightning_greedy/oracle.json` —
`vllm 0.23.1rc1.dev1511+g555967922`, model
`nemotron-3.5-lightning-30b-nvfp4`, `revision 29f2d1746d8f41e316523194b19018707749b1b1`,
`sampling {temperature: 0.0, max_tokens: 32}`, `golden` a list of **3** entries
each carrying `prompt`, `prompt_token_ids`, `token_ids` (**32 for all three**)
and `text`. The pre-tokenized ABI path therefore needs no tokenizer agreement
established first.

> **DONE means:** `vllm_engine_load` followed by `vllm_complete_tokens`, both
> through `include/vllm.h` and nothing else, reproduces all **32** golden tokens
> for all **3** prompts, token-exact, with `VT_NEMOTRON35_SNAPSHOT` **UNSET** so
> the revision check binds (`tests/parity/test_hf_snapshot_pinning.cpp:62`), and
> the resolved checkpoint directory recorded as evidence.

**All 32 tokens, not the first.** The parent spec's §6d already matched 3/3
*first* tokens against a forward carrying no state at all, which is exactly how
little a first token proves.

The oracle identity is asserted **by commit, aborting on mismatch** — use
`vllm-oracle-next`, never `$HOME/venvs/vllm-oracle` (a 0.25.0 rollback that fails
in a way reading as "the model is unsupported"). The pin reports
`0.23.1rc1.dev1511+g555967922`, a `setuptools_scm` nearest-ancestor-tag artefact;
a version-string assertion against `0.26.0.dev0` fails with nothing wrong. A v1
driver script also needs `if __name__ == "__main__":` or EngineCore's spawn
re-imports it and the failure names neither vLLM nor the caller.

### 5.5 Upstream tests owed in the same change

Per AGENTS.md, preserving parameters, modes, fixtures, tolerances, failure cases
and the revision anchor. All three verified to exist at the pin with the names
and lines below:

- `tests/v1/worker/test_mamba_utils.py:2136`
  `test_ds_conv_layout_bias_gt_0_byte_equal_to_sd` — a **method** of
  `TestPostprocessMambaFusedKernel` (`:410`). Makes §4.3's DS/SD claim a gate.
- `tests/v1/attention/test_mamba_update_block_table.py:75`
  `test_update_block_table_copies_block_idx_to_persistent_buffers`.
- `tests/v1/attention/test_mamba_update_block_table.py:178`
  `test_state_indices_tensor_d_includes_num_speculative_blocks` — port its
  *intent*; `num_spec == 0` here, so the assertion is that the decode slot vector
  is one column wide and indexed, not hardcoded.

**There is no upstream e2e hybrid case to port for this architecture, and that is
a searched result with the paths named.** `tests/models/language/generation/test_hybrid.py`
lists Nemotron-H in neither `SSM_MODELS` (`:28-34`) nor `HYBRID_MODELS`
(`:36-43`); a case-insensitive grep for `nemotron` over the whole file returns
zero. The only Nemotron-H-specific test at the pin is
`tests/model_executor/test_nemotron_h_quantization.py:7`. A3's committed oracle
golden is the substitute.

### 5.6 Mutations

Applied **alone**, in a scratch copy, rebuilt, run, tree restored to the
**baseline sha** — the restore is the control that catches `shutil.copy2`
preserving mtime so ninja skips the rebuild.

| # | Mutation | Must RED |
|---|---|---|
| P-M1 | the carried SSM state zeroed at the start of every step | arm 1 (multi-step) and the A3 token gate |
| P-M2 | the carried CONV state zeroed at the start of every step | arm 1 and A3 |
| P-M3 | K/V written to the paged cache but attention read from a freshly recomputed dense K/V | arm 2 (per-block numeric) at every attention layer |
| P-M4 | the fresh-request state zeroing dropped (`gdn_attn.h:126-139`) | arm 1 on the FIRST step after a state slot is reused. **If this survives, the gate is blind to the loudest trap in §3.3 and the row owes a direct assertion on the zeroed rows instead** |
| P-M5 | the state slot index replaced by a literal `0` | arm 1 must stay GREEN (it is slot 0) and the **assertion on the indexed read** must RED. Report as a pair; a mutation that reds arm 1 at `num_reqs == 1` means the harness is not the one described |
| P-M6 | the conv page dtype widened to f32 | the §5.2 memory-format assertion. **The token gate must NOT red** — that asymmetry IS the demonstration that a token gate cannot see a too-wide dtype, and it is worth reporting as a pair rather than as a failure |
| P-M7 | the production call site deleted — `ForwardNemotronHForCausalLM`'s paged branch removed, host arm left in place | the focused gate must RED. A gate that stays green without the call site measures a class, not a capability (AGENTS.md §"Nothing lands dead"; method at `reachability.md`) |
| P-M8 | the narrowed G-SAFE `num_reqs <= 1` clause replaced by a fall-through | the interlock test |
| P-M9 | the `pure_decode` branch inverted (decode path taken on prefill) | arm 1 |

**Report per mutation:** the exact `[doctest] test cases:` / `assertions:` /
`Status:` lines, a **non-zero case count**, `git diff --stat` proving the edit
applied, compile exit AND error count, and a binary sha256 distinct from
baseline. Three separate green-but-proves-nothing shapes have landed in this
tree — a mutation that failed to build, a mutation that never applied, and a
`-tc` filter that selected zero cases and printed `SUCCESS!`.

**Never put a comma in a `TEST_CASE` name.** `-tc` splits on commas, selects zero
cases, prints `SUCCESS!` and exits 0.

### 5.7 The gate hosts, and the discipline

**`dgx.casa` (GB10, sm_121a) is the primary gate host.** Verified at the time of
writing: `/usr/local/nas_share/checkpoints/nemotron-3.5-lightning-30b-nvfp4` and
`-gguf` both present; `CHECKPOINT_ROOT=/usr/local/nas_share/checkpoints` (`.env`);
119 GB total host memory.

Mandatory, none of it optional:

- **`flock $HOME/gpu.lock`, blocking**, around every heavy job — not only the ones
  that touch the GPU. `GPU_LOCK=$HOME/gpu.lock` (`.env:26`), **not**
  `/tmp/gpu.lock`. `echo >` is not a mutex.
- **Check `free -g` INSIDE the locked region.** A blocking flock proves the
  previous holder released; it never proves the box recovered. At the time of
  writing the box showed 76 GB used and 43 GB available with the lock free.
- **`-j 4`**, and CUDA `ctest` with `-j 1`.
- **The box OOM-REBOOTS rather than OOM-killing.** `gpu_memory_utilization` does
  not bound host RAM on GB10; both `nvidia-smi` and the flag are blind to it.
  Never run a large oracle alongside `ctest`.
- `df -h` before and after every build. An ENOSPC leaves the previous binary in
  place and prints a green status, and it also makes record checkers emit false
  policy refusals.
- Park and restore `local-ai-worker`; one log per run; never hang holding the
  lock.
- Assert `vllm.__file__` on the oracle venv — a copied editable venv imports the
  tree it was BUILT in.

**Thor (`192.168.68.23`, sm_110) is the portable leg, and it is NOT a substitute
for Marlin work.** [#962](https://github.com/mudler/vllm.cpp/issues/962) has
`marlin-nvfp4` disagreeing with itself there — `test_ops_moe_grouped.cpp:1144`,
`bitdiff=15/32768` on an ENABLED kernel. A2-P's own additions (paged attention,
conv/SSM state I/O) are not Marlin, so the Thor leg is meaningful for **them**;
but the moment the end-to-end gate runs the NVFP4 `lm_head` or MoE arms, the Thor
result is **PENDING #962** and is recorded as such rather than quoted. Thor also
reboots instead of OOM-killing, with `vm.overcommit_memory=1` and zero swap.

**The local x86_64 box is a development arm, not a gate host.** Run the unit gate
and the preflight there; an A3 result from it is not an A3 result.

### 5.8 Baseline — measured, not inherited

Measured 2026-08-16 in this worktree at
`10002648199cfbbaf1e423f7c80cacb2f4b56366`, local x86_64 dev box:
`scripts/agent-preflight.sh` green on every record gate **except** two, and both
are named:

- `role-undeclared` — an artefact of claiming the role in a different worktree;
  it clears on `agent-role.py claim` in the working tree.
- `test_cpu_x86_llamacpp_floor` — `NO_QUIET_WINDOW` at loadavg 22.47, which is
  the harness **refusing to measure** rather than a defect.
  [#618](https://github.com/mudler/vllm.cpp/issues/618).

**The #873 gates are FIXED.** A red on `check-release-binary-contract`,
`check-release-workflow` or `check-test-registration` is the implementer's, not
inherited. Genuinely inherited: `windows-msvc-cpu` / `windows-msvc-vulkan` on
every PR ([#584](https://github.com/mudler/vllm.cpp/issues/584),
[#968](https://github.com/mudler/vllm.cpp/issues/968) — they have no `main`
baseline, so they are PR-only red), `test_async_llm` under `ctest -j`
([#294](https://github.com/mudler/vllm.cpp/issues/294) — re-run serially before
calling it a regression), `test_engine_core_proc` under `-j4`
([#1052](https://github.com/mudler/vllm.cpp/issues/1052)), and on GB10
`test_qwen3_5_gdn_spec_routing` 119/123 and `test_linear_method` 83/85
([#907](https://github.com/mudler/vllm.cpp/issues/907)).

**Re-measure at your own base.** #775 is the precedent: a lane stayed red long
enough that people learned to discount it, and by the time the cause had changed
underneath, anyone subtracting "the known red" was subtracting the wrong thing.

---

## 6. ★ G-SAFE after A2-P — what a reviewer checks

```cpp
// BEFORE (origin/main, nemotron_h_registry.cpp:161-170)
VT_CHECK(input.attn_kv.empty() && input.gdn_state.empty() && input.num_reqs <= 1, ...)

// AFTER A2-P
VT_CHECK(input.num_reqs <= 1, ... "batched decode is not ported (issue #810, A2-B)" ...)
```

Three things must all be true, and a reviewer who cannot point at each returns
FAIL:

1. the `num_reqs <= 1` clause is present, is a `VT_CHECK`, and refuses by name;
2. its message names the architecture **and** the missing piece (batching, A2-B),
   and no longer claims the paged path is unported;
3. the same diff lands the paged forward that makes clauses 1 and 2 removable —
   AGENTS.md's G-SAFE condition (a): *the same commit lands the device/paged
   forward that consumes `attn_kv` / `gdn_state` / `gdn_meta` /
   `gdn_state_slots`.*

`ModelAs<NemotronHLoadedModel>` (`nemotron_h_registry.cpp:177`) stays. The guard
runs before it, reads only `input`, and never touches `model` — #775's guarantee
is untouched and A2-P does not reintroduce a `static_cast`.

---

## 7. Reachability — A2-P is what ENDS the unreached posture

Both A2-Q specs list under `## Owed`: *"the device arm has no production caller
until A2-P wires it through `ModelRegistry::Forward`"*. Measured at this base,
that is still exactly true — `NemotronHDeviceForward` has one non-declaration
call site in the tree, `tests/vllm/models/test_nemotron_h_forward.cpp:1805`, and
`ForwardNemotronHForCausalLM` still routes to the host `NemotronHForward`
(`nemotron_h_registry.cpp:185-187`).

**A2-P's PR body names the chain explicitly**, entry point down to the change:
`include/vllm.h::vllm_complete_tokens` → the engine → `GPUModelRunner` →
`runner.cpp:1465 ModelRegistry::Forward` → `ForwardNemotronHForCausalLM` → the
paged branch → `NemotronHAttnBlockPaged` / the recurrent block. Mutation P-M7 is
the proof, not the narrative.

If any part of the device arm still has no caller after A2-P — the most likely
candidate being an A2-Q2 `lm_head` path that lands separately — that is stated in
the commit body and the PR body with the owning row and the issue, and listed
under §11 `## Owed`. Silence is not an exception.

---

## 8. Risks

**R1 — the `lm_head` ordering, and it is the one thing to check first.** §3.5.
A2-P cannot return device-resident logits while `lm_head` is the host NVFP4 arm
(`nemotron_h.cpp:1031-1034`), so whether the allowlist entry is *removed* or
*narrowed* depends on A2-Q2's state. `row/MODEL-NEMOTRON-H-ABI-A2Q2A` was local
and unpushed at `4b71c16af` when this spec was written. **Depend on the issue and
on `origin/main`, never on a branch name a `git ls-remote` cannot reach.**
Re-verify at claim time; do not implement against a described dependency.

**R2 — a stale in-tree anchor becomes a wrong port.** Three are already measured
wrong: `nemotron_h_device.cpp:56` cites `dense_attn_block.h:496` (actual `:497`),
`nemotron_h_registry.cpp:141` cites `nemotron_h.cpp:585-630` (actual `:631`), and
the governing spec cites `nemotron_h.cpp:822-825` for the CPU-queue check (actual
`:868-871`, with a **second** one at `:1031-1034` that A2-P must preserve until
`lm_head` moves). Fix the ones A2-P is already editing; verify every anchor you
cite rather than copying it forward.

**R3 — a governing-spec blocker that is no longer real.** §4.4 is the measured
instance: the bf16 conv-state kernel arm landed `908bad0ac` on 2026-08-09, six
days before the spec that called it unplanned work. Re-measure a stated blocker
at your own base before planning around it.

**R4 — the fresh-request zeroing is a silent-wrong-answer path.** §3.3. The
kernels read the state buffer unconditionally; a stale mamba block produces
fluent, plausible, wrong output and no error. P-M4 is the instrument, and if it
survives, the coverage hole is a finding rather than a pass.

**R5 — the fixture must be the checkpoint the changed path loads.** Pin revision
`29f2d1746d8f41e316523194b19018707749b1b1`, leave `VT_NEMOTRON35_SNAPSHOT` UNSET
so the revision check binds (`tests/parity/test_hf_snapshot_pinning.cpp:62`), and
record the resolved directory. A repo silently re-quantized under an unchanged
name has cost this project a full campaign.

**R6 — a near-tie read as a defect.** If the goldens do not reproduce and the
divergence looks marginal, **ask the oracle's own top-2 margin FIRST** before
declaring a defect. A previous "divergence" on this project was a bit-exact
oracle tie.

**R7 — `origin/main` moves under a shared checkout.** Merge an immutable SHA and
**re-run the full gate after merging rather than reading the diff** — a clean
merge is not a merge that builds the behaviour either side had. #818 is exactly
that failure, in this model's own tests. Never force-push, including
`--force-with-lease`.

### 8.1 Stop conditions

- The narrowed interlock cannot be expressed — the forward cannot distinguish a
  batched step from a single-request one → **`NEEDS_DECISION`**; do not land.
- Reading the recurrent state from `GdnStateCache` changes any Qwen3.5 or
  Kimi-Linear allocated byte → **stop.** A2-P adds a consumer; it does not
  reshape a shared cache.
- The end-to-end gate cannot run on `dgx.casa` for a reason other than
  contention → **`NEEDS_DECISION`** stating the reason. Do not land a
  unit-gate-only result as if it were the A3 gate.
- P-M4 or P-M6 stays green → a coverage hole, recorded as a finding, with the
  direct assertion the spec then owes. Not a pass.
- A `quantized_layers` entry names an algorithm we do not implement → refuse by
  name, record as owed, never silently dequantize.
- Any temptation to widen the conv page to f32 to satisfy a precondition →
  **stop.** §4.4.
- Any temptation to add a NemotronH branch to `hf_config.cpp`, or to re-derive a
  per-layer signal from `config_` → **stop.** That is #810 itself, one layer up.

---

## 9. Records owed on landing

A2-P changes lifecycle state, so it owes `docs/STATUS.md`, `docs/BENCHMARKS.md`
(pending, failed or void is a result — silence is not; the Thor leg is recorded
**PENDING #962** if it touches the Marlin arms), this spec's `## Now`, and the
row + checklist entry + rollup in `.agents/model-matrix.md:285` in the **same**
change (`scripts/check-model-checklist.py` enforces the rollup).

Plus:

- `docs/FEATURES.md:144`, whose `NemotronHForCausalLM` row currently reads
  "config+enumeration+KV-shape gated; hybrid … forward COMPUTES";
- `scripts/runner-routing-allowlist.txt:26` — **removed** if §3.5's condition is
  met, **narrowed** otherwise, never left stale;
- `docs/USAGE.md` — AGENTS.md §"Say which weights, and from where" applies the
  moment the capability is reachable: file name, size, exact HuggingFace repo
  **and revision** `29f2d174…`, grouped by arm, with the refused arms (GGUF, MTP,
  `NemotronHPuzzleForCausalLM`) named beside them. Run
  `scripts/check-doc-checkpoint.py` rather than assuming either way; note
  `kimi-linear-gen` is not currently documented there, so the checker's answer is
  the one that binds, not the precedent;
- the governing spec [`nemotron-h-abi-e2e.md`](nemotron-h-abi-e2e.md) — its §2
  seam claim and its §2.7 R4 are both corrected by §2.3 and §4.4 here; the
  pointer edit rides in A2-P's PR, which is what
  [#941](https://github.com/mudler/vllm.cpp/issues/941) item 1 asks for.

`.agents/NOW.md` is authored at operator cadence and is **not** a per-row
lifecycle write.

---

## 10. Now

**State at this commit:** spec only. No product code, no lifecycle change. Per
the governing spec's §1.4 the implementation is a separate PR by a different
agent and the reviewer is a third.

**A2-P is claimable now.** Its one prior blocker is cleared and re-verified
rather than inherited: the governing spec's R3 named #496 W2 (the CUDA Mamba2 SSD
arm) as A2's hard dependency, and it landed at `43a6c5518` —
`kMamba2ChunkScan`, `kMamba2StateUpdate`, `kCausalConv1dFwd` and
`kRmsNormGatedGroup` are all registered from `src/vt/cuda/cuda_gdn.cu`, an
unconditionally compiled CUDA source (`CMakeLists.txt:1589`), so they resolve on
sm_121a and sm_110 alike.

**Next action:** a fresh implementer claims this file, captures the §5.1 red
first — `ModelRegistry::Forward` on a NemotronH engine, refusing at
`nemotron_h_registry.cpp:161` — and lands the paged forward with G-SAFE
**narrowed to `num_reqs <= 1`**, not deleted. A fresh reviewer, never the
implementer, runs the §5.6 mutations and reports P-M5 and P-M6 as pairs.

**Four things to read before the first edit**, because each has already cost
somebody a cycle: §3.5, so the allowlist entry is removed or narrowed on
evidence rather than deleted to satisfy a checker; §4.4, so the conv page is not
widened to f32 against a precondition that no longer exists; §3.3's zeroing
obligation, which is the silent-wrong-answer path in this unit; and §2.3, so
nobody routes this architecture through a block that ropes.

## 11. Owed

- [#941](https://github.com/mudler/vllm.cpp/issues/941) — item 1 (correct the
  governing spec's seam claim) is answered by §2.3 of this file. Item 2 (a
  model-local block) was already answered by A2-R and is extended here. **Item 3
  — whether `hf_config.cpp:551` defaulting `rms_norm_eps` to `0.0` rather than
  refusing is right in general — is NOT this unit's**, is tree-wide, and stays
  open on #941.
- [#962](https://github.com/mudler/vllm.cpp/issues/962) — NVFP4 Marlin disagrees
  with itself on sm_110. A2-P's Thor leg is PENDING on it wherever the gate
  reaches the NVFP4 arms (§5.7); the fix is not this unit's.
- [#1052](https://github.com/mudler/vllm.cpp/issues/1052),
  [#294](https://github.com/mudler/vllm.cpp/issues/294),
  [#618](https://github.com/mudler/vllm.cpp/issues/618),
  [#584](https://github.com/mudler/vllm.cpp/issues/584) — the inherited-red
  baseline of §5.8. Subtracted, not chased, and re-measured at the implementer's
  own base.
- Batching (`num_reqs > 1`) stays refused by the narrowed G-SAFE clause and is
  owed to **A2-B**. Tracked on
  [#810](https://github.com/mudler/vllm.cpp/issues/810).

## 12. Outcome

Not yet written. Per AGENTS.md this section is filled when the unit reaches
`DONE`: what was measured, what was rejected and why, and why each default is set
the way it is — including, explicitly, the §4.4 conv-page dtype result with its
cost in bytes per token if it diverges from upstream, and the §3.5 allowlist
disposition with the evidence that decided it.
