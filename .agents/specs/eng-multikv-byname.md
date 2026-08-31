# ENG-MULTIKV-BYNAME — the by-name KV channel addresses RECURRENT caches, not only attention ones

Issue: **OWED, not filed.** GitHub writes answer `403` (account suspended) from
this host, so no issue could be opened and no `.agents/issue-index.md` row could
name one. The work is owned by `KV-DSV4-MULTICACHE`
([#1925](https://github.com/mudler/vllm.cpp/issues/1925),
[#2068](https://github.com/mudler/vllm.cpp/issues/2068)), whose spec
[`kv-dsv4-multicache.md`](kv-dsv4-multicache.md) carries the matching `## Owed`
entry; the defect this fixes was MEASURED under
[#2343](https://github.com/mudler/vllm.cpp/issues/2343).

Row: `ENG-MULTIKV-BYNAME`. No `.agents/engine-matrix.md` row was added, because a
row cannot be opened without an issue to name and adding one would owe four
record edits this session cannot complete. That is recorded under `## Owed`
rather than left implicit.

**Ordering disclosure.** AGENTS.md says "Commit the spec before implementation.
Never write the spec after the implementation." This document was AUTHORED after
the implementation was written, and committed before it. The commit order
satisfies the checkable half of that rule and NOT its intent, and saying so here
is the only thing that stops the git history from implying otherwise.

## Now

`DONE` for the channel; the consuming forward is still `KV-DSV4-MULTICACHE` W5's.

## Scope

In: making `vllm::MultiKvCacheIndex` able to address every published cache by
layer name, recurrent members included, and making
`GPUModelRunner::initialize_kv_cache` populate it that way.

Out: lifting the `multi_kv` refusal in `ModelRegistry::Forward`. That refusal
must land WITH its first consuming forward — none of the three arriving
architectures consumes the channel yet, so removing it early trades a refusal for
a silent full-prefix recompute. This change repairs the refusal's WORDING, which
its own counts made wrong, and leaves the guard standing.

Also out: the block-table geometry (#2085), the `KVBytesPerBlock` layer-count
budget, speculation on a multi-cache topology, and the `fa_draft`-on-`spec_on()`
seam. Each is already owned by `kv-dsv4-multicache.md` `## Owed`.

## The gap

`MultiKvCacheIndex` carried three vectors documented as PARALLEL to
`ModelForwardInput::attn_kv`. The runner filled them from `attn_group_ids_`,
which collects only groups whose spec is an `AttentionSpec`. A `MambaSpec`
group's layers therefore contributed no entry at all: their states reached the
forward through `gdn_state` POSITIONALLY, and `Find()` answered -1 for every one
of them.

DeepSeek-V4 hid this. It publishes only MLA / SlidingWindowMLA specs, so all 167
of its caches are attention ones and the channel was already complete for the
seven-group shape. Both three-group hybrids — `qwen4_exp` and `glm5_next` — carry
a `MambaSpec`, and #2343 measured the consequence on GLM-5.3-Flash: the forward
refusal read `22 KV cache(s) from 2 published group(s)` beside `block tables
gathered for 3 of 3 published group(s)`. 34 recurrent states invisible while
their group's block table was not.

## Upstream anchors

Read at pin `5559679229bc961848b121ccdeaa8fa5d79bec98` under
`/home/mudler/_git/vllm`, verified with `git -C /home/mudler/_git/vllm log -1`.
Every line range below was re-derived by reading the file, not copied from a
sibling row's citation.

| what it says | where |
|---|---|
| ONE dict holds every cache: `kv_caches: dict[str, torch.Tensor] = {}` | `vllm/v1/worker/gpu_model_runner.py:7354` |
| its insertion order is GROUP order then the group's own `layer_names` order | `:7365-7372` |
| the attention arm fills it | `:7418-7427` |
| the `MambaSpec` arm fills THE SAME dict, "Keeping one tensor per layer lets the KV connector register it without special-casing Mamba" | `:7429-7441` |
| the by-name map must cover EVERY layer of EVERY published group | `:7318-7326` |
| one dict in, every layer bound out of it | `vllm/v1/worker/utils.py:450-465` (`bind_kv_cache`) |
| group id is the BUILD key, layer name is the LOOKUP key | `:2318-2334` (`_get_block_table`), `:2554-2567` (`for kv_cache_gid, kv_cache_group in enumerate(...)`) |

**Upstream has no attention/recurrent split in this channel**, and no "named
group ids" concept. This change mirrors that: one list, one order, one `Find`.

## Design

`MultiKvCacheIndex`'s vectors stop being parallel to `attn_kv` and become
parallel to EACH OTHER, covering every published cache in upstream's insertion
order. Two vectors are added, `payload_kinds` and `payload_slots`, carrying the
one fact a name cannot: which of this tree's two typed payload containers holds
the cache, and at which slot. `enum class KvCachePayload { kPaged, kRecurrent }`
records that distinction and nothing else — it is a C++ type tax, not an
addressing split, because upstream's payloads are all `torch.Tensor` and ours
are `PagedKvCache` and `GdnStateCache`.

`Find(name)` now returns a FLAT index over the published caches. On a topology
with no recurrent group the flat index still equals the `attn_kv` slot, which is
why DeepSeek-V4's existing assertions are untouched; on a hybrid it does not, and
that non-identity is what the new gate asserts.

Two runner-side guards make drift a refusal rather than a silent hole: the paged
half of the index must equal `attn_kv_layer_names_` entry for entry, and the
index length must equal `attn_kv_.size() + recurrent_state_buf_.size()`.

`attn_kv_group_ids_` and `attn_kv_layer_indices_` are DELETED. They existed only
to back the channel's two pointers; the flat vectors back them now, and keeping
them would be two derivations of one order with nothing comparing them.

## Gates

- `./build/tests/test_runner` — the focused gate. 36/36, 1844/1844.
- Uniform-topology neutrality: `test_qwen35_paged_forward` 7/7,
  `test_nemotron_h_paged_forward` 13/13, `test_kimi_linear_paged` 8/8,
  `test_llm_engine` 24/24, plus the in-file `runner: W3 is BYTE-NEUTRAL for
  every topology shipped today`.
- All-attention neutrality: `runner: DeepSeek-V4's real 167-entry topology
  allocates end to end` now asserts all 167 flat indices equal their `attn_kv`
  slots.
- `scripts/agent-preflight.sh --fail-on-skip` — 0 SKIP, 1 FAIL
  (`role-undeclared`, the operator's record).

## Owed

- **An issue, and an index row naming it.** GitHub writes are `403` from this
  host, so neither could be created. Owned by `KV-DSV4-MULTICACHE`; falls due at
  the next session with forge write access.
- **The engine-matrix row.** Not added, for the same reason: a row owes an issue
  it can name.
- **The SACRED `test_qwen35_paged_engine` regression was NOT run.** It exits 77
  — "GATE NOT RUN — SKIPPED, this is NOT a pass" — because the pinned
  `Qwen--Qwen3.5-0.8B` snapshot `2fc06364` is not cached on this host. PENDING on
  a named resource, not satisfied. Owned by this row.
- **No consuming forward reads the recurrent half yet.** `ModelRegistry::Forward`
  still refuses a multi-cache index; it now reports the recurrent count instead of
  omitting it. Owned by `KV-DSV4-MULTICACHE` W5.
- **`num_groups()` counts groups that CONTRIBUTED an entry**, which is now every
  published group on any topology this runner accepts, so it agrees with
  `num_published_groups()`. It is kept as a separate accessor rather than
  collapsed, because a future shape that publishes a group with no addressable
  member would make them differ again and that difference is the diagnostic.

## Evidence

| what | where | read |
|---|---|---|
| our tree | `/home/mudler/_git/vllm.cpp`, base `7d53ae3b405761202b13065ee32ea1d0dd13cd29` | 2026-08-30 |
| pinned vLLM | `/home/mudler/_git/vllm` at `5559679229bc961848b121ccdeaa8fa5d79bec98` | 2026-08-30 |
