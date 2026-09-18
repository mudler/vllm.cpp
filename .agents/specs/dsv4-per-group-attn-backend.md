# DSv4 per-group attention-backend dispatch

Row: MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm
Issue: ISSUE-LOCAL-01M2EMPC6T63TVDPQ90GVPRC5F
Upstream pin: vLLM `e126687a9a` (`.agents/upstream-sync.md`)
State: DONE

## Now

DONE 2026-09-18. The per-group dispatch is ported and the production-shaped
fixture gets past engine construction with every group answered by the backend
that serves it. The fixture is SYNTHETIC: only a leased run against the real
82,438,622,112-byte artifact can show the checkpoint itself gets further, and no
such run was taken. The next wall on that path is the W7-device forward refusal
at `deepseek_v4.cpp:4658`, owned by
`.agents/specs/deepseek-v4-device-decode.md`.

## Scope

One change: the engine asks THE BACKEND THAT SERVES A KV-CACHE GROUP whether
that group's block size is legal, instead of asking one backend resolved for the
whole model.

In scope:

- Three attention-backend descriptors that upstream defines and this tree does
  not: `CompressorBackend`, `DEEPSEEK_SPARSE_SWA`, `DEEPSEEK_V4_INDEXER`.
- A per-group backend name on `KVCacheGroupSpec`, which is upstream's
  `AttentionGroupKey.attn_backend`.
- `MakeDeepseekV4KVCache` naming the backend for each group it publishes,
  exactly as each upstream layer's `get_attn_backend()` names it.
- `GPUModelRunner::initialize_kv_cache` honouring that name.

Out of scope, and deliberately unchanged:

- The `% 16` refusal in `FlashAttentionBackend`, `RocmAttentionBackend` and
  `TritonMLABackend`. That refusal is CORRECT for those backends. The defect is
  that they are consulted about a group they do not serve. Widening or deleting
  the check would let a group land on a backend that cannot page it.
- Any attention KERNEL. These three are host metadata, which is all upstream's
  `CompressorBackend` contributes to this decision path too.
- The hash-granularity port (`BlockHashListWithBlockSize`), still owed by
  `.agents/specs/deepseek-v4-flash-vision.md`.
- The W7-device forward refusal at `deepseek_v4.cpp:4658`, which is the NEXT
  wall and belongs to `.agents/specs/deepseek-v4-device-decode.md`.

## The defect, in one paragraph

`MakeDeepseekV4KVCache` publishes seven groups at block sizes
`{256,256,256,64,4,4,8}` (`deepseek_v4_registry.cpp:546-556`). The runner's view
loop resolves ONE backend name per class and CACHES it
(`runner.cpp:1691-1710`, `mla_backend_resolved` / `dense_backend_resolved`), so
the 256-token latent group resolves `TRITON_MLA` and the 4/4/8 compressor and
indexer groups then reuse that name. `CheckKvCacheShape`
(`registry.cpp:160`) calls `TritonMLABackend::get_kv_cache_shape`, whose
`% 16` guard (`backend.cpp:279`) throws. The cache is not the caching itself: a
per-group re-resolution through the SELECTOR would find no candidate for a
4-token MLA group and fall through to op-driven silence. The missing piece is
the backend that legitimately serves those groups.

## Upstream anchors, all read at `e126687a9a`

| Fact | Anchor |
|---|---|
| No global 16-multiple rule; the default is `[MultipleOf(1)]` | `vllm/v1/attention/backend.py:72-74`, accessor `:116-133` |
| The 16-multiple is a PER-BACKEND override, in three backends | `triton_attn.py:314`, `triton_mla.py:152`, `rocm_aiter_unified_attn.py:53` |
| The dispatch key is the LAYER, not the model | `gpu_model_runner.py:7150` (`layers[layer_name].get_attn_backend()`), keyed into `AttentionGroupKey` at `:7170` |
| `CompressorBackend` declares `[MultipleOf(1)]` | `vllm/models/deepseek_v4/compressor.py:66-68` |
| `CompressorBackend` head sizes `[512, 1024]` | `compressor.py:70-72` |
| The compressor layer NAMES that backend | `compressor.py:189-190` (`get_attn_backend`) |
| Block size 4 for ratio 4, 8 for ratio 128, and WHY | `compressor.py:152-167` |
| Compressor publishes a real `SlidingWindowMLASpec` | `compressor.py:173-185` |
| `DeepseekSparseSWABackend` declares `[MultipleOf(64)]` | `vllm/v1/attention/backends/mla/sparse_swa.py:126-128` |
| ...head sizes `[512]`, preferred block 256 | `sparse_swa.py:130-136` |
| The SWA layer names it | `sparse_swa.py:116-118` |
| `DeepseekV4IndexerBackend` declares `[256]` | `vllm/v1/attention/backends/mla/indexer.py:194-196` |

Three escapes were checked upstream and are NOT available, so none is copied:
the DSA cache IS a KV-cache group (`compressor.py:133`, `:173`);
`unify_kv_cache_spec_page_size` only ever RAISES a block size
(`kv_cache_utils.py:1158-1162`); `kernel_block_size` only ever selects a DIVISOR
(`worker/utils.py:371-375`). Evidence in
ISSUE-LOCAL-01M2EMPC6T63TVDPQ90GVPRC5F.

## Design

1. `KVCacheGroupSpec` gains `std::string attn_backend`, default empty. Empty
   means "resolve through the platform selector", which is what every existing
   topology does today, so every existing topology is byte-identical.
2. Three backend descriptors, upstream names and upstream declared lists. All
   three are MLA-shaped: the page is ONE vector per token, which is why the
   factory already gives their groups an `MLAAttentionSpec` /
   `SlidingWindowMLASpec` and why `CheckKvCacheShape` expects the fused rank-3
   view for them.
3. They self-register per `DeviceType` like every other backend, but they are in
   NO platform priority list. That is upstream's shape exactly: a layer names
   them, the capability walk never reaches them.
4. `MakeDeepseekV4KVCache` names them per group, mirroring each upstream layer.
   The two latent `MLAAttentionSpec` groups keep the empty name, because
   upstream's MLA attention layer resolves through the ordinary selector.
5. The runner uses a named backend DIRECTLY (no selector walk, no caching) and
   keeps the existing lazy dense/MLA resolution for every unnamed group.

## Divergences from upstream, and why

- Our backends are INSTANCES behind a factory, not classes. That adaptation
  predates this row (`registry.h`) and is unchanged here.
- Upstream keys the backend per LAYER inside a group; we key it per GROUP. Every
  group this factory publishes is homogeneous — one layer class per group — so
  the two are the same map for this model. A model that needed two backends
  inside one group would need the finer key, and nothing here blocks that.
- `get_preferred_block_size` (`sparse_swa.py:130-132`) is NOT ported: the engine
  block size is already resolved to 256 by this tree's own path before the
  factory runs, so the hook has no caller here. Recorded under `## Owed`.

## Tests

Ported into `tests/vllm/entrypoints/test_deepseek_v4_multigroup_kv.cpp`, which
already drives `LoadedEngine::FromModelDir` on DEFAULT `EngineParams` — the
loader entry every server and command line takes for a `.gguf`.

1. RED-FIRST, production entry point: the child engine prints the per-group
   backend names the runner RESOLVED (`GPUModelRunner::attn_backend_names()`,
   reached through the public `LoadedEngine::runner()`), and the parent asserts
   the upstream name for every published cache. Before the change the list is
   nine empty strings on CPU and `TRITON_MLA` for every entry on CUDA; both are
   the same defect, and both fail this assertion.
2. The guard is INTACT: `TRITON_MLA` still refuses block size 4 with the
   verbatim `Block size must be a multiple of 16.`, and `FLASH_ATTN` still
   refuses it too. A change that made the model load by widening either of those
   fails this case.
3. Each new backend accepts EXACTLY what upstream declares and refuses the rest:
   `CompressorBackend` accepts 4 and 8, `DEEPSEEK_SPARSE_SWA` accepts 64 and 256
   and refuses 4, `DEEPSEEK_V4_INDEXER` accepts 256 and refuses 64.
4. The existing cases (topology, named-refusal, explicit `--kv-cache-dtype`) keep
   passing unchanged, and the suite keeps answering identically with and without
   `NDEBUG` — the child-process shape is untouched.

## Risks

- A backend name that no device registers would turn a served group into a hard
  `MakeAttentionBackend` throw at engine construction. Mitigated by registering
  all three for every `DeviceType`, which is honest because not one line of
  their metadata is device-specific.
- `attn_backend_names_` is recorded and NOT dispatched on
  (`runner.cpp:1663-1667`, owed to #1332 M4). This change makes the name CORRECT;
  it does not make it a route. Stated so a green selection is not read as a
  working kernel.

## Gates

- `test_deepseek_v4_multigroup_kv` green, Release (NDEBUG) and default configure.
- `test_attn_backend_registry`, `test_attn_validate_configuration`,
  `test_runner`, `test_kv_cache_interface` green.
- `scripts/agent-preflight.sh --staged`.

## Owed

- `get_preferred_block_size` (`sparse_swa.py:130-132`) has no caller in this
  tree; ported when the block-size resolution grows the hook.
- The three backends carry NO `get_impl_cls()`. The DSA forward is the model's
  own op path (`deepseek_v4.cpp`), so nothing dispatches through them yet; that
  is #1332 M4's seam and is not widened here.
- `BlockHashListWithBlockSize`, still owed by
  `.agents/specs/deepseek-v4-flash-vision.md`.

## Stop conditions

- Stop and report `NEEDS_DECISION` if making a group load requires weakening the
  `% 16` refusal in any of the three backends that declare it.
- Stop if the synthetic fixture stops reproducing the seven-group topology.

## Outcome

WHICH OF THE THREE REFUSAL SITES FIRED, measured rather than guessed. Every
group this factory publishes is fused, so `runner.cpp:1412` marks all nine
caches MLA and the view loop takes the `is_mla` arm for every one. That arm
cached its resolution, the 256-token latent group resolved `TRITON_MLA`, and the
4/4/8 groups inherited it: the site is `backend.cpp:279`. The FlashAttention and
ROCm copies of the same string never ran, because no group reached the dense arm.

WHY THE SUITE WAS GREEN ON A DEFECT THAT STOPS EVERY CUDA LOAD. On a CPU box the
same arm resolves nothing — no MLA backend is registered for `kCPU` — the name is
empty, and `CheckKvCacheShape` is skipped outright. At base the child engine
reported nine caches and not one backend consulted:
`BACKENDS=-:256;-:256;-:256;-:64;-:64;-:64;-:4;-:4;-:8`. An engine that consults
NO backend and an engine that consults the WRONG one are the same defect, and
the new case fails on both, which is why it could be written red here and is
meaningful on the box where the checkpoint actually dies.

WHAT WAS REJECTED.

- **Widening or deleting the `% 16` rule.** It is correct for the three backends
  that declare it, and upstream keeps it in exactly those three
  (`triton_attn.py:314`, `triton_mla.py:152`,
  `rocm_aiter_unified_attn.py:53`). Removing it would let a 4-token group land on
  a backend that cannot page it, and the fixture would go green on that too. A
  case now asserts the verbatim refusal survives.
- **Re-resolving per group through the SELECTOR instead of caching.** This looks
  like the smaller change and it is wrong: no registered backend accepts a
  4-token MLA group, so the walk throws, the runner's `catch` swallows it, and
  the group silently becomes op-driven. That converts a loud wall into a quiet
  one. The missing piece was a backend that legitimately serves those groups.
- **A spec-kind predicate instead of a named backend.** The compressor states
  and the sparse-SWA cache publish the SAME spec class
  (`SlidingWindowMLASpec`) upstream and here, so spec kind cannot tell them
  apart. Upstream's key is the LAYER (`gpu_model_runner.py:7150`), and a name on
  the group is the nearest expressible form of that key in this tree.

WHY EACH DEFAULT HAS ITS VALUE.

- `KVCacheGroupSpec::attn_backend` defaults to EMPTY, meaning "resolve through
  the platform selector". Empty is what every other model publishes, so the
  runner enters its pre-existing lazy resolution exactly as often as before and
  no other topology moves.
- The two latent `MLAAttentionSpec` groups are deliberately left unnamed:
  upstream's MLA attention layer resolves through the ordinary selector too, so
  naming them would diverge and would also hard-code `TRITON_MLA` onto devices
  that have no MLA backend.
- The three new backends register for EVERY `DeviceType` and appear in NO
  platform priority list. Both halves are upstream's shape: nothing in their
  metadata is device-specific, and a layer names them rather than a capability
  walk selecting them. A missing registration would turn a named group into a
  hard throw at a customer's engine construction, which is the failure this row
  exists to remove.
- `DEEPSEEK_V4_INDEXER` declares `{256}` and not `{1}`. `supports_block_size`
  reads every entry as a MultipleOf, so `{256}` refuses 64 — which is the
  nearest expressible form of upstream's exact `[256]` and is what makes the
  indexer's answer different from the SWA cache's.
