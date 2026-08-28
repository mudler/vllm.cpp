# ENG-RECURRENT-MULTISTATE — a recurrent layer carries N states, because upstream's does

Issue: [#2131](https://github.com/mudler/vllm.cpp/issues/2131).
Row: `ENG-RECURRENT-MULTISTATE`.
Kind: ENGINE. This is shared machinery every recurrent model in the tree runs
through, not a model port.

**The index row for #2131 already exists**, appended when the issue was filed and
keyed to `MODEL-MM-QWEN4-EXP`. `.agents/issue-index.md` is append-only under
`merge=union`, and a second row for the same issue number is refused as exactly
the duplicate two branches appending one issue would produce — verified by
running `scripts/check-agent-record.py` against an appended row and reading its
refusal. So this row appends nothing there, and this document is where the issue
is linked from the work.

vLLM registers `MambaSpec`, `MambaBase` and the recurrent half of the GPU runner
at the parity pin, so vLLM is the mirror source for every decision here and no
secondary oracle is admissible. Pin: `5559679229bc961848b121ccdeaa8fa5d79bec98`,
verified with `git -C /home/mudler/_git/vllm log -1` on 2026-08-28.

## Now

`ACTIVE`. This document is written before the code it scopes, and the wave it
scopes is the FIRST of at least two: it makes a recurrent group carry N states,
and it leaves per-layer state heterogeneity inside one group to a later wave.
`## Owed` names both, with the upstream anchor for each.

## Scope

`GPUModelRunner::initialize_kv_cache` refuses any recurrent group whose
`MambaSpec` does not carry EXACTLY two shapes and two dtypes:

```cpp
VT_CHECK(mamba_spec->shapes.size() == 2 && mamba_spec->dtypes.size() == 2,
         "runner: recurrent MambaSpec must contain conv then temporal state");
conv_state_shape = mamba_spec->shapes[0];
ssm_state_shape  = mamba_spec->shapes[1];
```

and `GdnStateCache` carries exactly two named tensors, `conv_state` and
`ssm_state`. Between them, a recurrent layer in this tree cannot hold a third
state at all.

In scope: the state COUNT and the per-state dtype, end to end — spec read,
allocation, byte accounting, and the view carrier the models read.

Out of scope, each named under `## Owed` rather than dropped: per-layer state
heterogeneity within one recurrent group, a state count of ONE, a SECOND
recurrent group, and any model that publishes N >= 3.

## The finding: vLLM never had a two-state assumption, and we invented one

The issue's premise is that upstream may not express this either. It does, it
expresses it fully generally, and it SHIPS three different values of N at the
pin. Read at `5559679229`:

| Upstream | Anchor | What it says |
|---|---|---|
| the carrier | `vllm/model_executor/layers/mamba/abstract.py:26` | `kv_cache: tuple[torch.Tensor, ...]` — an ordered tuple of unbounded length, NOT a named `(conv, ssm)` pair |
| the unpack | `abstract.py:29-43` `bind_kv_cache` | `for shape, dtype in zip(self.get_state_shape(), self.get_state_dtype())`, slicing one page at a running byte offset. N states, each with its OWN shape and its OWN dtype |
| the contract | `abstract.py:46-52` | "For mamba layers this is **usually** a (conv_state, ssm_state) tuple". Two is a convention the docstring itself hedges |
| N == 1 | `vllm/model_executor/layers/mamba/short_conv.py:87` | `self.kv_cache = (torch.tensor([]),)` |
| N == 5 | `vllm/model_executor/layers/mamba/mamba_mixer2.py:517-520` | `_n_state = 5 if self.use_replayssm else 2`, and `:722-724` `x_cache, dt_cache, B_cache = self.kv_cache[2:]` |
| N == 5 shapes | `vllm/model_executor/layers/mamba/mamba_utils.py:202-221` | the three appended shapes are rank 3, rank **2** and rank 3 — a rank change inside one layer's state set |
| N == 5 dtypes | `mamba_utils.py:84-93` | `(*base_dtypes, activation_dtype, torch.float32, activation_dtype)` — a `float32` beside two activation dtypes |
| the runner | `vllm/v1/worker/gpu_model_runner.py:7429-7440` | allocates `num_blocks * page_size_bytes` RAW int8 and hands the layer one untyped page. The runner never learns N, and cannot |
| the spec | `vllm/v1/kv_cache_interface.py:698-707` | `page_size_bytes` is `sum(prod(shape) * get_dtype_size(dtype))` over the zip — already N-general |

Our `MambaSpec` (`include/vllm/v1/kv_cache_interface.h`) already mirrors the last
row: it holds `std::vector<std::vector<int64_t>> shapes` and
`std::vector<vt::DType> dtypes`, and `MambaSpec::page_size_bytes` sums over both.
`vllm::v1::recurrent_state_bytes` reads nothing but `page_size_bytes()`. **The
two-shape assumption exists in exactly two places, the runner and the state
carrier, and both are local inventions.** That is why this is a repair and not a
feature.

## Design

Mirror `bind_kv_cache`. The recurrent cache becomes an ORDERED LIST of states
whose length, per-state shape and per-state dtype all come from the group's own
`MambaSpec`.

1. **`GdnStateCache` grows `std::vector<vt::Tensor> states`** — the mirror of
   `kv_cache: tuple[torch.Tensor, ...]`. `conv_state` and `ssm_state` stay, and
   are `states[0]` and `states[1]`. Every existing consumer — `qwen3_5.cpp`,
   `kimi_linear_device.cpp`, `nemotron_h_device.cpp`, `gemma4_mm.cpp` — reads
   those two names and is untouched.
2. **The runner's recurrent geometry becomes vectors over N.** One
   `CacheBuffer` per (recurrent layer, state), allocated in SPEC ORDER, which is
   the order `bind_kv_cache` slices in. `kv_cache_allocated_bytes` sums every
   one of them, so the memory the runner reports stays the memory it took.
3. **The refusal widens from `== 2` to `>= 2`, and gains
   `shapes.size() == dtypes.size()`.** The second half was never checked: a spec
   with two shapes and one dtype read `dtypes[1]` out of bounds.
4. **The per-state dtype predicate widens from `{F16, BF16, F32}` to any
   non-block-quantized `vt::DType`.** `bind_kv_cache` imposes no dtype
   constraint at all; the local floating-only rule was justified by "all-zero
   bytes are `+0.0f` for every supported floating storage type", which is
   equally true of an integer zero. The real constraint is that a block-quant
   dtype has no per-element size, and that is what the widened predicate names.
   This is what makes an INTEGER state expressible — a `qwen4_exp` PLE layer's
   n-gram history holds `input_ids.long()`, i.e. token ids and not activations.

Widening an assertion is a semantic checker change, so it lands red-first: the
new test is RED at the base tree for BOTH halves (the count and the dtype), and
the widening is justified by the upstream anchor rather than by making a gate
green.

### What this wave deliberately does NOT do

`gdn_group_id_` stays a scalar and the `recurrent_seen > 1` refusal stays. The
issue reads the one-group limit as a second half of the same blocker. It is a
real hole, but it is NOT the one a PLE topology hits, and that correction is
worth recording: upstream does not split recurrent layers with different state
sets into different groups. `vllm/v1/core/kv_cache_utils.py:1099-1110` pads the
smaller `MambaSpec` page up to `max_page_size` through `page_size_padded` and
keeps ONE group, precisely because "MambaSpec's page size is determined by its
state shapes and does not scale with block_size". Heterogeneous recurrent layers
are therefore a PER-LAYER SPEC problem inside one group — the recurrent twin of
the `per_layer_attn_specs` seam this tree already has for Gemma-4's
heterogeneous attention head_dim — and not a multi-group problem. That is the
next wave, and it is named under `## Owed`.

## Risks

- **Silent byte drift on the existing arms.** Four model families flow through
  these lines. Mitigated by an existing literal byte-neutrality case
  (`test_runner.cpp`, "the Qwen3.5 allocation is BYTE-IDENTICAL after #810") and
  by running the recurrent suites before and after and comparing case and
  assertion counts exactly.
- **A cosmetic generalization.** A vector that is only ever length 2 proves
  nothing. Mitigated by a mutation that reverts the loop to `states[0..1]` and
  must RED the new case, and by shapes chosen so the third state genuinely
  changes the allocated bytes, the view count and the reported total.
- **The reverse: the OLD path stops being exercised.** Mitigated by a mutation
  inside the two-state path that must RED an EXISTING recurrent suite.

## Tests

`tests/vllm/v1/worker/test_runner.cpp`:

- a THREE-state recurrent group, with a third state of a different rank, a
  different element count and a different dtype from either of the first two:
  three buffers, three views, `page_size_bytes` identity over all three, and
  `kv_cache_allocated_bytes` counting the third.
- an INTEGER third state (`kI64`), which is what a token-id history is.
- a spec whose `shapes` and `dtypes` disagree in length is REFUSED.
- a block-quantized state dtype is REFUSED.

## Gates

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DVLLM_CPP_BUILD_EXAMPLES=OFF
ninja -C build -j 6 test_runner test_qwen27_paged_forward test_nemotron_h_paged_forward test_kimi_linear_paged
./build/tests/test_runner
./build/tests/test_qwen27_paged_forward
./build/tests/test_nemotron_h_paged_forward
./build/tests/test_kimi_linear_paged
scripts/agent-preflight.sh --fail-on-skip
```

The three model suites are the regression gate named by the issue. Their case
and assertion counts are recorded in `## Outcome` before and after, because a
count that moved is the only thing that can see a case that stopped running.

## Stop conditions

Return `NEEDS_DECISION` rather than redesigning silently if the per-layer
heterogeneity turns out to be reachable only by changing `GdnStateCache`'s two
named fields, because that is a four-model blast radius and a different review.

## Outcome

Landed as the FIRST wave. The runner reads the state COUNT and every per-state
dtype off the group's own `MambaSpec`; nothing in the runner or the state carrier
names two any more.

### Regression counts, before and after, on the same tree

Read from the suite output, not from an exit code. The base is `8997c62b3`.

| Suite | Before | After |
|---|---|---|
| `test_runner` | 29 cases / 831 assertions / rc 0 | 31 cases / 884 assertions / rc 0 |
| `test_qwen27_paged_forward` | 31 / 770 / rc 0 | 31 / 770 / rc 0 |
| `test_nemotron_h_paged_forward` | 13 / 3269 / rc 0 | 13 / 3269 / rc 0 |
| `test_kimi_linear_paged` | 8 / 206 / rc 0 | 8 / 206 / rc 0 |

The three model suites are byte-identical in both numbers. `test_runner` moves by
exactly the two cases this row adds.

### RED, before the change

The three-state case threw the production refusal it was written against:

```
test_runner.cpp:823: ERROR: test case THREW exception:
  vt: runner: recurrent MambaSpec must contain conv then temporal state
  at src/vllm/v1/worker/gpu/runner.cpp:916
[doctest] test cases: 31 | 30 passed | 1 failed
```

### Mutation record

Each mutation was sha256-proven applied, its BUILD status was read before any
test result, and the tree was restored byte-for-byte and re-measured afterwards.
`runner.cpp` at the head is `c01eb6ee8d522d7cd7816b97584e87152d03b1fa14a0e33551f57dfad2644527`.

| # | Mutation | sha256 of `runner.cpp` | Build | Result |
|---|---|---|---|---|
| M1 | the widened refusal back to `shapes.size() == 2` | `f6c8d819…` | rc 0 | `test_runner` RED, 1 case, at the old message. The three model suites stay GREEN, so the mutation is scoped to the new arm |
| M2 | the VIEW loop reads `state_dtypes[i < 2 ? i : 0]` — the THIRD state's dtype mishandled, the first two untouched | `f4d7c05f…` | rc 0 | `test_runner` RED, 9 assertions, all on `states[2].dtype` and `states[2].Bytes()`. The three model suites stay GREEN. This is what makes the third state load-bearing rather than decorative: it is the only state whose answer moves |
| M3 | `gs.ssm_state = gs.states[0]` — the OLD two-state path, inside the same generalized loop | `ed8b76cf…` | rc 0 | `test_nemotron_h_paged_forward` RED (11 of 13 cases, 23 assertions), `test_kimi_linear_paged` RED (5 of 8, 11 assertions), `test_runner` rc 139. The existing arms genuinely run through the new loop |

M2 is the one that answers "is this cosmetic". The fixture's third state is a
different RANK (1-D against 2-D and 3-D), a different ELEMENT COUNT (7 against
192 and 256) and a different DTYPE (`kI64` against two `kF32`) from either of the
first two, so no implementation that reuses `shapes[0]`, `dtypes[0]` or a factor
of 2 can produce its bytes.

### A gate the issue named that does not gate this

[#2131](https://github.com/mudler/vllm.cpp/issues/2131) names
`test_qwen27_paged_forward` as the regression gate for this change. MEASURED: it
is not one. Under M3 — the runner handing every recurrent layer its conv state
where the temporal state belongs — that suite reads 31 cases / 770 assertions /
rc 0, unchanged, while `test_nemotron_h_paged_forward` and
`test_kimi_linear_paged` both go red and `test_runner` faults. The suite builds
its own `GdnStateCache` views rather than reading the runner's, so it cannot see
a defect in the runner's state assignment. The real regression gate for this seam
is those other three, and this row used all four.

## Owed

- **Per-layer recurrent specs inside one group.** A `qwen4_exp` PLE topology
  needs it: only ONE of its linear-attention layers carries the PLE conv and the
  n-gram history, and upstream serves that by padding the smaller page
  (`vllm/v1/core/kv_cache_utils.py:1099-1110`) while each layer keeps its own
  `get_state_shape()` / `get_state_dtype()` (`abstract.py:29-43`). The seam to
  mirror is the existing `KVCacheConfig::per_layer_attn_specs`. Owned by W5c of
  [#2031](https://github.com/mudler/vllm.cpp/issues/2031) or a successor of this
  row; tracked by [#2131](https://github.com/mudler/vllm.cpp/issues/2131).
- **A recurrent group of ONE state.** Upstream's `ShortConv`
  (`short_conv.py:87`) has no temporal state. `GdnStateCache::ssm_state` is a
  named field every consumer reads, so N == 1 needs those consumers to stop
  assuming it, which this wave does not touch. Refused with a message naming the
  missing part.
- **A SECOND recurrent group.** `recurrent_seen > 1` still refuses, and
  `gdn_group_id_` is still a scalar. Not needed for a PLE topology (see
  `### What this wave deliberately does NOT do`) but still unrepresentable.
- **`test_qwen27_paged_forward` does not gate the runner's recurrent state
  assignment**, measured above. Either it should enter through the runner's own
  `GdnStateCache`, or the issue text and any future dispatch should stop naming
  it as this seam's gate. Tracked by
  [#2131](https://github.com/mudler/vllm.cpp/issues/2131) until a row picks it
  up.
- **`GdnStateCache::states` is filled by the runner only.** The host-path
  scaffolds in `qwen3_5.cpp` and several test fixtures build the two named
  fields and leave the list empty. Inert while nothing outside the runner reads
  it; a consumer that starts reading `states` owes those builders the
  assignment.
- **Nothing publishes N >= 3.** Every recurrent registry in the tree publishes
  two states, so the N >= 3 arm lands EXPRESSIBLE and UNREACHED. The two-state
  arm is reached by every recurrent model through the same generalized loop —
  the special case is deleted rather than bypassed — so what is unreached is the
  VALUE of N, not the code. Owned by W5c of
  [#2031](https://github.com/mudler/vllm.cpp/issues/2031), tracked by
  [#2131](https://github.com/mudler/vllm.cpp/issues/2131).
