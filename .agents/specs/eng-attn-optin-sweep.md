# The opt-in attention sweep: every remaining `vt::Attention` caller, and what the seam should become

Issue: [#1552](https://github.com/mudler/vllm.cpp/issues/1552).
Owning row: KERNEL-ATTN-DENSE-FLASH (kernel-matrix.md), the row that already owns
`AttentionDenseFlash`, its head-dim contract and
[`attention-rung-visibility.md`](attention-rung-visibility.md). This spec is a
second increment on that row, for the same reasons that one gives: it is
deliberately not promoted into the row's `Spec` column, because
`check-gate-commands.py` classifies a row from the FIRST spec link there and the
row is pinned `no-gates-section`. No new matrix row is created, so no lifecycle
edit is owed.

Note on spelling: this document writes every row id WITHOUT backticks.
`check-agent-record.py::check_spec` selects a row's governing spec by searching
for the backticked token, and a second spec carrying it can change which file is
held to the structured-section contract.

## 1. What #1552 asks, and what was already true when this row started

#1552 owes three things. Two of them had substantially landed before this row
opened, under a different issue, and verifying that is the first obligation
AGENTS.md `## Spec before code` places on a claim.

[#1544](https://github.com/mudler/vllm.cpp/issues/1544) — the class issue, which
#1552 generalises from — already shipped `scripts/check-attention-rung-consistency.py`
and the `// VT-ATTN-NAIVE:` marker convention. Every `vt::Attention` call the
scanner can see must record, beside itself, why it stays on the naive kernel;
the refusal names all three fast rungs. Two routing rows then removed the two
genuine defects: `47a918d8f` (#1579, issue #1545) took Muse Glimmer's perception
encoder to `vt::AttentionDenseFlash`, and `90e8c3c85` (#1557, issue #1549) took
the LTX-2.5 DiT device forward to the same op, since raised to
`vt::AttentionDenseFa2` by #1551.

So on this row's base `f9af269f9`:

```
$ python3 scripts/check-attention-rung-consistency.py
OK (attention rung): 8 vt::Attention call site(s) in 8 model source file(s);
8 carry a recorded reason, 0 unmarked and excused by 0 allowlisted in-flight stem(s).
```

**Deliverable 2 is therefore already discharged for every site**, and this row
routes nothing. What was NOT recorded anywhere is deliverable 1's substance —
head_dim, the sequence length each site actually runs at, and dense/non-causal
eligibility. A marker records a REASON. It does not record a SHAPE, and a reason
cannot be re-checked against a shape that is not written down. §2 is that table.

Deliverable 3, the seam decision, was untouched. §4 is it, and it returns
NEEDS_DECISION as #1552 anticipates.

What is left as executable work is §3: the checker's population is not the tree,
and two holes in it are MEASURED rather than argued. "Sweep every REMAINING call
site" is a claim about a moment; a population that cannot see a whole directory
is what makes the next one silent.

### 1.1 The Muse Glimmer premise, checked rather than assumed

The dispatch that opened this row carried #1545's claim that Muse Glimmer's
vision tower "is believed to still be in that state". **It is not.**
`src/vllm/model_executor/models/muse_glimmer_vision.cpp:655` reads

```cpp
vt::AttentionDenseFlash(q, os, qs, ks, vs, aargs);
```

unconditionally, once per attention segment, with no env knob and no naive arm.
`muse_glimmer_vision` names `vt::Attention` nowhere, which is why it is absent
from §2's table and why `scripts/attention-rung-allowlist.txt` records its stem
as deleted rather than parked. #1545's premise was true when filed and was fixed
by #1579. The residual debt on that tower is a different one and belongs to
[#1566](https://github.com/mudler/vllm.cpp/issues/1566): the tower has no
production caller at all, so the routing landed inside an unreached slice, and
the CUDA A/B was never taken for want of a lease.

## 2. The enumeration

Every live `vt::Attention(` call site in `src/`, `include/`, `examples/`,
`tools/` and `benchmarks/`, found with the checker's own
`checker_text.py::normalize_source` so a commented-out or `#if 0`-ed call is a
deletion here exactly as it is to the compiler. Eight sites, and the widened
scan of §3 returns the same eight, which is how this table is known to be
complete for that spelling.

Sequence lengths are the value the call actually runs at, traced to the line
that computes it. Where a site recomputes the whole span on every invocation
that is stated, because it is the difference between a shape the naive kernel
handles and one it does not.

| # | Site | Subsystem | Causal | head_dim | T at runtime | Flash-eligible | Disposition |
|---|---|---|---|---:|---|---|---|
| 1 | `whisper_audio.cpp:328` | Voxtral / Whisper audio encoder self-attention | no | **64** | **1500**, fixed | yes | A/B EAGER rung, `VT_WHISPER_ENC_EAGER=1`; default is `AttentionDenseFlash` |
| 2 | `qwen3_vl_vision.cpp:531` | Qwen3-VL vision tower, windowed per frame | no | **72** | **784** per frame | yes | A/B EAGER rung, `VT_QWEN3VL_ATTN_EAGER=1`; default is `AttentionDenseFlash` |
| 3 | `kimi_linear_device.cpp:605` | Kimi-Linear device MLA attention core | yes | **192** (padded qk, f32) | whole token span, recomputed per call | **no** | no fast rung exists at this shape; behind `VT_KIMI_DEVICE_MLA`, default off, measured negative |
| 4 | `qwen3_5.cpp:5368` | Qwen3.5 `FullAttnBlock`, the non-paged reference dense arm | yes | **128** | whole token span | yes, in principle | reference golden for `FullAttnBlockPaged`; rerouting moves the golden, not the shipping kernel |
| 5 | `nemotron_h.cpp:676` | Nemotron-H host reference attention mixer | yes | **128** | whole running context, fresh forward per generated token | yes, in principle | host half of the host/device equivalence gate |
| 6 | `nemotron_h_device.cpp:347` | Nemotron-H device attention block | yes | **128** | whole running context | yes, in principle | device half of the same gate; the fast rungs are not bit-identical to this one |
| 7 | `ltx2.cpp:966` | LTX-2.5 DiT **host** self-attention | no | **128** video / **64** audio | **2352** video / **51** audio | moot | CPU-only by construction; on CPU `kAttention` and `kAttentionDenseFlash` are the same registered function |
| 8 | `ltx2_device.cpp:547` | LTX-2.5 DiT **device** self-attention, `VLLM_LTX2_DIT_FLASH_ATTN=0` arm | no | **128** video / **64** audio | **2352** video / **51** audio | yes | naive control arm of a three-rung same-binary A/B; the unset default is `vt::AttentionDenseFa2` |

### 2.1 How each shape was obtained

Grep alone does not give a runtime shape, so each number below names the line
that produces it.

1. **Whisper / Voxtral.** `head_dim = d_model / num_heads`
   (`include/vllm/model_executor/models/whisper_audio.h:62`). The struct default
   is whisper-small, `768/12`, but the production config is
   `VoxtralEncoderConfig()` (`src/vllm/model_executor/models/voxtral.cpp:689-700`):
   `d_model = 1280`, `num_heads = 20`, so **head_dim 64**. T is not the audio
   duration: `const int64_t L = cfg.max_source_positions;  // 1500`
   (`whisper_audio.cpp:185`), fixed because the encoder always consumes a
   conv-downsampled 3000-frame 30 s window.
2. **Qwen3-VL.** `L = grid_thw[0] * grid_thw[1] * grid_thw[2]`
   (`qwen3_vl_vision.cpp:386`). For the checked-in 448x448 fixture at
   `patch_size 16` that is a 28x28 grid, **784 patch tokens** per frame — the
   pre-merge count the tower attends over, not the 196 post-merge tokens the
   language model sees. head_dim 72 is `1152/16` from the real loaded geometry,
   asserted against the real 27B mmproj GGUF through the production loader in
   `tests/vllm/entrypoints/test_gguf_mmproj_reach.cpp:334-335`, and it is the
   value `src/vt/cuda/cuda_ops.cu` already names beside `AttentionDenseFast`.
3. **Kimi-Linear.** The padded query/key head dim is
   `qk_nope_head_dim + qk_rope_head_dim = 128 + 64 = 192`
   (`include/vllm/model_executor/models/kimi_linear.h:85-87`), in f32. This is
   the one site where **no fast rung is available at all**:
   `AttentionDenseFlash` would request `2 * 64 * 192 * 4` = **98,304 bytes** of
   dynamic shared memory against CUDA's default 48 KiB cap, so the launch would
   throw. That is arithmetic from `AttentionDenseFlashSmemBytes` in
   `include/vt/ops.h`, not an estimate, and it is why site 3's reason is a
   capability statement rather than a preference.
4. **Qwen3.5.** `cfg.head_dim` read from `HfConfig`; the served checkpoint is
   `head_dim 128`.
5, 6. **Nemotron-H.** `p.head_dim = GetInt(doc, "head_dim", 128)`
   (`nemotron_h_weights.cpp:838`), and the struct comment pins the checkpoint's
   own geometry as 32 q / 2 kv heads at head_dim 128. Neither arm is an
   incremental decode: `NemotronHGreedyDecode` calls `NemotronHForward` fresh
   per generated token over the full running sequence, so T grows to the whole
   context on every call rather than settling at 1.
7, 8. **LTX-2.5.** Video `attention_head_dim = 128` over 32 heads and audio
   `audio_attention_head_dim = 64` over 32 heads
   (`include/vllm/model_executor/models/ltx2.h:123-131`). At the production
   render geometry `768x448/49f` the video latent grid is **2352 tokens**, and
   the audio stream is **51**, computed at
   `src/vllm/multimodal/ltx2_video.cpp:3226-3231` from `sample_rate 16000`,
   `hop_length 160` and `audio_latent_downsample_factor 4`. Both numbers are the
   ones #1549 attributed its 47.84 s to, so they are already load-bearing
   elsewhere.

### 2.2 The routing verdict: none, and why that is a result rather than an omission

**No site in §2 is routed by this row, and none should be.** Grouped by the
reason, which is the part a future reader needs:

- **Sites 1, 2 and 8 already default to a fast rung.** The naive call is the
  control arm of a same-binary A/B that exists precisely so a speed claim can
  name the rung it beat. `VT_WHISPER_ENC_EAGER`, `VT_QWEN3VL_ATTN_EAGER` and
  `VLLM_LTX2_DIT_FLASH_ATTN=0` are each unset in every shipped configuration,
  and site 8's knob refuses an unrecognised value by name rather than falling
  back (#1751). Rerouting any of them deletes the denominator of a measurement
  already recorded, which is the opposite of the repair #1552 wants.
- **Sites 4, 5 and 6 are reference arms of numeric gates.** The fast rungs are
  NOT bit-identical to `AttentionKernel` — `AttentionDenseFast`'s own header
  says so, the head_dim partial-sum grouping over 32 lanes differs from the
  block version. Site 4 is what `FullAttnBlockPaged` is compared against, and
  sites 5 and 6 are the two halves of one host/device equivalence gate. Moving
  one side makes the two arms measure different things; moving both deletes the
  gate. Neither is the shipping kernel: production Qwen3.5 decode runs
  `FullAttnBlockPaged`, and `ForwardNemotronHForCausalLM` routes to
  `NemotronHPagedForward` whenever the runner supplies paged caches, which it
  always does.
- **Site 3 has no fast rung to route to.** §2.1 item 3 is the arithmetic.
- **Site 7 cannot be routed to anything different.** On CPU, `kAttention`,
  `kAttentionDenseFlash` and `kAttentionDenseFa2` are all registered to the same
  `AttentionKernel` function (`src/vt/cpu/cpu_ops.cpp`), so the edit would be a
  byte-identical no-op that moved the L2 parity reference off the reference op.

There is consequently **no routed site, so no routing reachability mutation and
no new numerics gate is owed by this row**, and none is claimed. That is stated
rather than left to inference, because a report that quietly omits an expected
piece of evidence reads the same as one that had none to give.

### 2.3 A second finding, which #1552 did not ask for and a reader needs

Across all eight sites, **zero are reachable on the naive kernel from a
production entry point in a default configuration on CUDA today**. Sites 1, 2
and 8 need an env var nothing sets; site 3 needs a second env var that is off
and recorded as a measured negative; sites 4 and 6 sit in forwards whose only
callers are under `tests/`; site 5 is a fallback branch the shipped CLI no
longer takes; site 7 is CPU-only, where the op choice is a naming distinction
with no kernel behind it.

This matters for §4 and is the single strongest input to the seam decision: a
runtime warning added today would fire on **nothing** in the shipped tree, and
where it did fire it would fire on the deliberate arms.

## 3. What this row changes: the population, which was never the tree

`scripts/check-attention-rung-consistency.py` states its detection limit
honestly for SPELLINGS — four ways to reach `kAttention` that its regex cannot
see, each named in the docstring. It states nothing equivalent about its
POPULATION, and it closes with a sentence that is false as written:

> A green here therefore means "no unmarked `vt::Attention(` call", never "no
> model is on the naive rung".

A green means no unmarked call **in two named directories, non-recursively, in
`.cpp` and `.h` only**. Both gaps are measured on this row's base `f9af269f9`,
each restored byte-for-byte against a pre-taken `sha256`:

| Probe | Result |
|---|---|
| An unmarked `vt::Attention(` appended to `src/vllm/v1/attention/backend.cpp` | `rc=0`, and the OK line still reports **8** sites |
| An unmarked `vt::Attention(` in a new `src/vllm/model_executor/models/newarch/probe.cpp` | `rc=0`, still **8** sites |

Neither is hypothetical in the way the four spellings are. A model whose
attention lives in a subdirectory, and an attention call that lands in
`src/vllm/v1/` or `src/vllm/multimodal/` rather than in a model translation
unit, are both ordinary shapes for this tree — `src/vllm/multimodal/ltx2_video.cpp`
already drives the LTX-2.5 denoise loop from outside `models/`.

### 3.1 The change

The population becomes the compiled tree rather than two directories:
`src/`, `include/` and `examples/`, walked recursively, over the C++ suffixes
this repository actually uses. `tests/` stays out, deliberately and by name:
the suite constructs unmarked naive calls as fixtures, and including it would
make the checker refuse its own tests.

**Widening reds nothing.** The widened scan over this row's base returns exactly
the same eight sites §2 tables, so the change adds enforcement for the future
without moving a single present verdict. That is measured in §5, not assumed.

The false closing sentence is corrected in the same change, because it describes
the behaviour this change alters and would otherwise be false in a second, new
way.

What this still does NOT detect is unchanged and stays stated: the four
spellings, and any call reached through a function pointer. What closes those is
a compiler-side population, which is a different instrument and not a longer
regex.

## 4. The seam decision — NEEDS_DECISION

#1552 names three options and says they are not equal. They are argued here
against §2's table rather than in the abstract, and the recommendation is
returned as an escalation rather than taken.

### (a) Leave it caller-opt-in

The honest version of (a) is not "do nothing". Since #1544 the tree already
carries a BUILD-TIME gate that makes a silent naive call impossible to merge in
the scanned population: an author who types `vt::Attention` gets a refusal that
names all three fast rungs and the marker form, before review. Every one of
§2's eight sites went through it, and §2.2 shows all eight reasons survive
scrutiny. With §3's widening, the population becomes the tree.

The residual weakness is real and should be stated: the checker cannot judge
whether a recorded reason is TRUE. It enforces that a choice was made and
written down, not that it was the right one. That is a reviewer's job, exactly
as it is for `scripts/fusion-consistency-allowlist.txt`.

### (b) A one-time runtime warning on a large-token `kAttention` selection on CUDA

Implementable in `vt::Attention` (`src/vt/ops.cpp`) in a few lines, behind a
`std::once_flag`, gated on `device.type == kCUDA && t >= threshold`. It changes
no numerics and cannot touch the byte-identity guarantee.

**This row recommends AGAINST it, on three grounds it measured.**

1. **It would fire on nothing.** §2.3: no site reaches the naive kernel on CUDA
   in a default configuration. A gate whose live population is empty is not
   protection, it is a claim about a future caller.
2. **Where it did fire, it would fire on the deliberate arms.** Every CUDA site
   that can reach `vt::Attention` at a non-trivial T — 1, 2 and 8 — is an A/B
   control arm somebody switched on ON PURPOSE to measure the naive rung. A
   warning that shouts at exactly the operator who asked for the naive kernel is
   noise, and noise on a rare channel is worse than silence because it trains
   the reader to skip it.
3. **It cannot make the distinction the marker already makes.** Separating "a
   deliberate reference arm" from "an author who did not know" is the whole
   problem, and at runtime that information is gone. Recovering it needs an
   extra field on `AttentionArgs` — a seam change of (c)'s size that duplicates
   a record the source file already carries. The build-time marker is where that
   distinction is cheap, and it is already there.

The one thing (b) buys that (a) does not is coverage of a call the regex cannot
see. That is a genuine gap, and the instrument for it is the compiler-side
population §3 names, not a warning that also mis-fires on three known-good arms.

### (c) Route `kAttention` by shape on CUDA

The only option that removes the failure mode, and the only one that puts a
byte-identity guarantee at risk. **Not implemented here, per scope.** What this
row adds to the decision:

- The guarantee (c) risks is **narrower than it looks and wider than the decode
  path**. The freeze exists so text decode stays byte-identical, but §2 shows
  three non-decode sites (4, 5, 6) that depend on byte-identity for a reason
  that has nothing to do with decode: they are the reference arms of numeric
  equivalence gates. Shape-routing `kAttention` would silently move those
  goldens. Any (c) spec must therefore enumerate the byte-identity consumers,
  not assume they are the decode shapes.
- **A shape predicate alone is not enough.** Site 3 at head_dim 192 in f32
  cannot launch `AttentionDenseFlash` at all, so (c)'s router needs the
  `AttentionDenseFlashSmemBytes` capability check, not just a token-count
  threshold — which is `supports_head_size()` and is the half vLLM already has
  (`vllm/v1/attention/backend.py:155-163`) and we do not.
- The right shape for (c) is probably **not** a router inside `kAttention` but
  the thing vLLM actually has: a selector a caller ASKS
  (`get_vit_attn_backend(head_size, dtype)`), which leaves `kAttention` frozen
  and makes the fast path the default answer for a caller who does not care.
  That preserves every reference arm by construction, because a reference arm
  names the op instead of asking.

**Recommendation returned for decision: (a), strengthened by §3, and NOT (b).**
Open (c) as its own row with its own spec, with the byte-identity consumer
enumeration above as its first obligation. This row does not take that decision.

## 5. Tests and evidence

Red-before / green-after for every new assertion, against the BASE checker.

| Case | Red before | Green after |
|---|---|---|
| A call in a source file outside the two model directories is a site | base `scan_tree` has no such concept; the shipped-tree probe returns `rc=0` at 8 sites with a live unmarked call present | the call is reported and the run is `rc=1` |
| A call in a SUBDIRECTORY of a model directory is a site | as above, `rc=0` at 8 sites | reported |
| A call in a `.cu` translation unit is a site | base scans `*.cpp` and `*.h` only | reported |
| `tests/` is excluded, so the suite's own fixtures are not sites | — | asserted positively, not left to the shipped tree happening to be green |
| The shipped tree's population is unchanged by widening | — | still exactly the eight sites of §2, asserted by name |

The two shipped-tree probes of §3 are the mutation evidence: each was applied to
the real tree, the checker ran, and the tree was restored and verified against a
`sha256` taken before the edit.

## 6. Reachability

This row ships no product code, so it owes no production-entry-point proof for a
kernel. The checker is reached by `scripts/agent-preflight.sh` and by
`.github/workflows/ci.yml`, both of which already invoke it by name, and the
widened population is exercised there on every run. The deleting mutation for
this change is the red-before column of §5: with the base checker's population,
every new case fails.

## 7. Risks and decisions

- **D1. Widening the population could red an unrelated future file.** Accepted.
  That is the enforcement, and the remedy is one marker comment with a reason,
  which is the same remedy a model file has.
- **D2. The allowlist matches on file STEM, and a wider population makes a stem
  collision more likely.** Not repaired here. The allowlist is currently EMPTY
  and its header restricts it to in-flight removals, so there is nothing to
  collide with today. Recorded under `## Owed`.
- **D3. No new matrix row.** This is an increment on KERNEL-ATTN-DENSE-FLASH, in
  the shape `attention-rung-visibility.md` established. Creating an ENG row
  would owe four record edits and change how two checkers classify the row, for
  no gain to a change that adds no capability.
- **D4. No index row is appended for #1552.** The index already carries one, and
  `check-agent-record.py` refuses a second row for the same issue by design,
  because under `merge=union` a duplicate is what two branches appending the
  same issue look like. The `## Owed` entry in
  [`ltx25-dit-attn-flash.md`](ltx25-dit-attn-flash.md) is repointed here
  instead, which is the record edit this change made stale.

## Owed

- **[#1552](https://github.com/mudler/vllm.cpp/issues/1552) — the seam decision
  itself, returned as NEEDS_DECISION.** §4 argues all three options against a
  measured table and recommends (a) plus §3. Taking (c) needs its own row and
  its own spec, and §4's byte-identity consumer enumeration is its first
  obligation. Owner: this row until that row exists.
- **The four undetectable spellings, and the function-pointer case.** Unchanged
  by this row and restated in §3. What closes them is a compiler-side
  population — the CUDA op registry, or a clang tooling pass over the real
  translation unit — which is a different instrument. Owner: this row.
- **D2, the allowlist's stem matching under a wider population.** No collision
  exists today because the allowlist is empty. Owner: this row.
- **No CUDA timing is claimed anywhere in this spec.** §2's per-site cost
  statements are shapes and arithmetic, not measurements, and the only measured
  wall clock quoted is #1549's 47.84 s, which is that row's number and not this
  row's. No GPU lease was taken and none was needed.

## Now

Spec committed before implementation. The row's work is §2's enumeration, §3's
population widening with its red-before suite, and §4's escalation.
