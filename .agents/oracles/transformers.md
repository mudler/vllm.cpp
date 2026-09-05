# HuggingFace `transformers` — the reference vLLM itself mirrors

Where vLLM implements a model, `transformers` is not the authority: vLLM is,
including where the two differ. What `transformers` answers is the layer beneath
that — a model's own reference implementation, and the processors, feature
extractors and tokenizers vLLM delegates to rather than reimplements. That is
why it is a legitimate oracle for a preprocessing golden even on models where
vLLM is the behavior spec.

**The pin is the one resolved inside the pinned vLLM environment**, not an
independently chosen release: `transformers` 5.14.1 is what
[`../upstream-sync.md`](../upstream-sync.md) records alongside vLLM
`555967922`. Pinning it separately would let the oracle environment hold two
different `transformers` at once, which is the drift this registry exists to
stop.

Precedent for it running as an oracle: the AUDIO track's A0 captured the
`WhisperFeatureExtractor` log-mel reference, the mel filterbank and the audio
placeholder expansion directly from it
([`../specs/audio-track.md`](../specs/audio-track.md) §A0), and the C++ STFT is
gated against those goldens.

**It cannot adjudicate a near-tie, and one attempt is recorded here so the next
agent does not repeat it.** A CPU `transformers` probe of the three Qwen3.8-27B
divergences ([`../specs/qwen38-27b-bf16-gate.md`](../specs/qwen38-27b-bf16-gate.md),
[#915](https://github.com/mudler/vllm.cpp/issues/915)) reported all three as
exact ties — and every runner-up gap it printed was an exact multiple of
**0.125**, one bf16 ULP in that exponent range. An instrument quantized to one
ULP cannot resolve a gap below one ULP, so "tied in bf16" does not imply "tied
in the model" and agreement with it is not confirmation. vLLM computes logprobs
from **fp32** and separates pairs bf16 collapses, so a logit-margin verdict
comes from the pinned vLLM oracle; this one stands only as a secondary
cross-check, with that limitation part of the result.

```oracle-pin
id = transformers
role = secondary
upstream = https://github.com/huggingface/transformers
scope = a model, processor, feature extractor or tokenizer's own reference implementation, at the version the pinned vLLM environment resolves
pin = 5.14.1
pin_label = 5.14.1
pinned_on = 2026-07-26
gateable = yes
evidence = .agents/specs/audio-track.md
```

## EXPIRED lane exception: `qwen4_exp` (`MODEL-MM-QWEN4-EXP`, [#1978](https://github.com/mudler/vllm.cpp/issues/1978))

**EXPIRED 2026-08-31, on its own stated condition. Kept in full, not deleted,
because a lane that vanishes cannot be audited and the next reader needs to see
what was accepted as well as what ended it.**

The exception's stop condition is written into it below: "It expires the moment
vLLM registers `qwen4_exp`." vLLM landed `[Model] Support Qwen3.8-Flash-Next
(#53896)` at `e126687a9a` on 2026-08-31, adding `vllm/models/qwen4_exp/` and
registering `Qwen4ExpForCausalLM`, `Qwen4ExpForConditionalGeneration` and
`Qwen4ExpMTP`. The condition has fired.

What that changes: `transformers` stops being the ALGORITHM oracle for
`model_type: qwen4_exp` and demotes to the preprocessing and
checkpoint-semantics role it holds for every other model in this file. The row
reconciles onto vLLM, and the reconciliation is recorded in
[`../specs/qwen4-exp-flash-next.md`](../specs/qwen4-exp-flash-next.md) under
`## Oracles` and `### Component-by-component reconciliation`
([#2489](https://github.com/mudler/vllm.cpp/issues/2489)).

What it does NOT change: the pin above stays where it is, because 5.16.0 is
still the release that contains the architecture and `transformers` is still an
oracle for its ordinary role. Nothing built under the exception is invalidated
by its expiry; two of its load-bearing calls, the unweighted mean pool and the
divide-after-the-head-sum block score, were checked against vLLM's own kernels
and CONFIRMED.

Also worth carrying: `e126687a9a` is NOT reachable from the vLLM parity pin
`555967922` and is 595 commits ahead of it. Every citation of it is a forward
reference to an unpinned upstream. That does not un-fire the condition, which is
about what vLLM implements and not about what our pin has caught up with.

The accepted record follows unchanged.


**ACCEPTED by the developer, 2026-08-26.** The `oracle-pin` block above is unchanged
and remains the pin for every other consumer; this lane pin is additional and
narrower. It was put as an explicit accept-or-reject because it changes the
semantics of the invariant this file exists to hold, and it was accepted on that
footing rather than passed as housekeeping.

`Qwen/Qwen3.8-Flash-Next` declares `Qwen4ExpForConditionalGeneration` /
`model_type: qwen4_exp`. vLLM does not implement it: read live 2026-08-26 at
`origin/main` = `6a5e8f5979`, there is no `qwen4*` path and no registry entry, and a
repository-wide search for `qwen4` returns zero results. `vllm-omni` likewise. The
reference implementation is
[transformers#48337](https://github.com/huggingface/transformers/pull/48337),
merged 2026-08-26.

**The pin above cannot serve that row: 5.14.1 does not contain `Qwen4Exp`.**

The rule this would except is stated above as "Pinning it separately would let the
oracle environment hold two different `transformers` at once, which is the drift
this registry exists to stop." The argued exception is narrow, and its narrowness is
the whole case: that invariant guards a **vLLM environment** against drifting from
the `transformers` it resolves. For `qwen4_exp` there is no vLLM implementation, so
no such environment exists and nothing can drift from it. A lane pin here cannot
produce the inconsistency the rule prevents.

Its scope and expiry, both binding if accepted:

- It covers `model_type: qwen4_exp` and nothing else. Every other model, processor,
  feature extractor and tokenizer continues to resolve against 5.14.1.
- It supplies the **algorithm** only. Per the row's direction, the optimized form of
  each primitive still mirrors vLLM, which is the polarity AGENTS.md sets and which
  a missing model registration does not suspend.
- **It expires the moment vLLM registers `qwen4_exp`.** At that point the row
  reconciles onto vLLM and `transformers` demotes to the preprocessing role it holds
  everywhere else in this file. That is a stop condition in the row's spec, not a
  reminder.
- `gateable` for the lane is **no** until an arm runs, which is currently blocked on
  memory rather than software: nothing published fits any fleet device. Constructing
  a config proves nothing, and this file's own precedent applies.

### The lane pin, and how it was bounded

`transformers` **5.16.0**, and it is a real release rather than a branch SHA, which
was not the expected outcome. `Qwen4Exp` merged to `main` at 2026-08-26T12:03:40Z and
`v5.16.0` was published at 2026-08-26T12:35:15Z, 32 minutes later, so the release
carries it by a margin of half an hour.

Bounded rather than assumed, because "the release is newer than the merge" is an
argument and not a check. Measured 2026-08-26 by fetching the model file at each tag:

| Revision | `src/transformers/models/qwen4_exp/modeling_qwen4_exp.py` |
|---|---|
| `v5.16.0` | HTTP **200** (present) |
| `v5.15.0` | HTTP **404** (absent) |

`v5.16.0`'s `models/auto/auto_mappings.py` carries 5 occurrences of `qwen4_exp`, so
the registration landed with the model rather than trailing it. 5.16.0 is therefore
the FIRST release containing this architecture, which is the tightest pin available
and the one a lane exception should take.

```oracle-pin-lane
id = transformers
lane = qwen4_exp
role = secondary
scope = the algorithm for model_type qwen4_exp ONLY; every other model, processor, feature extractor and tokenizer stays on the pin above
pin = 5.16.0
pin_label = 5.16.0
pinned_on = 2026-08-26
accepted_by = developer, 2026-08-26
expires = when vLLM registers qwen4_exp
expired_on = 2026-08-31
expired_by = vllm-project/vllm e126687a9a ([Model] Support Qwen3.8-Flash-Next #53896)
expiry_record = https://github.com/mudler/vllm.cpp/issues/2489
gateable = no
gateable_reason = no published artifact fits any fleet device; blocked on memory, not software
owner_row = MODEL-MM-QWEN4-EXP
issue = https://github.com/mudler/vllm.cpp/issues/1978
evidence = .agents/specs/qwen4-exp-flash-next.md
```

**`gateable = no` and the version string is UNMEASURED.** The value above is the
release that provably contains the model, established by fetching its source. It is
NOT a `transformers.__version__` read off a running oracle, and this file's own
precedent says an oracle is gateable only once it demonstrably builds and runs the
model. Resolving the runtime string is owed to the first wave that stands an oracle
up. Do not promote this pin to `gateable = yes` by editing the line.

**The version string is no longer unmeasured; `gateable` still is.** W4
([#1991](https://github.com/mudler/vllm.cpp/issues/1991)) stood the lane oracle up
and `transformers.__version__` reads **5.16.0** on a live import, with
`transformers.models.qwen4_exp.modeling_qwen4_exp.Qwen4ExpTextQSAIndexer`
importable and running on CPU against random weights.
`tests/vllm/models/fixtures/gen_qwen4_exp_qsa_goldens.py` refuses to emit under any
other version string, so that reading is now executable rather than recorded. This
resolves only the half this note said was owed. `gateable` stays `no`: running one
module on random weights is not running the model, no published artifact fits any
fleet device, and the field means what `## Gateability` says it means.

See [`../specs/qwen4-exp-flash-next.md`](../specs/qwen4-exp-flash-next.md)
`## Oracles`.

## Lane exception: `glm5_next` (`MODEL-MM-glm5-next-glm5-next-for-conditional-generation`, [#2096](https://github.com/mudler/vllm.cpp/issues/2096))

**This is the SECOND application of the exception the developer accepted on
2026-08-26, not a new one and not a separately accepted one.** The `qwen4_exp`
section earlier in this file argues the exception and records the acceptance.
The argument there is the argument here, unchanged, so this section states what
is different about this lane and does not re-argue the rule. The `oracle-pin`
block earlier in this file stays at 5.14.1 and remains the pin for every other
consumer. If a reviewer holds that each lane needs its own accept-or-reject,
that is a `NEEDS_DECISION` on this section, and nothing else in the row depends
on the answer.

`zai-org/GLM-5.3-Flash` declares `Glm5NextForConditionalGeneration` /
`model_type: glm5_next`. vLLM implements it at no revision, re-verified
2026-08-27 in the local oracle checkout rather than transcribed:
`git grep -n "Glm5\|glm5_next" -- vllm/` exits 1 with no output at the parity
pin `555967922` and at vLLM `origin/main` = `d85708f7a4`, which has already
moved past the `c71f6f8a81` the row's spec read on 2026-08-26. The same grep for
`Glm4MoeForCausalLM` at the pin returns a hit, so the search itself works and
the empty result is absence rather than a failed command.
[vllm#53906](https://github.com/vllm-project/vllm/pull/53906) "[Model] add
GLM-5.3-Flash support" is OPEN and unmerged, re-read 2026-08-27. An unmerged
pull request is not a revision and cannot be pinned, so vLLM supplies nothing
here. `vllm-omni`, SGLang and llama.cpp implement nothing either;
[`../specs/glm5-next-flash.md`](../specs/glm5-next-flash.md) `## Oracles`
carries each search and its result.

**The pin earlier in this file cannot serve this row: 5.14.1 does not contain
`Glm5Next`.**

### The `glm5_next` lane pin, and how it was bounded

`transformers` **5.16.1**. The implementing commit is `eb4d9e2a64a0`
([transformers#48342](https://github.com/huggingface/transformers/pull/48342),
"[Glm 5.3 Flash] GLM 5.3 Flash Support"), merged 2026-08-26T14:26:41Z. `v5.16.1`
was published 2026-08-26T14:50:01Z, 23 minutes later.

Bounded rather than assumed, because "the release is newer than the merge" is an
argument and not a check. Re-measured 2026-08-27 by fetching
`src/transformers/models/glm5_next/modeling_glm5_next.py` at each tag:

| Revision | Published | `modeling_glm5_next.py` |
|---|---|---|
| `v5.15.1` | 2026-08-19T10:50:47Z | HTTP **404** (absent) |
| `v5.16.0` | 2026-08-26T12:35:15Z | HTTP **404** (absent) |
| `v5.16.1` | 2026-08-26T14:50:01Z | HTTP **200** (present) |

5.16.1 is therefore the FIRST release containing this architecture, which is the
tightest pin available and the one a lane exception must take. It is also the
LAST release as of 2026-08-27: `v5.16.2` does not exist, and its 404 is the tag
missing rather than the file missing, which was separated by fetching `setup.py`
at both tags (`v5.16.1` HTTP 200, `v5.16.2` HTTP 404). `setup.py` at `v5.16.1`
declares `version="5.16.1"`, and `models/auto/auto_mappings.py` at that tag
carries 8 occurrences of `glm5_next`, so the registration landed with the model
rather than trailing it.

**The `qwen4_exp` lane pins 5.16.0 and this lane pins 5.16.1. Do not tidy one
onto the other.** `Qwen4Exp` merged 2026-08-26T12:03:40Z, before the 5.16.0 cut
at 12:35:15Z; `Glm5Next` merged 2026-08-26T14:26:41Z, after it. Two lanes, two
releases, one day apart. Each lane takes the first release that contains its own
architecture, which is what a lane pin is for.

```oracle-pin-lane
id = transformers
lane = glm5_next
role = secondary
scope = the algorithm for model_type glm5_next ONLY; every other model, processor, feature extractor and tokenizer stays on the pin above
pin = 5.16.1
pin_label = 5.16.1
pinned_on = 2026-08-27
accepted_by = precedent, the qwen4_exp lane exception the developer accepted 2026-08-26; NOT a fresh acceptance
expires = when vLLM registers glm5_next
gateable = no
gateable_reason = no oracle has ever run this model, and none can on this fleet; the reference needs 305.78 GiB (FP8) or 598.5 GiB (BF16) resident and the largest reachable device is dgx:gpu0 at ~119.63 GiB. Owed as O1 by https://github.com/mudler/vllm.cpp/issues/1998
owner_row = MODEL-MM-glm5-next-glm5-next-for-conditional-generation
issue = https://github.com/mudler/vllm.cpp/issues/2096
evidence = .agents/specs/glm5-next-flash.md
```

Its scope and expiry, both binding:

- It covers `model_type: glm5_next` and nothing else. Every other model,
  processor, feature extractor and tokenizer continues to resolve against
  5.14.1.
- It supplies the **algorithm** only. The optimized form of each primitive still
  mirrors vLLM, which is the polarity AGENTS.md sets and which a missing model
  registration does not suspend.
- **It expires the moment vLLM registers `glm5_next`.** At that point the row
  reconciles onto vLLM and `transformers` demotes to the preprocessing role it
  holds everywhere else in this file. That is a stop condition in the row's
  spec, not a reminder.

**`gateable = no`, and unlike the `qwen4_exp` lane this one can never become
`yes` on this fleet.** The bar AGENTS.md sets is that the oracle demonstrably
builds and runs THE MODEL. The published artifacts are 305.78 GiB (FP8) and
598.5 GiB (BF16); the largest device this project can reach is `dgx:gpu0` at
~119.63 GiB of unified memory. No device here, and no combination of them, can
execute the reference implementation. The row's `## Gates` builds a tiny-shape
reference oracle on CPU from this pin, which is a real oracle for the NUMERICS
of each component and is NOT gateability for the MODEL. Do not promote this pin
to `gateable = yes` by editing the line.

**The version string is UNMEASURED.** 5.16.1 is the release that provably
contains the model, established by fetching its source over HTTP. It is not a
`transformers.__version__` read off a running oracle. Resolving the runtime
string is owed to the first wave that stands the tiny-shape oracle up, which is
W2.

**No gate reads the block above, and W0 measured that rather than assuming it.**
`scripts/check-oracle-pins.py` matches ```` ^```oracle-pin\n ````, so an
`oracle-pin-lane` fence does not match it and no lane block in this file is
parsed by any checker in the tree. The checker's `--self-test` corpus and
`tests/scripts/test_check_oracle_pins.py` name no lane case. A lane pin is
therefore a documentary record whose fields go unchecked, including `pin`,
`gateable` and `expires`. The row records this as O13 and
[#2099](https://github.com/mudler/vllm.cpp/issues/2099) owns it. Read "the
checker accepts the lane pin" as "the checker stays green", never as "the
checker validated these fields".

See [`../specs/glm5-next-flash.md`](../specs/glm5-next-flash.md) `## Oracles`.
