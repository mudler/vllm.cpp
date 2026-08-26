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

## ACCEPTED lane exception: `qwen4_exp` (`MODEL-MM-QWEN4-EXP`, [#1978](https://github.com/mudler/vllm.cpp/issues/1978))

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

See [`../specs/qwen4-exp-flash-next.md`](../specs/qwen4-exp-flash-next.md)
`## Oracles`.
