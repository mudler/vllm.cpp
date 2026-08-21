# Qwen3.8-27B, three engines, one checkpoint, each at its own best

| Field | Value |
|---|---|
| Issue | [#1574](https://github.com/mudler/vllm.cpp/issues/1574) |
| Owning rows | `BENCH-QWEN38-27B-SOTA` (new, [backend-matrix](../backend-matrix.md)) |
| Depends on | `KV-FP8` W2 and W3 (the critical path), `QUANT-QWEN38-27B-NVFP4-ARM` W5, `SPEC-DFLASH2` W6 ([#1314](https://github.com/mudler/vllm.cpp/issues/1314)) whose mechanism landed 2026-08-20 |
| Sibling | [bench-qwen38-27b-four-way.md](bench-qwen38-27b-four-way.md) (#979) keeps the llama.cpp pair and the per-pair not-comparable verdicts. Section 2 below falsifies its premise for the other three engines |
| Subject | `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121` @ `36f717a22990e82c54c1d48ee77c491b87825680`, and `z-lab/Qwen3.8-27B-DFlash2` @ `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` as the draft |
| Host | `dgx.casa`, GB10 sm_121a, under one `rc hold` lease for the whole series |
| Role | operator campaign, branch `row/BENCH-QWEN38-27B-SOTA`, base `e2a9e035dbf8662f4bd87fc21a768d184f547c73`. Section 3 was re-derived when `origin/main` moved from `cae8ace0c` to `947e5f648` mid-change, and the branch was rebuilt onto `e2a9e035d` after it moved twice more |
| Status | `SPIKE`. Scoping only. **No number in this file is measured.** Every number quoted below is the publisher's own claim, attributed as such |

## 1. What this is for

The deliverable is one table a reader can act on: three engines, one
checkpoint, each engine on its own best profile, every flag recorded, and a
recipe that reproduces our column. It is written to be published. A cell we
lose is published as a cell we lose, because a benchmark that can only report
wins is not a measurement.

The competitor position is not inferred. It is published with its flags at
[`r0b0tlab/qwen38-27b-nvfp4-sm121-vllm`](https://github.com/r0b0tlab/qwen38-27b-nvfp4-sm121-vllm)
and its SGLang sibling. Their claims, quoted and not verified here:

| Engine | Profile | dedicated c1 | ladder c1 / c2 / c4 / c6 / c8 |
|---|---|---:|---|
| vLLM | DFlash2 K8 | — | 67.1 / 121.5 / 211.5 / 279.2 / — |
| SGLang | DFlash2 K8 | 28.38 | 68.6 / 124.3 / 212.0 / 276.4 / — |
| vLLM | MTP K3, think-off | 27.83 | 19.24 / 32.00 / 34.61 / — / 82.89 |
| vLLM | DSpark K7, think-off | 28.46 | 16.05 / 28.47 / 43.88 / — / 61.53 |
| SGLang | EAGLE, think-off | — | — / — / — / — / 123.90 |
| vLLM | AR, no speculation | 11.35 | — |

Two of their own methodology rules travel with those numbers and are adopted
here. Dedicated-c1 and ladder figures come from different harnesses and are
never mixed. Their vLLM prefill figure is an e2e wall proxy and is not the
pure-prefill number SGLang's row reports.

## 2. The subject, read from the artifact

`.agents/quantization-matrix.md` records four blockers for
`QUANT-QWEN38-27B-NVFP4-ARM` against `unsloth/Qwen3.8-27B-NVFP4`. **Three are
properties of that artifact and not of the format.** Established 2026-08-21
from `model.safetensors.index.json` and from shard 1's safetensors header by
HTTP range request, on the revision named in the header table. Nothing here is
taken from a model card.

| Recorded blocker | This tree |
|---|---|
| the unconditional `.input_scale` read, against `*.input_scale` appearing ZERO times | **208 present**, `F32` scalar `[]`. The unconditional read succeeds |
| a per-channel BF16 `weight_scale` that `ReadF32Scalar` refuses on count AND dtype | **`F32` scalar `[]`** on every FP8 module. Per-tensor static |
| no representation for a DYNAMIC per-token activation scheme | `config_groups.group_0` sets `"dynamic": false` on both weights and `input_activations` |
| the scheme is never read from the config | **NOT discharged for this artifact** — see the correction below |

The NVFP4 half is the easier one too. unsloth ships `nvfp4-pack-quantized`
W4A**4**; this tree ships **W4A16_NVFP4, `group_size` 16**, weight-only, in the
two-level layout this project already runs:

```
model.language_model.layers.0.mlp.gate_proj.weight          U8       [17408, 2560]
model.language_model.layers.0.mlp.gate_proj.weight_scale    F8_E4M3  [17408, 320]   # 5120/16
model.language_model.layers.0.mlp.gate_proj.weight_scale_2  F32      []
model.language_model.layers.0.linear_attn.in_proj_qkv.weight        F8_E4M3 [10240, 5120]
model.language_model.layers.0.linear_attn.in_proj_qkv.weight_scale  F32     []
model.language_model.layers.0.linear_attn.in_proj_qkv.input_scale   F32     []
```

Accounting, by EXACT suffix. The first revision of this section wrote 594
`weight_scale` and omitted `A_log`, and those buckets summed to 2146 against a
stated total of 2001. The 594 came from a SUBSTRING match that also caught
`weight_scale_2`. **Buckets that do not sum are the check that was skipped**, and
the corrected ones do:

| count | suffix |
|---:|---|
| 937 | `.weight` |
| 401 | `.weight_scale` |
| 208 | `.input_scale` |
| 193 | `.weight_scale_2` |
| 166 | `.bias` |
| 48 | `.A_log` |
| 48 | `.dt_bias` |
| **2001** | **total** |

The 193/208 split matches the publisher's stated 193 W4A16_NVFP4 + 208 FP8.
**W4's accounting gate is pinned to the unsloth NAME SET** (1953 + 15 = 1968)
and does not cover this one.

**`k_scale` and `v_scale` appear ZERO times** while `hf_quant_config.json` sets
`kv_cache_quant_algo: "FP8"`. The fp8 KV scales are therefore defaulted by the
engine, not read from the checkpoint, and `--kv-cache-dtype fp8` is load-bearing
for CORRECTNESS and not only for memory: the publisher's canary is `19 x 23`,
which answers `437` with the flag and `417` without it. That canary is adopted
as this campaign's smoke gate on all three arms.

**The format is ModelOpt, and the fourth blocker is NOT discharged.** The
`quantization_config` declares `quant_method: "modelopt"`, `quant_algo:
"MIXED_PRECISION"`, and an EMPTY `ignore` list against unsloth's 303 entries. It
carries both a `config_groups` block, which is the compressed-tensors shape this
section originally read, and a `quantized_layers` block naming 401 exact
modules, which is the ModelOpt shape. The DECLARED method decides, and it is
`modelopt`, so `ct::Config` stops at `quant_method != "compressed-tensors"` and
**nothing in this tree reads this config at all**. Every routing decision then
falls to a tensor-name probe, which can be silently wrong in both directions.

The work is therefore not "extend W4's resolver to a second name set". It is the
first production wiring of the ModelOpt reader, which is what
[#1603](https://github.com/mudler/vllm.cpp/pull/1603) lands.

**This falsifies #979's premise for three of the four engines.** #979 concluded
that no single quantization is common to all four. That survives for llama.cpp,
whose container is disjoint. It does not survive here: this one checkpoint is
served by vLLM and by SGLang today, and is the checkpoint their numbers were
measured on. #979's matrix owes that correction, and this spec does not make it —
a record edit rides with the change that staled it.

## 3. What blocks our column, and in what order

Re-derived against `947e5f648` on 2026-08-21, and one entry moved a long way
between the first draft of this spec and this one. `SPEC-DFLASH2` W3, W4 and W5
all landed on 2026-08-20: the candidate selector, the path walk, and the GGUF
drafter arm. `RefuseDflash2CandidateSelector` is RETIRED, both containers draft,
and the startup refusal is gone from both. **DFlash2 is no longer a mechanism
gap for this campaign**, which is why the table below reads differently from a
reader's expectation formed at `cae8ace0c`.

| Stage | Buys | Needs | State at `947e5f648` |
|---|---|---|---|
| S1 | the **AR** cell against their 11.35 | `KV-FP8` W2 (CUDA fp8-e4m3 store + fp8 paged-attention read) and W3 (half-sized blocks, `--kv-cache-dtype` threading), then `QUANT-QWEN38-27B-NVFP4-ARM` W5 on this name set | `KV-FP8` W1 landed CPU-ONLY; W2, W3 and W4 are all recorded `later`. W5 unclaimed |
| S2 | the **DFlash2 K8** headline against their 67.1-279.2 | S1, plus `SPEC-DFLASH2` W6, plus the FIRST load of a published artifact | the mechanism is DONE end to end on a GPU. What is owed is the gate and the fact that **no published DFlash2 artifact has been LOADED yet** |
| S3 | the **MTP K3** cell against their 27.83 / 82.89 | S1, plus MTP execution | owed. `docs/USAGE.md` records the artifact present and execution owed |
| S4 | the **concurrency and cache-invalidation** axis | a dense, prefix-cache-capable subject — see D4 | subject UNCHOSEN |

**The critical path is fp8 KV on CUDA, and it is the only thing that gates every
cell.** No cell of this checkpoint can be served correctly without it, because
of what section 2 establishes about the canary. It is owed by a row whose W1 is
CPU-only and whose W2, W3 and W4 all read `later`.

S2 now outranks S3 by more than the publisher's own guidance that DFlash2 wins
c1-c6 while MTP wins c8-and-up. S2's mechanism exists and needs a gate; S3's
does not exist at all. If GPU time is scarce, S3 is the cell to drop.

## 4. Methodology

### D1 — `--enforce-eager` is their recipe and it cannot be our denominator unqualified

Every published profile keeps `--kv-cache-dtype fp8 --enforce-eager
--no-enable-prefix-caching`. `AGENTS.md` §Gates says never use `--enforce-eager`
as the denominator, and vLLM's production configuration is the denominator.
Those two statements collide on this subject.

Resolution: measure the vLLM arm **twice** — once at the published recipe
(eager, which reproduces their claim and is the only configuration their numbers
describe), and once with graphs enabled if it runs at all on this checkpoint.
The denominator is the FASTER of the two. If graphs refuse on the hybrid GDN
family, record the refusal and its message, and the eager configuration becomes
vLLM's production configuration on this box **as a measured fact rather than as
a concession**. Never report a ratio against eager without stating which of the
two produced it.

### D2 — one harness for all three arms, and it already exists

Their ladder is `r0b0bench` and their dedicated-c1 is `run_perf_suite.sh`. We
have neither, and adopting an unpinned harness for our own claim would make the
claim unreproducible. **Their published numbers are reference, never a
denominator.** A denominator is a server we ran.

Do not write a new client. `tools/bench/run_serve_low.py` is already the
three-arm harness this campaign needs: it takes `--engine ours|vllm|sglang`,
`--concurrency`, `--repetition`, `--model-repo` with `--model-revision`, and it
writes fail-closed evidence. Its timed requests are executed by the unmodified
pinned `sglang.bench_serving` at `28b095c`, which is a THIRD-PARTY client for
two of the three arms and therefore a better instrument than one we wrote. It
has already measured our own server over SSE, so it does not repeat the trap
that deletes our arm: `vllm bench serve` marks our requests FAILED on our SSE
keepalive frame, and reports the deletion as a result.

What the harness owes this campaign is a subject, not a rewrite: `--model-key`
admits only `27` and `35`, which are the Qwen3.6 gate models. Extending it to
this checkpoint is a scoped change with its own red-first test, and it is the
first code this campaign lands.

### D3 — clocks, and what to do when the pairing is discarded

Pin the clocks on the HOST inside the lease (`sudo nvidia-smi -lgc`), which is
what the `rc hold` plus `ssh` authorization exists for
([#1354](https://github.com/mudler/vllm.cpp/issues/1354)). Then verify the pin
took, rather than assuming it: the 2026-08-19 series recorded 12.9% to 26.4%
within-run SM-clock spread against a 5% ceiling and `gpu_clock_state compare`
returned `PAIRING_VERDICT=DISCARD` on every c1 pairing.

If a pairing is still discarded after pinning, publish the absolutes and mark
the ratio OWED. Do not publish a ratio the clock gate refused.

### D4 — the cache-invalidation axis needs a different subject, and this one cannot carry it

The requested workload is high concurrency with a DIFFERENT system prompt per
request, to measure prefix-cache hit-rate collapse and eviction. **It cannot
run on this subject.** Qwen3.8-27B is the hybrid GDN family: the publisher
serves every profile with `--no-enable-prefix-caching`, and our own port
defaults hybrid and attention-free models to the no-prefix coordinator, which
mirrors vLLM. With no cache on any of the three engines, distinct system prompts
measure the scheduler, not the cache.

S4 therefore takes a DENSE, full-attention subject that all three engines serve
with automatic prefix caching genuinely on. The subject is not chosen in this
spec, because choosing it needs a survey this spec has not run. Recorded as an
open decision, and the survey is the first task of S4.

Two axes, never mixed, and never averaged into one headline: this checkpoint
carries CONCURRENCY SCALING with the cache off, and the S4 subject carries CACHE
INVALIDATION with the cache on.

### D5 — correctness before speed, on every arm

The `19 x 23 -> 437` canary runs on all three servers before any timing leg. A
speed number from an arm that failed the canary is not a slow result, it is a
different computation. Beyond it, the token gate for our arm against the pinned
vLLM oracle is owed by `QUANT-QWEN38-27B-NVFP4-ARM` W5 and is not re-specified
here.

### D6 — the SGLang arm runs at THEIR pinned digest, and that is not a pin advance

The competitors' SGLang recipe pins
`lmsysorg/sglang@sha256:3c0abdf41ef22de9d7a859dc16ed71eae69452e36c91f071a25e60c85a6d1fc6`
plus a DFlash2 overlay built from their `docker/Dockerfile.dflash2`. This
project's SGLang oracle is pinned elsewhere: v0.5.15 `f63458b5`, image
`@sha256:d0a667e` ([sglang.md](../oracles/sglang.md)).

Measuring "each engine at its best" on their subject means running THEIR digest,
because that is the artifact their published numbers describe and the only one
carrying DFlash2 at all. Their digest is a real pin — a content-addressed image
is exactly what `AGENTS.md` asks an oracle record to carry — so this is not the
unpinned-upstream hazard.

**It is still not our pin, and nothing here advances one.** The SGLang oracle
record is untouched, `gateable` does not move, and no number produced against
their digest may be cited as an oracle result for any other row. It is a
COMPETITOR measurement scoped to this campaign, recorded with its digest beside
it. Advancing the project's SGLang pin stays separate deliberate work that
[bench-qwen38-27b-four-way.md](bench-qwen38-27b-four-way.md) already lists as
unowned.

The same rule governs the vLLM competitor arm. Its image
`ghcr.io/r0b0tlab/qwen38-27b-nvfp4-sm121:v0.27.2rc0-sm121` is vLLM `v0.27.2rc0`
at `7f7a32c`, far ahead of our parity pin `555967922`. The pinned oracle stays
the MIRROR for behaviour; their image is a COMPETITOR arm. Those are two
different jobs and neither may stand in for the other.

## 5. The workloads

| ID | Shape | Reports | Runs on |
|---|---|---|---|
| `L-LADDER` | fixed prompt set, c1 / c2 / c4 / c6 / c8, greedy, think-off | aggregate client tok/s per concurrency | this checkpoint |
| `L-C1` | dedicated single stream, 512 -> 2048 and 1024 -> 256 | median tok/s, TTFT, TPOT | this checkpoint |
| `L-DISTINCT` | c8 / c16 / c32, every request a DIFFERENT system prompt, shared user suffix | tok/s, TTFT, and the engine's own reported prefix-cache hit rate | the S4 dense subject |
| `L-SHARED` | the control for `L-DISTINCT`: identical system prompt, everything else equal | the same three | the S4 dense subject |

`L-DISTINCT` without `L-SHARED` measures nothing: a hit rate has to be read
against the hit rate the same workload reaches when the prefix IS shared. The
pair is the measurement and neither half is published alone.

## 6. Gates

| G | Condition |
|---|---|
| G1 | the `19 x 23 -> 437` canary passes on all three servers, at the exact flags each is measured with |
| G2 | our arm is token-exact against the pinned vLLM oracle on this checkpoint, greedy, before any timing leg (owed by W5) |
| G3 | every reported ratio carries a `gpu_clock_state compare` verdict that is not `DISCARD`, or the ratio is published as OWED with its absolutes |
| G4 | every arm's build, revision, image digest, flags, and the checkpoint sha are recorded beside its number |
| G5 | `L-DISTINCT` is published only together with `L-SHARED` |
| G6 | the reproduce recipe in `docs/USAGE.md` is executed from scratch by someone who did not write it, on the recorded artifacts, before publication |

## 7. Stop conditions

Stop and report rather than working around, if: the checkpoint's four shards do
not match the publisher's `final-sota-shards.sha256`; an arm fails G1; the box
has under 10 GB of headroom at the configuration under test, which is what
killed the c8 vLLM denominator on 2026-08-19; or a competitor image cannot be
pulled, in which case that column is `REMOTE_UNVERIFIED` and never absent.

## Now

`SPIKE`. This spec and issue [#1574](https://github.com/mudler/vllm.cpp/issues/1574)
are the whole deliverable of this change. Nothing is measured.

**Only ONE new artifact has to be staged.** The draft this campaign needs is
`z-lab/Qwen3.8-27B-DFlash2` @ `50307d4c`, 3 848 817 896 bytes — the same tree
`SPEC-DFLASH2` already has authorized and recorded in `docs/USAGE.md`, and the
same one the publisher calls "the 3.6 GB z-lab draft". The new download is the
target alone: four shards, about 20.5 GiB, verified against the publisher's
`final-sota-shards.sha256`. The publisher records a publication bug in which the
hardlinked body shards were omitted and only `model-00004-of-00004.safetensors`
arrived, so a tree with one shard is incomplete rather than small, and the
sha256 file is the check that says which.

The next action is S0, which needs the box and no new code: take the lease, pin
the clocks and verify the pin took, stage the checkpoint and the draft under
`CHECKPOINT_ROOT`, verify the four shards against the publisher's sha256 file,
and then answer the one question that resizes S1 — **does our current `main`
load this checkpoint's FP8 tower and NVFP4 half, now that the three unsloth
blockers are known not to apply here?** W4's resolver landed. Nobody has pointed
it at this artifact. The answer is worth a day of scoping because it decides
whether S1 is a wave or a campaign.

## Owed

- The dense subject for S4 (D4), and the survey that chooses it.
- The correction #979's matrix owes, once section 2's finding lands with a
  change that stales it.
- A `BENCH-QWEN38-27B-SOTA` row in [backend-matrix.md](../backend-matrix.md),
  which this change adds, and the `## Outcome` section this spec owes at `DONE`.
- Whether vLLM runs this checkpoint with CUDA graphs at all on sm_121a (D1). If
  it does, their entire published table is an eager under-report of their own
  engine, and that is a finding about their claim rather than about ours.
- MTP execution for S2, unowned today.
