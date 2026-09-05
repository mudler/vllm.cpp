# Limb 3 of the near-tie gate: is there a vehicle on this fleet at all?

Row: `QUANT-QWEN38-27B-GGUF-ARM`
Issue: [#2864](https://github.com/mudler/vllm.cpp/issues/2864)
Refs: [#2854](https://github.com/mudler/vllm.cpp/issues/2854),
[#2497](https://github.com/mudler/vllm.cpp/issues/2497),
[#2510](https://github.com/mudler/vllm.cpp/issues/2510),
[#2534](https://github.com/mudler/vllm.cpp/issues/2534),
[#2740](https://github.com/mudler/vllm.cpp/issues/2740)

**VERDICT: NO. Limb 3 is not satisfied, and no admissible vehicle for it exists
on this fleet.**

## 0. Now

The determination stands, no benchmark ran, and no row changes lifecycle state.
`STRIX_ARM_SPEED_RATIFIED_BY` stays unset and
[`../scripts/rocm-strix-ourarm-staged.sh`](../scripts/rocm-strix-ourarm-staged.sh)
stays refusing.

## 1. Scope

The 2026-07-20 ratification's item 3, quoted in
[`qwen38-27b-q4km-neartie-band-adjudication.md`](qwen38-27b-q4km-neartie-band-adjudication.md)
§7:

> ALSO verify the forward on a BIGGER dense model ... where vLLM IS
> deterministic, for a clean STRICT token-exact pass.

That spec's §5.5 already reports `LIMB_BIGGER=NO_VEHICLE`, and §7 disqualifies
three candidates. **This spec is not a rescore of that verdict.** §7 searched the
*records* -- the matrices, the parity ledger, `tests/` and `scripts/` -- and
asked whether an existing gate satisfies the limb. It closes with "Candidate 3, a
smaller Q4_K_M model. Not found."

This spec asks the other question, which nobody has asked: **does an artifact
exist on this fleet that a NEW gate could be built on?** Two things changed that
make it worth asking.

- [#2740](https://github.com/mudler/vllm.cpp/issues/2740) established by
  execution that the pinned vLLM `5559679229` builds and runs on `strix:gpu0`,
  and that `vllm-gguf-plugin` at `d4c1f0d082fc7cd4350da56689109a01c1f29d6c`
  loads and generates from a Q4_K_M GGUF there. Before that run there was no
  vLLM on this board at all, so a vehicle search had nothing to search for.
- `3fede4162` landed IQ3_S, which closed the loader hole
  [#2510](https://github.com/mudler/vllm.cpp/issues/2510) named. Any vehicle
  search that predates it was run against a narrower loader than the one that
  exists now, and §4 below records what that changes.

So limb 3 is the one limb of the three whose failure is a **missing
measurement** rather than an unmade decision. Limb 2's failure turns on a
denominator choice [#2534](https://github.com/mudler/vllm.cpp/issues/2534) owns
and this spec does not touch it.

**Out of scope, and measured to be out of scope.** No throughput, latency or
memory figure for vllm.cpp appears here and none was measured; no model was run.
`AGENTS.md` §Gates admits no performance result from an arm whose declared token
gate has not passed, and #2497 already carries one retraction for exactly that.

## 2. The six conditions

A limb-3 vehicle has to satisfy all six at once. They are stated before the
measurement so the outcome cannot be fitted to the inventory afterwards.

| # | Condition | Why the limb needs it |
|---|---|---|
| 1 | The GGUF stores **k-quant** tensors | "same forward code": the arm's kernels are `DotQ4K`, `KQuantGemmK`, `QuantizeQ8KK`, `MatmulBTQuantKernelRocm`, which serve the k-quant tier and compile only under `VLLM_CPP_HIP` |
| 2 | Its `general.architecture` has an arm in **our** GGUF dispatch | otherwise our engine refuses the file and there is no numerator |
| 3 | The **pinned vLLM** registers the family | limb 3 is scored against the primary oracle, whose determinism the limb's own text names |
| 4 | **Dense**, not MoE | the limb says "dense model" |
| 5 | It **fits the board**: 65536 MiB firmware VRAM carve, 59934 MiB free before load, 62 GiB host RAM | measured in [`../../docs/bench-evidence/rocm-strix-qwen38-q4km-20260901.md`](../../docs/bench-evidence/rocm-strix-qwen38-q4km-20260901.md) |
| 6 | **Not the arm's own model** | limb 3 exists to corroborate the arm from outside it; the arm scoring itself is not corroboration |

Condition 6 is why the arm's own `Qwen3.8-27B-Q4_K_M.gguf` is excluded even
though it satisfies 1 through 5. #2740 already ran our engine against the pinned
vLLM on exactly that artifact and recorded divergence on 5 of 6 prompts. Running
it again would re-derive a committed result, not establish limb 3.

## 3. The measurement

[`../../docs/bench-evidence/limb3-vehicle-search-20260904/`](../../docs/bench-evidence/limb3-vehicle-search-20260904/).
`vehicle_scan.sh` re-runs the whole search; `vehicle-scan.txt` is its verbatim
output on `f2930e918`, and `gguf_header.py` is the header reader it drives.
Conditions 1 through 5 are read off the artifacts' own bytes and off two
checkouts, not off any document. It needs no GPU and no lease.

**The predicate is evaluated against the artifact histogram, never against the
file name.** #2510 is usually restated as "our reader cannot load the UD quant
family", which reads like a property of a name. It is a property of the bytes,
and two candidates here are decided by type ids their names do not disclose.

### 3.1 What our build accepts, from the tree

`src/vllm/entrypoints/model_loader.cpp` `kGgufArchArms` dispatches eight
architectures and its default refuses by name:

```
deepseek4, muse-glimmer, qwen35, qwen35moe, qwen3next, qwen4exp, glm5next, glm-dsa
```

There is **no `llama` arm and no `qwen3` arm**. This closes an avenue that would
otherwise look open: converting one of the fleet's bf16 checkpoints
(`qwen3-4b-bf16`, `llama32-1b-instruct-bf16`) to a k-quant GGUF would produce a
file our engine refuses on architecture, whatever its quant mix.

`gguf_reader.cpp` `GgmlTraits()` throws `gguf: unknown ggml type id N` for any id
it has no case for, so **one unaccepted tensor refuses the whole file**. The
accepted set, printed from the tree by the scan, is
`0 1 2 6 8 10 11 12 13 14 16 17 18 19 20 21 22 23 24 25 26 27 28 30 39 40 41 66`.
Id 7 (`Q5_1`) is absent, and that decides the Nemotron candidate below.

### 3.2 What the pinned vLLM registers

Measured against `~/_git/vllm` at `5559679229`, asserted equal to the pin by the
scan before any answer is read from it:

| our GGUF arm | vLLM `registry.py` hits |
|---|---|
| `deepseek4` -> `DeepseekV4ForCausalLM` | 1 |
| `qwen35` -> `Qwen3_5ForConditionalGeneration` | 1 |
| `qwen35moe` -> `Qwen3_5MoeForConditionalGeneration` | 2 |
| `qwen3next` -> `Qwen3NextForCausalLM` | 1 |
| `glm-dsa` -> `GlmMoeDsaForCausalLM` | 1 |
| `muse-glimmer` -> `MuseGlimmer*` | **0** |
| `qwen4exp` -> `Qwen4Exp*` | **0** |
| `glm5next` -> `Glm5Next*` | **0** |

Two families own a staged artifact without appearing in `kGgufArchArms`, so the
scan probes them too rather than leaving them unmeasured:

| family | vLLM `registry.py` hits |
|---|---|
| `Laguna*` | 1 |
| `NemotronHForCausalLM` | 2 |

### 3.3 The decision table

77 GGUFs are staged on this fleet. Every one was read. Collapsed to the
model-bearing files, dropping `clip` projectors, `dflash` draft heads and the
`audiocpp` RVQ sub-module:

| Artifact | Bytes | Arch | k-quant | Dense | Fits | Our arm | vLLM pin | Verdict |
|---|---|---|---|---|---|---|---|---|
| `Qwen3.8-27B-Q4_K_M.gguf` | 17,106,775,008 | `qwen35` | yes | yes | yes | yes | yes | **fails 6**: it IS the arm |
| `Qwen3.8-27B-UD-Q4_K_M.gguf` | 16,464,440,224 | `qwen35` | yes | yes | yes | reader: yes (now) | yes | **fails 6**: same model |
| `Qwen3.8-27B-BF16.gguf` | 53,808,281,952 | `qwen35` | **no** (BF16+F32) | yes | yes | yes | yes | fails 1, and 6 |
| `muse-glimmer-30B-kquant-17gb.gguf` | 16,756,681,056 | `muse-glimmer` | yes | yes | yes | yes | **no** | **fails 3 only** -- see §5 |
| `muse-glimmer-30B-kquant-dynamic.gguf` | 19,653,957,984 | `muse-glimmer` | yes | yes | yes | yes | **no** | fails 3 |
| `...Nemotron-3.5-Lightning-30B-A3B-UD-Q4_K_XL.gguf` | 25,505,724,480 | `nemotron_h_moe` | **no** (`Q5_0`/`Q5_1`/`Q8_0` only) | **no**, 128 experts | yes | **no** (arm OWED; and `Q5_1` is id 7) | yes | fails 1, 2, 4 |
| `...Nemotron-3.5-Lightning-30B-A3B-Q8_0.gguf` | 35,004,643,392 | `nemotron_h_moe` | **no** (`Q8_0` only) | **no**, 128 experts | yes | **no** (arm OWED) | yes | fails 1, 2, 4 |
| `Laguna-S-2.1-UD-Q4_K_XL` (3 shards) | 73,395,172,000 | `laguna` | yes | **no**, 256 experts | **no** | see below | yes | fails 4, 5 |
| `GLM-5.3-Flash-UD-Q2_K_XL` (4 shards) | 108,720,071,427 | `glm5next` | mixed | **no**, 288 experts | **no** | yes | **no** | fails 3, 4, 5 |
| `GLM-5.3-UD-IQ1_S` (6 shards) | 216,715,365,893 | `glm-dsa` | mixed | **no**, 256 experts | **no** | yes | yes | fails 4, 5 |
| `Qwen3.8-Flash-Next-UD-*` (6 quant families) | 72,546,461,344 -- 111,334,654,784 | `qwen4exp` | mixed | **no**, 512 experts | **no** | yes | **no** | fails 3, 4, 5 |
| `Qwen3.8-2.4T-A95B-UD-Q1_0` (10 shards) | 397,256,393,248 | `qwen35moe` | mixed | **no**, 512 experts | **no** | yes | yes | fails 4, 5 |
| `Qwen3.8-2.4T-A95B-UD-IQ1_S` | 49,305,286,176 in 3 of 12 shards | `qwen35moe` | mixed | **no**, 512 experts | **no** | yes | yes | fails 4, 5, and the download never finished |

**Two rows carry a nuance the column cannot hold.** `laguna` has no entry in
`kGgufArchArms`, so a GGUF-only load refuses it on architecture -- yet
`LoadLagunaFromGguf` exists and `laguna_registry.cpp` reaches it with an
`HfConfig` supplied from outside the GGUF, so "our engine cannot load it" would
be too strong. It is not resolved here because it cannot move the verdict: that
artifact is MoE and is 73.4 GB against a 62.8 GB free carve, and either of those
alone disqualifies it. `nemotron_h_moe` is the reverse case: the pinned vLLM DOES
register `NemotronHForCausalLM`, and it is our side that refuses, by the named
refusal `HfConfigFromGgufDispatch` routes to `NemotronHGgufRefusal()` -- and the
artifact stores no k-quant tensor in any case.

**The intersection is empty.** The only artifacts satisfying conditions 1
through 5 are the two Qwen3.8-27B Q4_K files, which are the arm's own model and
fail condition 6.

## 4. A record this measurement falsified

`docs/bench-evidence/rocm-strix-qwen38-q4km-20260901.md` records
`Qwen3.8-27B-UD-Q4_K_M.gguf` as "Loadable by us: **no**", over #2510. **That is
no longer true of this tree.** `3fede4162` landed the IQ3_S reader, so ggml type
id 21 is now accepted, and the artifact's histogram
(`F32` 360, `IQ3_S` 4, `IQ4_NL` 7, `IQ4_XS` 117, `Q3_K` 7, `Q4_K` 104, `Q5_K`
131, `Q6_K` 30, `Q8_0` 106) contains no id our reader refuses.

It is reported here because it moved *against* the finding: it opens an artifact,
and the artifact still does not help, because it is the same model. That is the
whole of what it changes. It is recorded so the next reader does not carry a
stale refusal forward, and it is why this spec re-measured the loader from the
tree instead of citing #2510's summary. #2510 is already CLOSED, so nothing is
owed by this paragraph beyond the correction itself.

**The scope of that claim is the READER, and it is narrowed on purpose.** What
was measured is that every ggml type id the artifact stores has a case in
`FindGgmlTraits`. `3fede4162` landed a CPU row decoder and a CUDA gather codec;
this spec did not load the file on `gfx1151` and does not claim the whole ROCm
path admits it. The distinction cannot move the verdict, because the artifact
fails condition 6 whatever the device path does.

## 5. The near miss, and why it is decisive

`muse-glimmer-30B-kquant-17gb.gguf` is **exactly** the shape limb 3 asks for. It
is dense (its header declares no `expert_count`), it is bigger than the arm
(`block_count` 52, `embedding_length` 6656, `feed_forward_length` 19968, against
the arm's 65 / 5120 / 17408), its tensors are pure k-quant (`Q4_K` 365, `Q6_K`
52, `Q5_K` 1, `F32` 313), it is 16.8 GB against a 62.8 GB free carve, and our
GGUF dispatch has an arm for it.

It fails on one condition, and it fails it completely: **no pinned oracle can
load it.** The scan probes four surfaces rather than vLLM's registry alone,
because both vLLM and the plugin have Transformers fallbacks and a registry miss
would not settle it on its own:

```
vllm_pin_glimmer_files   = 0
vllm_omni_glimmer_files  = 0
llama_cpp_glimmer_files  = 0
gguf_py_glimmer_files    = 0
gguf_py_qwen35_files     = 2  <- the positive control
```

The last two lines are what settles it. `vllm-gguf-plugin` declares `gguf` as a
dependency and that package is llama.cpp's own `gguf-py`, which carries no
mapping for the `muse-glimmer` architecture string the file declares, so the
plugin cannot build a config for it before any model class is looked up. Its own
`_ADAPTER_REGISTRY`, read from the staged archive
`9e15c20e0b75f75bbf886966df07843c4b70a7952fad4b80e8e8183e2f70743b`, is
`Gemma3`, `Gemma4`, `OLMoE`, `Qwen35`, `Qwen35Mtp`.

`muse_glimmer_mm.cpp`'s own header already says this: every anchor it cites is
vllm#51655 head `075d645af`, an open CI-red upstream PR, and "the pinned oracle
cannot load this model". The measurement above is that statement made
falsifiable, extended to vLLM-Omni and to llama.cpp, and carrying a positive
control so a silent grep failure cannot read as an absence.

## 6. What this does NOT do

It ratifies nothing, weakens nothing and rescores nothing. It does not touch the
verdicts of #2497, #2534, #2546, #2740, #2809 or #2854/#2859. `TOKEN_GATE` stays
declared against llama.cpp `b10451` and stays FAIL. The row stays `PARTIAL`, no
matrix row moves and no keyed record is touched. No file under `src/`, `include/`
or `tests/` changes and no build ran.

It does not claim limb 3 is unsatisfiable in principle. It claims that on the
artifacts this fleet holds today, no vehicle exists, and §7 says what one would
be.

## 7. What a vehicle would need

Any one of these, in decreasing order of how little it costs:

1. **An oracle that reaches MuseGlimmer.** The artifact, the loader arm and the
   board all already exist. What is missing is a pinned upstream that can run
   it. `muse_glimmer_mm.cpp`'s own header records vllm#51655 as an OPEN, CI-red
   upstream PR, so pinning it would need its own file under
   [`../oracles/`](../oracles/) and a gateability measurement under `AGENTS.md`
   §"When vLLM has no implementation" -- the bar being that it demonstrably
   builds and runs the model, because constructing a config proves nothing. This
   is the cheapest path by a wide margin, and it is a decision rather than a
   download.
2. **A k-quant GGUF of a DIFFERENT dense model in one of the five families both
   sides register**, at or above 27B and under about 55 GB. The five are
   `deepseek4`, `qwen35`, `qwen35moe`, `qwen3next` and `glm-dsa`. Every family
   in that list except `qwen35` is MoE in the checkpoints this fleet holds, so
   in practice this means a second dense `qwen35` checkpoint that is not
   Qwen3.8-27B. Whether one exists was not established here, because fetching it
   needs recorded authority and none was given.
3. **A `llama` or `qwen3` GGUF arm in our dispatch.** That would make the fleet's
   existing `qwen3-4b-bf16` and `llama32-1b-instruct-bf16` convertible into
   vehicles. Both are far *smaller* than the arm, so a pass there would not
   satisfy the limb as written, which says BIGGER. An argument that a smaller
   model is the harder case, and therefore a fortiori evidence, is available but
   is NOT in the committed record: `multimodal-speed.md` §12.2 does not make it,
   and this spec does not make it either. It would need its own ratification.

Option 3 is listed to be complete and is not recommended: it changes product
code to serve a gate, which is the wrong order.

## 8. Design

No design. This is a search with a pre-registered predicate, an evidence
directory and a verdict.

## 9. Risks

**The scan reads three checkouts outside this repository** (`~/_git/vllm`,
`~/_git/llama.cpp`, `~/_git/vllm-omni`) and one CIFS share. It asserts the vLLM
checkout is the pin before reading any answer from it, and it carries a positive
control for the grep that produces the decisive negative. It does not assert the
llama.cpp checkout's revision, so the `gguf-py` result is scoped to "the working
checkout", and the positive control is what keeps that honest: a `gguf-py`
missing entirely would print 0 for both lines and be visible.

**A grep over a registry can miss an alias.** §3.2 greps ten family names
rather than one, and the `muse-glimmer` answer is corroborated by a whole-tree
`grep -rli glimmer` on three upstreams rather than by `registry.py` alone.

## 10. Tests

None. No product code changes, so there is nothing to gate that a test could
reach. The re-runnable artifact is `vehicle_scan.sh`, whose output is committed
verbatim beside it.

## 11. Gates

Run by name on this head, each exit code read from the process:

- `scripts/check-agent-record.py`
- `scripts/check-commit-style.py --range origin/main..HEAD`
- `scripts/check-commit-trailers.py --range origin/main..HEAD`
- `scripts/check-pr-size.py --base origin/main --head HEAD`, which
  `agent-preflight.sh` skips because it supplies no `--base`/`--head`

## 12. Stop conditions

Reached. The output contract admitted "no admissible vehicle exists on this
fleet" as a complete answer, and that is the answer.
