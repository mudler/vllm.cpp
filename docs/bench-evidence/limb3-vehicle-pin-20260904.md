# A limb-3 vehicle exists after all, and this is the pin

Issue [#2884](https://github.com/mudler/vllm.cpp/issues/2884), row
`QUANT-QWEN38-27B-GGUF-ARM`, spec
[`limb3-vehicle-strict-gate.md`](../../.agents/specs/limb3-vehicle-strict-gate.md).
Predecessor:
[`q4km-limb3-kquant-vehicle.md`](../../.agents/specs/q4km-limb3-kquant-vehicle.md)
(#2864), whose verdict was **NO VEHICLE ON THIS FLEET** and whose §7 said what
one would have to be.

**No model is run in this document and no number in it is a performance
result.** `AGENTS.md` §Gates admits no performance result from an arm whose
declared token gate has not passed, and
[#2497](https://github.com/mudler/vllm.cpp/issues/2497) already carries one
retraction for exactly that.

## What changed since #2864

Nothing about the fleet's artifacts. What changed is that the developer
recorded authority to fetch one, which #2864 explicitly did not have:

> Whether one exists was not established here, because fetching it needs
> recorded authority and none was given.

The grant is in `.agents/developer-preferences.md` (2026-09-04) and is scoped
by the six conditions #2864 pre-registered. This document spends it on one
artifact and says why that artifact and no other.

## The candidate, derived rather than chosen

#2864 §7 option 2 already narrowed the space to "a second dense `qwen35`
checkpoint that is not Qwen3.8-27B". Two exist: `Qwen/Qwen3.5-27B` and
`Qwen/Qwen3.6-27B`. Both declare `architectures: ["Qwen3_5ForConditionalGeneration"]`
and `model_type: qwen3_5`, both are 64 layers at hidden 5120 / FFN 17408 /
vocab 248320 / head_dim 256, and neither carries an `expert` key.

`Qwen3.6-27B` is taken, and BOTH reasons are measured rather than editorial.

**The decisive one is the histogram**, read from each candidate's own bytes by
range request before either was fetched:

| candidate | revision | tensors | type histogram |
|---|---|---:|---|
| `unsloth/Qwen3.6-27B-GGUF` `Qwen3.6-27B-Q4_K_M.gguf` | `82d411acf` | 851 | `F32` 449, `Q4_K` 289, `Q5_K` 48, `Q6_K` 65 |
| `unsloth/Qwen3.5-27B-GGUF` `Qwen3.5-27B-Q4_K_M.gguf` | `3221f178a` | 851 | `F32` 353, `Q4_K` 263, `Q5_K` 96, `Q6_K` 43, **`Q8_0` 96** |

Both files carry the `Q4_K_M` label and they are not the same artifact under it.
The Qwen3.5 file routes 96 tensors through the `Q8_0` path, which is a DIFFERENT
kernel from the three k-quant kernels limb 3 exists to exercise. The Qwen3.6
file has no non-k-quant weight tier at all. This is #2864's own discipline
applied to the choice itself: the predicate is evaluated against the histogram,
never against the name, and the name would have said the two files were
interchangeable.

**The second reason is the config surface.** `Qwen3.6-27B`'s `text_config` key
set is **identical** to the arm's, `output_gate_type`, `partial_rotary_factor`
and `tie_word_embeddings` included. `Qwen3.5-27B` carries none of those three.
The closer the config, the more of the same forward code both engines execute.

**No dense `qwen35` checkpoint larger than 27B exists.** The family's larger
members (`35B-A3B`, `122B-A10B`, `397B-A17B`) are all MoE, which condition 4
excludes. §"The shortfall" below states plainly what that costs.

## The pin

| | |
|---|---|
| repo | `unsloth/Qwen3.6-27B-GGUF` |
| revision | `82d411acf4a06cfb8d9b073a5211bf410bfc29bf` |
| file | `Qwen3.6-27B-Q4_K_M.gguf` |
| bytes | 16,817,244,384 |
| sha256 | `5ed60d0af4650a854b1755bd392f9aef4872643dc25a254bc68043fa638392a0` |
| vision tower | `mmproj-BF16.gguf`, same revision, 931,146,304 bytes, sha256 `05353347512982ee62317b9d8c89372bc815f4b4043580e7ef3ad411ec1a1cd3` |
| tokenizer and config | `Qwen/Qwen3.6-27B` @ `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` |

**A repo id is not a pin, and this campaign has already been bitten by
that.** #2497 refused the UD family because `Qwen3.8-27B-UD-Q4_K_XL`'s published
bytes moved in place under an unchanged name (17,923,394,624 -> 17,559,178,144
B). The revision is therefore recorded, the size is recorded, and the
authoritative digest is the one measured on the staged bytes rather than the
one the forge advertises. The plain `Q4_K_M` arm is taken rather than a `UD-*`
one for the same reason, and because it is the quant tier the arm itself runs.

The vision tower is fetched with the model because the ecosystem ships this
family's tower as a separate `general.architecture = clip` file and #2740's own
working recipe passes one; every modality is at limit 0 in the gate, so the
tower is loaded and never entered.

## The six conditions, measured on the artifact's OWN bytes

Read **by HTTP range request** before the download, which is this repository's
practice for a checkpoint manifest, and read with #2864's committed
`gguf_header.py` rather than a second parser, so the two searches cannot
disagree about what a header says. `remote_gguf_header.py` is the range reader
and `vehicle_pin_check.sh` re-runs the whole determination.

```text
ARCHITECTURE       qwen35
GGUF_VERSION       3
N_TENSORS          851
SIZE_LABEL         27B
qwen35.block_count = 64
qwen35.embedding_length = 5120
qwen35.feed_forward_length = 17408
TYPE_HISTOGRAM     {"F32": 449, "Q4_K": 289, "Q5_K": 48, "Q6_K": 65}
```

The arm's own artifact, read by the same instrument, for the comparison the
limb turns on:

```text
ARCHITECTURE       qwen35
N_TENSORS          866
qwen35.block_count = 65
qwen35.embedding_length = 5120
qwen35.feed_forward_length = 17408
TYPE_HISTOGRAM     {"F32": 456, "Q4_K": 294, "Q5_K": 48, "Q6_K": 67, "Q8_0": 1}
```

| # | Condition | Verdict |
|---|---|---|
| 1 | stores k-quant tensors | **yes** -- 402 of 851 tensors are `Q4_K`/`Q5_K`/`Q6_K` and there is no non-k-quant weight tier at all, so the arm's `DotQ4K`, `KQuantGemmK`, `QuantizeQ8KK` and `MatmulBTQuantKernelRocm` are the kernels that serve it |
| 2 | an arm in `kGgufArchArms` | **yes** -- `qwen35` is the third entry, and all four ggml type ids the file stores (0, 12, 13, 14) have a case in `FindGgmlTraits`, so no tensor refuses the file whole |
| 3 | the pinned vLLM registers the family | **yes** -- `registry.py:572` at `5559679229`, one hit, and the four-surface check below |
| 4 | dense | **yes** -- the header declares no `expert_count`, and the HF config's `text_config` carries no key containing "expert" |
| 5 | fits the board | **yes** -- 16.04 GiB against 59934 MiB free before load |
| 6 | not the arm's own model | **yes** -- a different checkpoint of a different generation |

**The predicate is evaluated against the histogram, never against the file
name.** That is #2864's own discipline and it is kept here: the `Q4_K_M` label
would not have told anyone that this file stores no `Q8_0` tensor while the
arm's stores one, nor that it has 64 blocks where the arm has 65.

## The four-surface oracle check, with a NEGATIVE control this time

#2864 probed four surfaces for `muse-glimmer` because a registry miss would not
settle it on its own, and carried a **positive** control so a silent grep
failure could not read as an absence. Here the expected answer is positive, so
the control that matters is the **negative** one: `muse-glimmer` is re-probed
on the identical surfaces and must still read 0. A grep that matched everything
would light both rows and be visible.

```text
REGISTRY  Qwen3_5ForConditionalGeneration        hits=1
REGISTRY  MuseGlimmer                            hits=0

vllm_pin_qwen35_files      = 20
vllm_pin_glimmer_files     = 0    <- negative control
vllm_omni_qwen35_files     = 2
vllm_omni_glimmer_files    = 0    <- negative control
llama_cpp_qwen35_files     = 10
llama_cpp_glimmer_files    = 0    <- negative control
gguf_py_qwen35_files       = 2
gguf_py_glimmer_files      = 0    <- negative control
```

`VLLM_CHECKOUT_IS_THE_PIN = yes` is asserted before any answer is read from
that checkout. The llama.cpp checkout's revision is NOT asserted, exactly as in
#2864, so the `gguf-py` rows are scoped to "the working checkout"
(`237ad9b961f009ae19ac29dbce4cd0c1251f94b3`); the controls are what keep that
honest.

`vllm-gguf-plugin`'s `_ADAPTER_REGISTRY`, read from the staged source archive
`9e15c20e0b75f75bbf886966df07843c4b70a7952fad4b80e8e8183e2f70743b`, is
`Gemma3`, `Gemma4`, `OLMoE`, `Qwen35`, `Qwen35Mtp`. Its `Qwen35GGUFAdapter`
selects on `model_type`, and `QWEN35_MODEL_TYPES` is
`("qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text")`. The vehicle
declares `qwen3_5`, so the adapter maps its tensors by the same predicate it
maps the arm's by. This is what #2864 measured as absent for MuseGlimmer, and
it is present here.

## THE SHORTFALL, stated rather than argued away

The 2026-07-20 ratification's item 3 says a **BIGGER** dense model. This vehicle
is not bigger. It is 64 blocks against the arm's 65 at identical width, and the
arm's 65th block is its `nextn` drafter rather than a decoder layer, so the two
trunks are the same depth. The vehicle carries no `nextn_predict_layers` key at
all, which is convenient for a gate that runs MTP off on both sides and is
still not "bigger".

**No dense `qwen35` checkpoint larger than 27B is published**, so the strongest
vehicle available on this architecture is a same-class sibling. A pass here
satisfies the six conditions #2864 pre-registered. Whether that also satisfies
the word "bigger" is a ratification decision, and neither this document nor
`.agents/specs/limb3-vehicle-strict-gate.md` makes it.

## What this does NOT do

It ratifies nothing and rescores nothing. It does not touch the verdicts of
#2497, #2534, #2546, #2740, #2809, #2854 or #2864. `TOKEN_GATE` stays declared
against llama.cpp `b10451` and stays `FAIL`. The row stays `PARTIAL`, no matrix
row moves, and no file under `src/`, `include/` or `tests/` changes.
`STRIX_ARM_SPEED_RATIFIED_BY` stays unset and
[`../../.agents/scripts/rocm-strix-ourarm-staged.sh`](../../.agents/scripts/rocm-strix-ourarm-staged.sh)
stays refusing.

## Re-running it

```sh
bash docs/bench-evidence/limb3-vehicle-pin-20260904/vehicle_pin_check.sh
```

Its verbatim output is committed beside it as `pin-check.txt`. It needs network
and no GPU and no lease.
