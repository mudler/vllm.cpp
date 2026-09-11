# Oracle: llama.cpp release `b10766`, the first release that runs DeepSeek-V4 vision

A scoped, second llama.cpp record. It exists for one reason: `b10766` is the
first stock llama.cpp release that converts, loads and runs the vision variant of
`deepseek4`, and the row
`MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm` needs a runnable reference and a
quant-matched denominator on the artifact users actually download. It does not
replace the [`llama-cpp`](llama-cpp.md) oracle, it does not outrank vLLM, and it
is never a mirror source.

## Why the `llama-cpp` file cannot carry this pin

`scripts/check-oracle-pins.py` admits exactly one ` ```oracle-pin ` block per
file, so one file holds one revision. The `llama-cpp` pin is deliberately
release `b10451`, and every floor already measured against it means "what a user
gets from that release". Advancing that pin to reach DeepSeek-V4 vision would
silently redefine what those recorded numbers measured, 315 commits after the
fact.

Unlike [`llama-cpp-qwen4exp`](llama-cpp-qwen4exp.md) and
[`llama-cpp-glm5next`](llama-cpp-glm5next.md), this record does **not** pin an
unmerged pull-request head. The support is merged and released. The two records
therefore say two different true things about two different releases, and this
one may be retired the moment `llama-cpp` advances past `b10766`.

## Scope, and what this oracle may not do

Use it ONLY for the `deepseek4` vision variant: the `deepseek4v` clip projector
and its mmproj container, the DeepSeek-V4 Flash Vision image preprocessor, the
`exp_probs_b_vl` routing bias for media batches, the non-causal image-span
window behaviour, and the CPU and GGUF k-quant speed and memory numbers those
produce on a DeepSeek-V4 Flash Vision GGUF.

`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` at
`86f746b36186f0e567729a5c06a8c918caba82a9` remains the algorithm oracle for
every behaviour both implement; see [`deepseek-v4-vision`](deepseek-v4-vision.md).
Where llama.cpp and the model author disagree, the model author wins, and where
vLLM implements the behaviour at all, vLLM wins, exactly as `AGENTS.md`
§"When vLLM has no implementation" requires. llama.cpp's structure is not vLLM's;
a difference in structure is never a reason to diverge.

Two llama.cpp approximations are known and must not be copied without checking
the model author first:

1. `src/models/deepseek4.cpp` selects `exp_probs_b_vl` for the whole ubatch when
   `ubatch.embd != nullptr`, not per token. A mixed text/image ubatch therefore
   routes its text rows on the vision bias.
2. The same branch skips hash-layer `tid2eid` routing entirely for a media
   ubatch, because the converter drops `ffn.gate.bias` on hash layers.

## The pin, verified rather than relayed

Read on 5 September 2026 from `ggml-org/llama.cpp` refs and objects through the
GitHub API. No local llama.cpp working tree was read.

| Claim | Query | Result |
|---|---|---|
| the tag resolves to a commit | `git/ref/tags/b10766` | `commit 9400c8946e4da5e7694f2c26d6d4e50e14b690fa` |
| that commit is the merge of the vision PR | `commits/b10766` | `model: correctly support input vision for deepseek4 (#28154)`, 2026-09-02T17:14:46Z |
| the projector exists at the pin | `contents/tools/mtmd/models/deepseek4v.cpp?ref=b10766` | blob `ffe8f59d99977c0b556edb5908427dcbae9cf290`, 4100 bytes |
| it does NOT exist at the stock pin | `contents/tools/mtmd/models/deepseek4v.cpp?ref=b10451` | HTTP 404 |
| the stock pin is the one `llama-cpp` records | `commits/b10451` | `10bf611e533d81f739128304991c5e133c6aebd8` |
| this pin descends from the stock pin | `compare/b10451...b10766` | `ahead`, `ahead_by=315` |
| `b10766` is the FIRST release with it | `compare/9400c894...bNNNNN` over `b10762`-`b10767` | `b10762`-`b10764` `behind`; `b10766` `identical`; `b10767` `ahead` |

## Gateability

`gateable = yes`, measured on 11 September 2026 on `thor:gpu0` through
resource-controller. `AGENTS.md` admits `gateable = yes` only after an oracle
demonstrably builds and runs the model, and this one did both.

- **It builds.** rc job `b69b2fb9-23b9-42b8-b755-62b8ee93b6ea` cloned
  `ggml-org/llama.cpp` inside the job, checked out
  `9400c8946e4da5e7694f2c26d6d4e50e14b690fa`, asserted `rev-parse HEAD`
  against it (`git describe`: `b10766`), and built CPU-only and static on
  aarch64: `cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
  -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF
  -DLLAMA_BUILD_SERVER=OFF`, targets `llama-mtmd-cli` and the W6 driver
  `dsv4v-oracle-dump`, `-j 4`. The recipe is
  `tools/parity/dsv4v_w6_parity.sh`.
- **It runs the model.** In the same job, `llama-mtmd-cli` loaded the pinned
  `UD-IQ1_S` shards and `mmproj-BF16.gguf` together, encoded a 392x392 image,
  and generated a description that fits it.
- **It is deterministic on the CPU.** A second job,
  `2481ad2a-c109-4002-8ee6-13634a2bd7f5`, rebuilt it and reproduced the image
  block byte for byte. The W6 driver's block is byte-identical to
  `llama-mtmd-cli`'s own `MTMD_DEBUG_EMBEDDINGS` dump.

What it gates and what it does not: the `deepseek4v` tower and token block, on
the CPU provider. Flash attention was ENABLED in every run
(`warmup: flash attention is enabled`), so its attention takes F16 K and V, and
its GEMMs take bf16 inputs. Those are this oracle's own precision, and the row's
spec measures how far they move it. The raw dumps, logs and reports are on the
NAS under `/workspace/dsv4-vision/w6-parity/`, and the committed record is the
spec's `### W6 evidence` section, which the `evidence` field names.

```oracle-pin
id = llama-cpp-dsv4vision
role = secondary
upstream = https://github.com/ggml-org/llama.cpp
scope = the deepseek4 vision variant only: the deepseek4v clip projector and mmproj container, the DeepSeek-V4 Flash Vision image preprocessor, the exp_probs_b_vl media routing bias, the non-causal image-span window, and the GGUF k-quant floor those produce
pin = 9400c8946e4da5e7694f2c26d6d4e50e14b690fa
pin_label = release b10766
pinned_on = 2026-09-05
gateable = yes
evidence = .agents/specs/deepseek-v4-flash-vision.md
```
