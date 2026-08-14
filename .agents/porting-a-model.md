# Task guide — porting a model

The per-model checklist. [`porting.md`](porting.md) is the *method* for any port
(enumerate the surface, port the test first, capture the red, hand to a fresh
reviewer); this is the *coverage list* for a model specifically, so what a port
includes stops depending on who remembers what.

The rules are in [`AGENTS.md`](../AGENTS.md). Nothing here is binding on its own,
and nothing here may weaken a rule there.

Work top to bottom. Every line is either done, deliberately deferred with the
reason recorded in the row's spec, or not applicable to this architecture — and
"nobody thought of it" is none of those three.

## What "complete" looks like — the Qwen reference

Qwen3.5/3.6 is the most complete port in this tree; use its shape as the target.
Its translation units are the checklist in physical form:

```
qwen3_5_common.cpp        shared pieces
qwen3_5_dense.cpp         the dense arm
qwen3_5_moe.cpp           the MoE arm
qwen3_5_weights.cpp       safetensors loading
qwen3_5_gguf_weights.cpp  GGUF k-quant loading      <-- a SEPARATE TU, always
qwen3_5_mtp.cpp           the speculative drafter
```

Note what that layout tells you: **GGUF is its own translation unit, not an
afterthought bolted onto the safetensors loader**, and the drafter is part of the
model port rather than a later project. A port that produced only
`<model>.cpp` + `<model>_weights.cpp` has almost certainly skipped the quantized
arm and the drafter.

Across the family, a complete port has covered: dense and MoE variants, GGUF
k-quants, NVFP4 (W4A4 and W4A16), MXFP4 compressed-tensors, FP8, image, video and
audio, MRoPE and DeepStack on the vision side, sliding-window attention, MTP and
DFlash speculative decoding, and the tool/reasoning parsers. Not every
architecture has every one — but each is a line you consciously tick or record as
not applicable.

## 0. Before claiming

- [ ] An open issue tracks the work, linked from the append-only issue index, the
      row's spec, and the PR body.
- [ ] The gap is re-verified against the *current* pin and local head, not
      against a spec's description of them.
- [ ] The upstream anchor is recorded as an exact `file:line` at an exact
      revision. If the model does not exist at the pin, that is an exception:
      record it in [`porting-inventory.md`](porting-inventory.md) §9 and argue
      for it in the commit that introduces it.
- [ ] The spec is committed **before** implementation. At row claim, ask the
      developer whether the spec and implementation use one pull request or
      separate pull requests. Recommend one pull request and record the answer.

## 1. Config

- [ ] Every field the forward consumes is parsed from the **released**
      checkpoint's `config.json`, not only from the upstream Python defaults.
      Read the real file; publishers ship field names the reference code does
      not mention.
- [ ] Tri-state fields are handled: a key that is **absent** is not the same as
      `false`. Upstream frequently means "on" by omission.
- [ ] Alternate/legacy config layouts are normalized, and a fixture pins each.
      A layout that silently deserializes to all-defaults is a wrong-shaped
      model with no error.
- [ ] A test parses the **real released config values** as a committed fixture,
      so CI catches drift without needing the checkpoint.

## 2. Weight formats — all of them

**A model port covers the quantized arms, not just bf16.** This is a rule, not a
preference: the quantized arm is what most users can actually run (a 30B bf16
checkpoint is ~60 GB against a ~17 GB k-quant), and it is what a quant-matched
llama.cpp comparison needs.

- [ ] Safetensors bf16/fp16.
- [ ] **GGUF k-quants and i-quants**, through the shared GGUF loader.
- [ ] The quantized arms the checkpoint family actually ships — fp8, NVFP4
      (W4A4 and W4A16), MXFP4 compressed-tensors, AWQ/GPTQ — each either loaded
      or explicitly refused with a message naming the missing piece.
- [ ] Weights are enumerated by the names the checkpoint **ships on disk**, not
      by the upstream *module* names. Upstream fuses q/k/v at load time via its
      stacked-params mapping; the on-disk tensors are usually separate, often
      with biases the module-level view hides.
- [ ] A structural gate accounts for **every** tensor in the real checkpoint
      index — enumerated == present, zero unaccounted — reading headers only,
      env-gated so CI never needs the asset.

## 3. Forward

- [ ] Routed through the shared seams: `ModelRegistry::Forward`,
      `dense_attn::AttnBlock`, `vt::FusedChain`, `layers::MlpGateUpMethodBase`.
      If a seam cannot express the architecture, extend it or record one exact
      tracked exception; never hand-roll a parallel path.
- [ ] Each novel mechanism has its own RED-first test, and each is
      **mutation-proven**: break it, watch that gate go red, restore the tree
      byte-for-byte, re-run green. A mutation that stays GREEN is a coverage
      hole, not a pass.
- [ ] Numeric bounds, not just token equality. A mechanism can be missing while
      the argmax is unchanged — that has happened here — so a tokens-only gate
      can pass a model that dropped one.

## 4. Correctness gates

- [ ] Token-exact against the pinned oracle on identical artifacts, prompts,
      token counts, batching and sampling; or the ratified distributional gate
      where the oracle's own greedy decode is non-deterministic.
- [ ] If the pinned oracle cannot load the model, say so plainly and gate
      against the reference implementation instead — and do not call that
      token-exact against the runtime, because it is not.
- [ ] Upstream's tests ported in the same change, preserving parameters, modes,
      fixtures, tolerances and failure cases, with the revision anchor.
- [ ] Run on the **real checkpoint**, not only synthetic fixtures.

## 5. Capability surface

Each applicable, or recorded not-applicable:

- [ ] Multimodal: image, video, audio — processor, encoder, and the merge into
      the text stream, each gated.
- [ ] Speculative decoding, if the family ships a drafter.
- [ ] Reasoning parser and tool parser, with upstream's parser tests ported.
- [ ] Chat template and tokenizer specifics.
- [ ] The capability is reachable through `include/vllm.h`; examples and servers
      are thin ABI clients.

## 6. Speed

- [ ] vLLM, quant-matched, in its **production** configuration, is the bar —
      never `--enforce-eager`. llama.cpp is a labeled secondary bar.
- [ ] Correctness first: no performance result is accepted before the declared
      correctness gate passes.
- [ ] Every required axis recorded with values and ratios; any axis below floor
      is an open gap, and no ceiling is ever declared.
- [ ] If no denominator exists yet, the axis is an **open gap by construction** —
      not a waiver, and not silence.

## 7. Records, in the same change

- [ ] Row + checklist entry + rollup in the owning matrix.
- [ ] `issue-index.md` row appended, `coordination.md` claim.
- [ ] `docs/STATUS.md`, `docs/BENCHMARKS.md`, `.agents/NOW.md` on a lifecycle
      change; `docs/FEATURES.md` / `docs/USAGE.md` / `README.md` when the
      user-visible surface moves.
- [ ] The spec's `## Outcome`: what was measured, what was rejected and why, and
      why each default is what it is.
- [ ] A fresh reviewer — never the agent that wrote the code — has reviewed the
      immutable head and mutated its claimed guarantees.
