# Gemma 3 projection dtype

Row: `MODEL-GEMMA3-HEAD-DTYPE`.
Parent: `MODEL-TEXT-gemma3-gemma3-for-causal-lm`.
Issue: `ISSUE-LOCAL-01M2HPT4SGDCYZNAYH4VKBZJK4`.
Base: `39369747221a53daa6fe65c4b8991bf3f629f909`.
Integration: one pull request with the attention campaign, following repository policy.

## Now

`ACTIVE`. The broader attention workload exposes close-token differences.
Source inspection identifies a concrete model-format mismatch in the final
projection. The corrected projection passes 1510 focused assertions. Both
tied and untied heads fail the new dtype assertion before correction. The user authorized correctness repairs and prohibits subagents.
Independent human review remains due. This prerequisite has its own worktree.

The running primary confirms BF16 hidden states, weights, and projection
output. The combined original 96-token and expanded 256-token gates pass
with block sizes 16 and 32 in both prefill controls. Graph and eager decode
produce the same primary tokens. The attention row owns the combined final
gates. The row remains ACTIVE until the reviewed change lands.
[Measured report](../../docs/bench-evidence/rocm-rdna3-attention-wmma/README.md).

## Scope and design

Gemma 3 resolves BF16 activations and weights, but its final projection writes
FP32 directly. Project into BF16, then widen to FP32 at the existing
`ForwardLogits` boundary. This preserves the public float output contract and
the sampler input format. Both tied and untied heads use the same rule.
Do not modify attention arithmetic, sampling, token gates, or other models.

## Primary sources

vLLM `e126687a9a828d513c01a07cd69f025f27d63280`:
`model_executor/models/gemma3.py::compute_logits` calls `LogitsProcessor`.
`model_executor/layers/logits_processor.py::_apply_head` uses model dtype by
default. `UnquantizedEmbeddingMethod.apply` calls `dispatch_unquantized_gemm`.
On ROCm, `layers/utils.py:260-346` selects the skinny kernel or Torch GEMM.
`v1/sample/sampler.py:97` widens the already rounded logits to FP32.
The running primary BF16 model reproduces its generated IDs across two runs.

## Tests and gates

1. Add a synthetic model-forward assertion that every returned float is an
   exact widening of a BF16 value. Exercise tied and untied heads. Capture red
   before changing the projection.
2. Run the complete Gemma forward suite and the attention campaign's focused
   regression suites. Remove the projection rounding in a scratch copy and
   prove that the dtype assertion fails.
3. Rerun all original 96 public tokens and all 256 distinct-prompt benchmark
   tokens against the pinned primary. Keep every prior failed receipt.
4. Trace both engines with rocprofv3 and record the output dtype source chain.
   Accept performance only after token correctness. Run full preflight.

## Risks and stop conditions

The correction can change close-token decisions on existing Gemma checkpoints.
Do not preserve an incorrect FP32 projection solely to retain old outputs.
The existing checkpoint-dependent 1B gate needs its artifact to execute.
Unresolved token mismatches remain failures. Never widen a tolerance or select
only passing prompts to accept a performance result. Do not merge without
independent review.

## Evidence

The attention row owns the combined checkpoint outputs and performance report.
This worktree retains its focused red, green, and mutation receipts.

## Row inventory

| ID | Upstream source | Local anchor | Tests and evidence | Spec | State | Owner | Issue |
|---|---|---|---|---|---|---|---|
| `MODEL-GEMMA3-HEAD-DTYPE` | vLLM logits processor and sampler at e126687a9 | `Gemma3Model::Forward` | Gates above | [This spec](gemma3-head-dtype.md) | `ACTIVE` | Codex, single-agent user direction | `ISSUE-LOCAL-01M2HPT4SGDCYZNAYH4VKBZJK4` |
