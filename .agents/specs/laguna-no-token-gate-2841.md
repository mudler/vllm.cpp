# Laguna — the row has no oracle token comparison, and nothing on this host can produce one

**Row:** `MODEL-TEXT-LAGUNA-NO-TOKEN-GATE-2841` — new row, `ACTIVE`.
**Issue:** [#2841](https://github.com/mudler/vllm.cpp/issues/2841).
**Date:** 2026-09-04. **Base:** `3f12c617d`.
**Predecessor row:** `MODEL-TEXT-laguna-laguna-for-causal-lm`.

## Why this spec exists rather than a gate

#2841 asks for a committed Laguna golden and a token gate against it. Neither can
be produced from this host, and the spec says so in the record rather than
leaving a reader to rediscover it. `CLAUDE.md` §"Every change starts from an
issue" requires an issue not fixed in the flow to name an owning row or be listed
under a spec's `## Owed`; O1 below is that listing. **This row does not close
#2841.**

## What is true at `3f12c617d`, re-measured rather than quoted

- `tests/parity/goldens/` holds **108** directories. **23** of them carry a
  `greedy_ids.npy`. **0** of them are Laguna.
- `tests/vllm/models/` carries four Laguna files —
  `test_laguna_fused_gate_up.cpp`, `test_laguna_gguf_load.cpp`,
  `test_laguna_nvfp4_loader.cpp`, `test_laguna_scaffold.cpp`. **None** mentions
  `greedy_ids` or `neartie`. There is no token or distributional comparison
  against any oracle anywhere on this row.
- `docs/USAGE.md` carries **no checkpoint row** for `LagunaForCausalLM`: no file
  name, no size, no HuggingFace repo, no revision, no sha256. Its one mention of
  `laguna` is an unrelated sentence about forwards finishing on the host.
  `CLAUDE.md` §"Say which weights, and from where" requires that row.
- The only Laguna artifact present on this host is
  `~/.cache/huggingface/hub/models--mudler--Laguna-XS-2.1-APEX-GGUF`.
  `poolside/Laguna-S-2.1-NVFP4` is absent from the box and from the NAS.

## What the record already says, and where it is right

`docs/FEATURES.md` was corrected at `d74cb3967` and the Laguna row now reads
"**no golden, no token gate**", naming #2841. That is accurate and this row does
not touch it.

`.agents/model-matrix.md:128` is also accurate about what was measured: "vs the
vLLM MARLIN golden (vLLM's exact prompt ids injected): **FIRST 2 TOKENS MATCH
exactly**, then near-tie divergence", and it already calls `VT_LAGUNA_KV_BF16`
"a distributional near-tie left **UNRATIFIED**". The overclaim #2841 found lived
in `docs/FEATURES.md` alone, and it is gone.

## What this row deliberately does NOT change

**The 87.2% and 86.0% figures at `.agents/benchmark-record.md:2027`, `:2045`,
`:2084` and `:2090` stay.** #2841 flags them as superseded figures "which a grep
hits first". They are dated evidence entries, correct as of their measurement,
and superseded later in the same file. `CLAUDE.md` §Records: "Never delete
evidence to reduce context." Editing them would rewrite evidence to improve grep
ergonomics, which is the wrong trade.

## NEEDS_DECISION for the developer

`.agents/model-matrix.md:128` marks `LagunaForCausalLM` **✅** while the row has
no oracle token comparison at all. Every other ✅ text row in that table carries
one. Whether that is a lifecycle state to move, or a ✅ that means something
narrower for this row, is a developer call and not a helper's, so it is raised
here and not changed unilaterally.

## Stop conditions

Do not manufacture a Laguna golden from any artifact this host has. The GGUF
present here is `Laguna-XS-2.1-APEX-GGUF`, which vLLM cannot load, so nothing on
this host can produce an oracle side. A self-consistency A/B against our own
prior output is what #2841 already identifies as the thing being mistaken for a
gate; producing another one would repeat the defect.

## Owed

- O1. **A committed `tests/parity/goldens/laguna_*_greedy/` pair and a token gate
  against it**, or a ratified distributional gate with the non-determinism
  measurement that licenses it. Tracked by
  [#2841](https://github.com/mudler/vllm.cpp/issues/2841), which stays OPEN.
  Cost: the `poolside/Laguna-S-2.1-NVFP4` checkpoint, which is not on this host
  or the NAS, plus a fleet GPU lease to run the pinned vLLM oracle.
- O2. A `docs/USAGE.md` checkpoint row for `LagunaForCausalLM` — file name, size,
  repo AND revision, sha256 for the quantized artifact, refused arms named. It
  cannot be written from a host that does not have the artifact, because a repo
  id alone is not a pin.
- O3. The `✅` question above, pending the developer's answer.
- O4. The fixture half of this family is separately owned:
  [#2834](https://github.com/mudler/vllm.cpp/issues/2834) repairs the only
  forward-runnable Laguna fixture, which was constant. A discriminating synthetic
  fixture is still not an oracle, so O1 is not weakened by it.
