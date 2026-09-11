ID: ISSUE-GH-387
Title: Mistral3ForConditionalGeneration (Mistral Small 3.1/3.2 24B): offering to contribute the port, asking how the oracle gate can be met by an outside contributor
Row: MODEL-MM
State: OPEN
Kind: UNKNOWN
GitHub: 387
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-11
Updated: 2026-08-11
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## What this is
>
> A **proposal to collaborate**, not a PR announcement. `Mistral3ForConditionalGeneration` (Mistral Small 3.1 / 3.2, 24B) is the model our deployment is built on, and we would like to contribute it. Taken from the checkpoint our production server actually loads -- `--model RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-NVFP4 --quantization compressed-tensors`, read off the running container's argv, with the fields below read from that snapshot's own `config.json` on the device rather than from the Hub: `architectures: ["Mistral3ForConditionalGeneration"]`, `vision_config` present, `quantization_config.quant_method: compressed-tensors`, `format: nvfp4-pack-quantized`. The blocking question is not the code -- it is that we cannot produce the oracle evidence your gate requires. Rather than write a large model bring-up and discover at review time that its gate can never go green from our side, we would rather agree the shape first.
>
> Your matrix already carries the row: `MODEL-MM-mistral3-mistral3-for-conditional-generation`, `registry.py:486-489`, `vllm/model_executor/models/mistral3.py`, currently **`INVENTORIED`**, spec `☐ required`, owner `unassigned`.
>
> ## Why this one is smaller than it looks
>
> Most of it has already landed, by your own records:
>
> - **The text decoder is done.** `MODEL-TEXT-mistral-mistral-for-causal-lm` is correctness-complete -- `tests/parity/test_mistral_paged_engine.cpp` passes the full SACRED gate 16/16 vs vLLM 0.25.0 (15 strict + 1 near-tie at 0.0000 nats), on the shared dense stack (`MistralWeights = Qwen3DenseWeights`). Mistral3's language half is that decoder.
> - **The pattern is proven end to end.** `VoxtralForConditionalGeneration` is the same shape with a different modality: a tower plus a projector merged into "the LANDED Mistral/Llama decoder", additive and gated so that Mistral text stays byte-identical. Mistral3 is that, with images instead of audio.
> - **The quantization loaders exist and are generic.** `LoadCtNvfp4W4A16` (`dense_weight_loaders.h:398`) already reads compressed-tensors NVFP4 W4A16 into a raw fp4-resident `Nvfp4Weight` via a `weight_packed` / `weight_scale` probe, with `qwen3_weights.cpp` as the wiring pattern. Our production checkpoint is exactly that scheme (`compressed-tensors` / `nvfp4-pack-quantized`), so this is per-arch wiring rather than a new quant path.
> - **Vision-tower infrastructure exists** -- `gemma4_vision.h`, `qwen3_vl_vision.h`, `muse_glimmer_vision.h`, plus the `scripts/mm/*_ref_dump.py` / `*_weight_dump.py` harnesses.
>
> The genuinely new work is the **Pixtral-class vision encoder** and the Mistral3 merge/projector. There is no tower implementation for it in the tree; the only occurrence of the name is the sibling row `MODEL-MM-pixtral-pixtral-for-conditional-generation` (`registry.py:520`), itself `INVENTORIED` and unowned. Those two rows presumably want to share one tower, which is a design question for you rather than something we would decide.
>
> ## What we would contribute
>
> Mirroring the structure Voxtral and Gemma-4 established:
>
> 1. `include/vllm/model_executor/models/mistral3_vision.h` -- the Pixtral-class tower, additive alongside the existing towers.
> 2. `include/vllm/model_executor/models/mistral3.h` + `src/.../mistral3.cpp` -- the projector and the masked-scatter merge into the existing dense decoder, gated on image input so the text path stays byte-identical.
> 3. `src/.../mistral3_weights.cpp` -- loader, including the CT-NVFP4 W4A16 wiring through the existing helpers.
> 4. `src/.../mistral3_registry.cpp` -- `REGISTER_VLLM_MODEL`.
> 5. Tests: loader assertions on real weights, a tower reference-dump comparison, a registry case, and the e2e gate below.
> 6. The `.agents/` records the change owes -- spec, model-matrix row transition, and the public-doc projections -- moved in the same change, per the lesson from #326.
>
> We have sm_110 (Jetson AGX Thor) silicon and would carry the bring-up, the numerics, and the on-device performance work.
>
> ## What we cannot produce, and what we would need from you
>
> **The oracle.** Your gate is "vs vLLM 0.25.0" from the pinned oracle on identical artifacts, and for the multimodal rows it is explicitly dgx-only (Voxtral's row records the gate as GPU-dgx-only, captured with `scripts/mm/a3_voxtral_oracle_capture.py`). We cannot run the pinned vLLM oracle on a Mistral3 checkpoint. Concretely we would need, from your side or with your blessing:
>
> - a per-prompt greedy oracle dump for a named Mistral3 checkpoint (the `a3_voxtral_oracle_capture.py` shape), image fixtures included, and
> - a ruling on whether the gate is STRICT or the ratified near-tie/distributional form for this model -- which, per your own rule, is decided **by measuring the oracle's determinism**, not chosen up front.
>
> **The real question, and the reason for this issue rather than a PR:** is there any path by which contributor-supplied hardware evidence can stand in for part of that? We can produce on-silicon numerics, memcheck, A/B, and full loader/tower reference comparisons on sm_110; what we cannot produce is the vLLM oracle side of a token-exact comparison. If the answer is "no -- the oracle half must come from the maintainer's box", that is a fine answer and it just means the work has to be sequenced around your availability. We would rather know that now.
>
> ## Process questions
>
> - The row is `INVENTORIED` with spec `☐ required` and no owner. Should we propose the `.agents/specs/` spec first as its own change and let you review the design before any implementation, per spec-before-code?
> - How should an outside contributor claim a row? `scripts/check-role-discipline.py` expects `row/<ID>`, and we did not want to claim an internal roadmap row unasked (the same question is open on #326 and on our head_dim-128 PR).
> - Would you rather see this as one bring-up or split -- tower first with a reference-dump gate, then merge/decoder e2e?
>
> ## Context on us
>
> We contributed #326 (Marlin NVFP4 for sm_110, landed as `9875931f`) and #325 / #357 / #382 / #384 / #385. We have committed to vllm.cpp as our serving engine and are migrating production to it on Thor; today that production runs Python vLLM, and `Mistral3ForConditionalGeneration` is the model the migration is blocked on -- so this is one we need working and intend to maintain, not a drive-by. We did consider the obvious alternative of switching to an architecture vllm.cpp already serves, and rejected it on cost rather than preference: every accuracy baseline we hold is anchored on Mistral and would need re-establishing, which is more work than the port, and the EU-sovereign Apache-2.0 licensing matters for our product. If Mistral3 is already spoken for internally, or the vision-tower design is one you would rather own, say so and we will put the effort somewhere more useful to you.
>

## Resolution

-
