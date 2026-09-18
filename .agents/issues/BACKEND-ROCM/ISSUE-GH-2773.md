ID: ISSUE-GH-2773
Title: Characterize Qwen3.5-0.8B CPU and ROCm state numerics on gfx1100
Row: BACKEND-ROCM
State: OPEN
Kind: UNKNOWN
GitHub: 2773
Mirror: DIVERGED
Availability: FULL
Created: 2026-09-03
Updated: 2026-09-07
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `BACKEND-ROCM`
>
> Supersedes #1588 for implementation traceability. The original issue was opened without the required first-line `Row:` metadata, and the current contributor token cannot edit another author's issue body.
>
> ## Recorded gap
>
> `.agents/backend-matrix.md` records that Qwen3.5-0.8B runs all-native on gfx1100 and that its CPU/ROCm numerical characterization is open. This issue owns that characterization.
>
> ## Scope
>
> 1. Run the pinned safetensors model end to end on gfx1100 through the ROCm device path and the CPU comparison path on identical weights, prompts, token prefixes, and greedy sampling.
> 2. Compare numerics at named layer boundaries, not only at the logits: residual stream, attention output, MLP output, active K/V state, and active GDN convolution/SSM state after the persistent write.
> 3. Audit dtype polarity per `.agents/porting.md`: an `f32` buffer where vLLM resolves BF16 passes every token gate and moves twice the bytes. Name each wider buffer, its executing source chain, and its byte cost.
> 4. Report `max_abs`, RMS, and relative-L2 deltas for every joined boundary. Preserve applicable pinned-upstream per-operation tolerances. Where upstream defines no layer-level acceptance tolerance, report the metric descriptively and leave end-to-end oracle-backed correctness as the acceptance gate; do not invent a propagated rounding envelope.
> 5. Characterize `auto`, `bfloat16`, and `fp8_e4m3` KV-cache modes. Record requested, resolved, and physical cache dtypes and the selected attention/cache-store operators. Distinct physical modes require cache-matched pinned-vLLM captures and teacher-forced gaps before they can become permanent sacred-gate cases.
> 6. Prove that every ROCm arm uses its mode-appropriate native provider set with zero declines and zero reference-tier hits.
>
> ## Artifact pin
>
> The developer supplied `CHECKPOINT_ROOT=/home/vikash/models`. Use:
>
> ```text
> Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17
> /home/vikash/models/Qwen3.5-0.8B
> ```
>
> Required files include:
>
> ```text
> model.safetensors-00001-of-00001.safetensors
>   sha256 04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696
> model.safetensors.index.json
>   sha256 d8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e
> config.json
>   sha256 b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204
> ```
>
> ## Current gates
>
> 1. **#2856 landed.** Main `f98b638673b4d2edc0250eec56d229357ea38ab1` contains merge `d6c63e15ae6825f94dc18769163cfe7b037e7954`. The operator built this unchanged main for gfx1100 with HIP 7.15.26333 and Clang 23. The historical default Qwen3.5 gate passed 137 assertions, with 15 strict prompts, one near-tie-only prompt, maximum gap zero, and zero provider declines. The backend suite passed 46 cases and 84,078 assertions.
> 2. **The approved active-pin runtime builds and runs the model.** The dedicated runtime contains the locally built wheel for vLLM `e126687a9a828d513c01a07cd69f025f27d63280`, wheel SHA256 `7e6efb7b3226360cd67411d62e139425d550a77407ad16391099b8cbc55b340b`. It runs inside immutable image `sha256:80aab4c182a1f3eeebe286173977e57fcaf10a049b41f475655b35d285de31dc` with production graphs, V2ModelRunner, ROCM_ATTN, and Triton/FLA GDN. The configured historical wrapper remains unchanged. Using the reviewed `c75173` capture snapshot, the active runtime completed 16 prompts × 16 tokens × 10 repetitions for each requested cache mode: `auto`, `bfloat16`, and `fp8_e4m3`. All three captures were deterministic. Auto and BF16 output bytes matched each other; FP8 differed at prompt 7. Provenance records configuration resolution, with physical storage unobserved.
> 3. **Active auto token revalidation passes on unchanged main.** The actual C++ auto stream matched all 256 active-oracle positions. Ten full-prefix teacher-forced repetitions were identical, with all 256 gaps zero. An unchanged-source test binary selecting external candidate files passed 137 assertions, all 16 prompts strictly, and zero provider declines. No tracked golden changed. This establishes the scoped auto token result; physical-cache, state, operation, and performance gates remain open. Full-prefix scoring is not an incremental decode witness.
> 4. **PR #2932 is rebased and published for implementation review.** Its immutable head is `81855a3f7f0ed0a0bb21edd3fcd6c2f94b4a8766`, based on real main `f98b63867`. Fresh independent review passed with 44 detected effective mutations and no findings. The implementer, reviewer, and operator passed all 49 capture tests and two mode tests. The operator passed the full preflight in its recorded isolated environment and chained that result to exact-SHA publication. Resource-dependent skips, the separate handoff check, new-head CI, and maintainer acceptance remain explicitly separate. See the [current evidence and handoff](https://github.com/mudler/vllm.cpp/pull/2932#issuecomment-5564609782). Issue #2923 retains the shared-AttnBlock wiring debt.
> 5. **All three explicit C API captures are complete.** A private client enters through `include/vllm.h`, preserves ABI model defaults except the model path and cache selector, and records every raw request. Its success-marker defect was repaired before GPU use. Fresh independent review, effective mutations, and the operator's 31 original plus nine publication cases passed. The final client source SHA256 is `07874f1daa3973423b1dbf8bd4199e13b224efb3e65653e184a540ea4745b346`; it links unchanged main `f98b63867`. Each mode completed 160 requests and 2,560 token IDs. Independent review and the operator verified every completion, count, prompt copy, request file, and repetition.
> 6. **Cache-matched active-pin token comparison meets the existing near-tie rule.** Fresh production oracle captures use the active pin's memory-utilization default of 0.92, matching the requested ABI fraction. All ten local and oracle repetitions are individually deterministic. Auto and BF16 each match all 2,560 IDs strictly and have 256 zero teacher-forced gaps. FP8 matches 2,470/2,560 IDs strictly. Its nine differing positions are prompt 15, tokens 7–15, repeated in every repetition; indices are zero-based. The first divergence is local token 760 versus oracle token 9175. Scoring the actual local FP8 prefix gives 125 milli-nats there and zero at all other 255 positions. Ten unrounded scoring repeats agreed in-process. With the existing 500-milli-nat rule, FP8 classifies as 15 strict prompts, one near-tie-only prompt, and zero gap failures. Strict FP8 equality still fails. The tool persists a reference log-probability hash, not all raw repetitions. Full-prefix prefill scoring does not prove incremental-decode execution. No permanent golden, threshold, or test scope changed.
> 7. **Physical and paired state acceptance remains PENDING.** Native CPU/HIP provider/state controls and active-vLLM state captures now complete in all three cache modes. Independent audits verify within-mode repeatability and disabled-control output preservation. Actual resolved cache sizes and scheduler settings still differ between engines. The existing startup memory-profile gap now has canonical owner #3046 / `KV-WARMUP-PROFILE`; it is one prerequisite, not a complete equivalence fix. See the [native control result](https://github.com/mudler/vllm.cpp/issues/2773#issuecomment-5570314260) and [upstream nine-control result](https://github.com/mudler/vllm.cpp/issues/2773#issuecomment-5576185808). Cross-engine control eligibility, state comparison, executed-provider attribution, physical conformance, and performance remain owed. The reviewed live-budget observation completed all 18 native controls and records 2,048 tokens, versus the active oracle’s 8,192. The requested native value stays zero. See the [direct scheduler observation](https://github.com/mudler/vllm.cpp/issues/2773#issuecomment-5576455674).
>
> 8. **Windows CI remains failing under existing #2403.** Both new-head Windows jobs fail API-server cases 58, 59, and 61 with `0xC0000409`. The [exact f98 scheduled base run](https://github.com/mudler/vllm.cpp/actions/runs/34063031377) reproduces those same failures with matching recorded runner image, compiler, and backend arguments. The cause remains unresolved; this is no waiver. #2403 already appears under `## Owed` in `.agents/specs/bench-rocm-strix-vllmcpp-arm-head.md`. Its formal product-repair row remains unassigned. See [PR CI](https://github.com/mudler/vllm.cpp/actions/runs/34079668883).
>
> 9. **The upstream operation prerequisite is FAILING.** Unmodified active-pin tests ran in the dedicated ROCm runtime under the GPU mutex: 173 cases passed and five variable-length causal-convolution cases failed their existing `atol=0.05`, `rtol=0.01` output check. All eight packed-GDN cases, six non-speculative gated-delta cases, and 140 convolution update/gather cases passed; variable-length convolution passed 19 of 24. No case skipped. The test harness uses pinned source with the installed active wheel; its missing test-only `tblib==3.1.0` dependency was installed in that owned runtime. A fresh process reran exactly those five cases and reproduced all five failures. Their selected-state checks pass; their output checks fail. The failures are under investigation before operation acceptance. This is not evidence of a local C++ defect and does not authorize changing tolerances.
>
> ## Gate order
>
> Correctness comes first. Any throughput number for this model remains provisional until the unchanged default token gate passes. New FP8 goldens require a deterministic cache-matched pinned-vLLM capture; the existing BF16 gap table cannot adjudicate an FP8 stream or a suffix after input-prefix divergence.
>

## Resolution

-
