ID: ISSUE-GH-2375
Title: documentation-checkpoint reds on main: four recent commits reached main without arriving on a task branch
Row: ENG-PREFLIGHT-COMPILES
State: OPEN
Kind: UNKNOWN
GitHub: 2375
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-31
Updated: 2026-08-31
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## Observed
>
> The `documentation-checkpoint` CI job fails on main's head `8846e3c4b` (job 99351159519) with four attribution errors. The gate runs on main pushes only, which is why no open PR surfaces it:
>
> ```
> ERROR: e1b5df1a6: repository change (tests/vllm/multimodal/ltx2_video_fixture.h) reached main without arriving on a task branch.
> ERROR: c2501cc71: repository change (.w9a/bridge2.txt, .w9a/build.pid, .w9a/final_moe.txt, .w9a/green.txt, ... (+19)) reached main without arriving on a task branch.
> ERROR: 330599b26: repository change (tests/vllm/models/test_glm5_next_forward.cpp) reached main without arriving on a task branch.
> ERROR: 8846e3c4b: repository change (src/vt/cuda/cuda_quant_dot.cu) reached main without arriving on a task branch.
> ```
>
> ## What is unclear, and what the issue asks
>
> Whether these are (a) genuine direct-to-main landings, or (b) false positives of the branch-correlation heuristic — squash-merged PRs lose their branch name at the forge, and a branch named anything other than `row/<ID>` (for example the `docs/*` branches visible in the ref listing) may be invisible to the correlation. The `c2501cc71` case is suspicious in both directions: its message says it DROPS W9a's scratch directory, yet the gate attributes the `.w9a/*` changes to it, which implies the scratch files were first committed to main by some earlier commit the gate did not flag.
>
> ## Owed
>
> The gate's owner reconciles the four commits against their actual landing paths (forge PR numbers exist for squash merges) and either repairs the process that landed them or repairs the correlation so a squash-merged `row/<ID>` PR is recognized. Likely owning rows by content: `MODEL-MM-GLM53-FLASH` (`.w9a/`, `test_glm5_next_forward.cpp`), `QUANT-CUDA-IQ4XS-IQ2XS` (`cuda_quant_dot.cu`), and the LTX2 fixture's row for `e1b5df1a6`. Until reconciled, `documentation-checkpoint` stays red on every main push and masks the next genuine violation.
>
> Context found while classifying CI for #2368/#2370/#2373; none of those PRs touch this gate.
>
> FOLLOWING_AGENTS_PROTOCOL
>
> Following-Agents-Protocol: true
> AI-Assisted: true
> Assisted-by: AGENT:zai-glm-5.3-flash [maki]

## Resolution

-
