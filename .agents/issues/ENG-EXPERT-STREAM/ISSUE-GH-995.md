ID: ISSUE-GH-995
Title: `check-env-doc` and `test_check_env_doc` were RED on `origin/main`, so every branch cut from it inherited a preflight failure its own diff did not cause: `VT_MOE_EXPERT_STREAM`, `VT_MOE_EXPERT_STREAM_SLOTS` and `VT_MOE_EXPERT_STREAM_SLOT_BYTES` are read from `src/vllm/model_executor/models/qwen3_5.cpp` (`:5145`, `:5195`, `:5189` @ `4496ef196`) and appeared in neither `docs/ENVIRONMENT.md` (`grep -c` returned **0**) nor `scripts/env-doc-allowlist.txt`. They arrived with the `ENG-EXPERT-STREAM` W4 wiring commit `3005447f8` ([#993](https://github.com/mudler/vllm.cpp/pull/993)). Found while gating [#986](https://github.com/mudler/vllm.cpp/issues/986) and proved pre-existing with a matched-arm check rather than asserted: the three sites are in a file that branch does not touch, and `git diff origin/main...HEAD | grep '^+.*VT_MOE_EXPERT'` returns nothing. It is a PRE-FLIGHT gate, so it failed before every edit and presented to each author in turn as a red their own diff caused -- the shape [#965](https://github.com/mudler/vllm.cpp/issues/965) and [#968](https://github.com/mudler/vllm.cpp/issues/968) both took. The discoverer deliberately did NOT fix it in flow: documenting a knob means stating its default and when to touch it, and the expert-streamer's slot accounting belongs to the row that added it, so a plausible-sounding entry written by a passer-by is how `docs/ENVIRONMENT.md` stops being trustworthy. FIXED in [#997](https://github.com/mudler/vllm.cpp/pull/997) by DOCUMENTING all three in `docs/ENVIRONMENT.md`, not by allowlisting them: a knob that changes the host memory a MoE decode reserves and disables the default-on grouped-MoE path is a deployment surface, not a kernel-internal micro-tuning switch
Row: ENG-EXPERT-STREAM
State: UNKNOWN
Kind: bug
GitHub: 995
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:273`

### Frozen archive evidence

> | [#995](https://github.com/mudler/vllm.cpp/issues/995) | `ENG-EXPERT-STREAM` | `check-env-doc` and `test_check_env_doc` were RED on `origin/main`, so every branch cut from it inherited a preflight failure its own diff did not cause: `VT_MOE_EXPERT_STREAM`, `VT_MOE_EXPERT_STREAM_SLOTS` and `VT_MOE_EXPERT_STREAM_SLOT_BYTES` are read from `src/vllm/model_executor/models/qwen3_5.cpp` (`:5145`, `:5195`, `:5189` @ `4496ef196`) and appeared in neither `docs/ENVIRONMENT.md` (`grep -c` returned **0**) nor `scripts/env-doc-allowlist.txt`. They arrived with the `ENG-EXPERT-STREAM` W4 wiring commit `3005447f8` ([#993](https://github.com/mudler/vllm.cpp/pull/993)). Found while gating [#986](https://github.com/mudler/vllm.cpp/issues/986) and proved pre-existing with a matched-arm check rather than asserted: the three sites are in a file that branch does not touch, and `git diff origin/main...HEAD \| grep '^+.*VT_MOE_EXPERT'` returns nothing. It is a PRE-FLIGHT gate, so it failed before every edit and presented to each author in turn as a red their own diff caused -- the shape [#965](https://github.com/mudler/vllm.cpp/issues/965) and [#968](https://github.com/mudler/vllm.cpp/issues/968) both took. The discoverer deliberately did NOT fix it in flow: documenting a knob means stating its default and when to touch it, and the expert-streamer's slot accounting belongs to the row that added it, so a plausible-sounding entry written by a passer-by is how `docs/ENVIRONMENT.md` stops being trustworthy. FIXED in [#997](https://github.com/mudler/vllm.cpp/pull/997) by DOCUMENTING all three in `docs/ENVIRONMENT.md`, not by allowlisting them: a knob that changes the host memory a MoE decode reserves and disables the default-on grouped-MoE path is a deployment surface, not a kernel-internal micro-tuning switch | bug |

## Resolution

-
