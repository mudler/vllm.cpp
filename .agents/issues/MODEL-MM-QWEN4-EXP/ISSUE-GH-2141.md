ID: ISSUE-GH-2141
Title: **Supersedes one claim in the [#2123](https://github.com/mudler/vllm.cpp/issues/2123) row above: the epsilon-placement defect did NOT survive in the W3 HOST suite.** That row says it did. W3 gated the placement deliberately, at `tests/vllm/models/test_qwen4_exp_hc.cpp:268-276`, inside "qwen4_exp grouped RMSNorm mirrors RMSNormGated(group_size) at the lane pin" — a `big_eps = 4.0f` probe against the double reference `NormRefD`, whose own comment says an eps-placement defect is invisible to every golden at the model's real `1e-6` and that "a case at an eps large enough to separate them is the only thing that gates it". Pre-repair, with `+ eps` moved outside the rsqrt, that probe is RED at `CHECK( 0.802185 < 1e-05 )` — 2 of 14 cases at `origin/main` — while the unmutated kernel clears the same `kTol = 1e-5` by `7.77e-08`, so it discriminates by seven orders of magnitude. The ungated arm was the DEVICE suite ALONE, and W5b-2's golden case D at `hyper_scale = 0.01` is what shuts it. Case D additionally sharpens the HOST arm from those 2 red assertions to 10 (M16: RED, 3 of 15 cases), which is an ENHANCEMENT of a gate that already fired and not a hole closed. The [#2123](https://github.com/mudler/vllm.cpp/issues/2123) row is not edited and never will be: `merge=union` DUPLICATES an edited row instead of merging it, so the immutability protects the merge driver rather than the error, and an appended superseding row is what an append-only log is for. A second row keyed on #2123 is equally impossible — `check-agent-record.py` reports a repeated issue number as the duplicate two branches appending the same issue would produce — hence this row's own issue. The spec, its `## Mutation record — W5b-2`, the M16 table row and the case-D comment in the test file were already corrected in `11a61e3bd`
Row: MODEL-MM-QWEN4-EXP
State: UNKNOWN
Kind: record
GitHub: 2141
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:826`

### Frozen archive evidence

> | [#2141](https://github.com/mudler/vllm.cpp/issues/2141) | `MODEL-MM-QWEN4-EXP` | **Supersedes one claim in the [#2123](https://github.com/mudler/vllm.cpp/issues/2123) row above: the epsilon-placement defect did NOT survive in the W3 HOST suite.** That row says it did. W3 gated the placement deliberately, at `tests/vllm/models/test_qwen4_exp_hc.cpp:268-276`, inside "qwen4_exp grouped RMSNorm mirrors RMSNormGated(group_size) at the lane pin" — a `big_eps = 4.0f` probe against the double reference `NormRefD`, whose own comment says an eps-placement defect is invisible to every golden at the model's real `1e-6` and that "a case at an eps large enough to separate them is the only thing that gates it". Pre-repair, with `+ eps` moved outside the rsqrt, that probe is RED at `CHECK( 0.802185 < 1e-05 )` — 2 of 14 cases at `origin/main` — while the unmutated kernel clears the same `kTol = 1e-5` by `7.77e-08`, so it discriminates by seven orders of magnitude. The ungated arm was the DEVICE suite ALONE, and W5b-2's golden case D at `hyper_scale = 0.01` is what shuts it. Case D additionally sharpens the HOST arm from those 2 red assertions to 10 (M16: RED, 3 of 15 cases), which is an ENHANCEMENT of a gate that already fired and not a hole closed. The [#2123](https://github.com/mudler/vllm.cpp/issues/2123) row is not edited and never will be: `merge=union` DUPLICATES an edited row instead of merging it, so the immutability protects the merge driver rather than the error, and an appended superseding row is what an append-only log is for. A second row keyed on #2123 is equally impossible — `check-agent-record.py` reports a repeated issue number as the duplicate two branches appending the same issue would produce — hence this row's own issue. The spec, its `## Mutation record — W5b-2`, the M16 table row and the case-D comment in the test file were already corrected in `11a61e3bd` | record |

## Resolution

-
