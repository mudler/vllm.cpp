ID: ISSUE-GH-730
Title: `sanitize-cpu (address,undefined)` red on every PR, unattributed because the lane is CANCELLED on every `main` run (merges outrun the ~35 min job) so there is no baseline. Two findings, and the issue's title names only the first. **(a) `test_ltx2_video`** — `Ltx2LoadVaeWeights` misaligned `const uint16_t` load; that is [#674](https://github.com/mudler/vllm.cpp/issues/674), FIXED by `fc903b8dd`, which post-dates the run this was observed on ([31797185395](https://github.com/mudler/vllm.cpp/actions/runs/31797185395), head `a9063af53`). [#693](https://github.com/mudler/vllm.cpp/issues/693) is the same site and is satisfied by that landing. **(b) `test_nemotron_h_scaffold`** — the live one, and NOT in the issue title: `ForwardNemotronHForCausalLM` (`nemotron_h_registry.cpp:100`) opens its handle with the universal registry `static_cast<NemotronHLoadedModel&>(model)` — the cast itself is `nemotron_h_registry.cpp:102`, and `nemotron_h_registry.cpp:112:30` (`nh.params()`) is the member call THROUGH that reference which is what UBSan names, so the two are separate lines and neither substitutes for the other — but the refusal subcase fabricated a `struct StubModel : vllm::LoadedModel` and passed it in, so the downcast was type confusion. UBSan's vptr check reported "member call on address ... which does not point to an object of type 'NemotronHLoadedModel'" and `-fno-sanitize-recover=all` aborted. Latent from W3 (the forward was then an unconditional `VT_CHECK` that never touched `model`); W4 (#517) added the downcast and made it live. Repaired in the TEST — the subcase now forwards through the real `NemotronHLoadedModel` that `factory->load_weights` returns, which needs no checkpoint because no NemotronH weight loader exists yet, so it asserts the same refusal on the production type. `test_op_parity`, the third red on that run, is [#755](https://github.com/mudler/vllm.cpp/issues/755) and out of scope
Row: MODEL-TEXT-nemotron-h-nemotron-hfor-causal-lm
State: UNKNOWN
Kind: bug
GitHub: 730
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:200`

### Frozen archive evidence

> | [#730](https://github.com/mudler/vllm.cpp/issues/730) | — | `sanitize-cpu (address,undefined)` red on every PR, unattributed because the lane is CANCELLED on every `main` run (merges outrun the ~35 min job) so there is no baseline. Two findings, and the issue's title names only the first. **(a) `test_ltx2_video`** — `Ltx2LoadVaeWeights` misaligned `const uint16_t` load; that is [#674](https://github.com/mudler/vllm.cpp/issues/674), FIXED by `fc903b8dd`, which post-dates the run this was observed on ([31797185395](https://github.com/mudler/vllm.cpp/actions/runs/31797185395), head `a9063af53`). [#693](https://github.com/mudler/vllm.cpp/issues/693) is the same site and is satisfied by that landing. **(b) `test_nemotron_h_scaffold`** — the live one, and NOT in the issue title: `ForwardNemotronHForCausalLM` (`nemotron_h_registry.cpp:100`) opens its handle with the universal registry `static_cast<NemotronHLoadedModel&>(model)` — the cast itself is `nemotron_h_registry.cpp:102`, and `nemotron_h_registry.cpp:112:30` (`nh.params()`) is the member call THROUGH that reference which is what UBSan names, so the two are separate lines and neither substitutes for the other — but the refusal subcase fabricated a `struct StubModel : vllm::LoadedModel` and passed it in, so the downcast was type confusion. UBSan's vptr check reported "member call on address ... which does not point to an object of type 'NemotronHLoadedModel'" and `-fno-sanitize-recover=all` aborted. Latent from W3 (the forward was then an unconditional `VT_CHECK` that never touched `model`); W4 (#517) added the downcast and made it live. Repaired in the TEST — the subcase now forwards through the real `NemotronHLoadedModel` that `factory->load_weights` returns, which needs no checkpoint because no NemotronH weight loader exists yet, so it asserts the same refusal on the production type. `test_op_parity`, the third red on that run, is [#755](https://github.com/mudler/vllm.cpp/issues/755) and out of scope | bug |

## Resolution

-
