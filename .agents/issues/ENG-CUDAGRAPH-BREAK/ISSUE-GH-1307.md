ID: ISSUE-GH-1307
Title: W4 of the break-point capture seam: the PERSISTENT DEVICE INPUT PATH becomes a seam capability instead of one driver's private code. `StepDevInputs` exists on 41 lines of `src/vllm/model_executor/models/qwen3_5.cpp` and ZERO lines of `qwen3_moe.cpp`, `qwen3.cpp`, `deepseek_v2.cpp`, `voxtral.cpp` and `qwen3_dflash.cpp`, and the four drivers without it replay against HOST vectors — which is why `qwen3.cpp:1106` DECLINES its decode graph outright while the asynchronous device-token mirror is live (`depth-1 graph ON PASS 78/78`, `depth-2 graph OFF PASS 82/82`, `depth-2 graph ON FAIL, slots 1-3 degenerate`; [#323](https://github.com/mudler/vllm.cpp/issues/323), [#1179](https://github.com/mudler/vllm.cpp/issues/1179)) and why [#1305](https://github.com/mudler/vllm.cpp/issues/1305)'s three registrations admit an asynchronous step with no decline at all. Lands `vt::PersistentStepInput`: one capture-stable per-step device input, bound to a destination the driver owns, refreshed IN PLACE from either a pinned host staging block or a DEVICE source, with the address-stability rule as a refusal rather than a comment and the refreshing ARM as an observable rather than an inference. Also migrates `Qwen3_5DecodeGraph` and `Qwen3_5DenseDecodeGraph` onto `vt::GraphCaptureScope` + `vt::BreakableGraph`, which retires the last two batched-driver `VLLM_CPP_CUDAGRAPH` reads
Row: ENG-CUDAGRAPH-BREAK
State: UNKNOWN
Kind: feature
GitHub: 1307
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:430`

### Frozen archive evidence

> | [#1307](https://github.com/mudler/vllm.cpp/issues/1307) | `ENG-CUDAGRAPH-BREAK` | W4 of the break-point capture seam: the PERSISTENT DEVICE INPUT PATH becomes a seam capability instead of one driver's private code. `StepDevInputs` exists on 41 lines of `src/vllm/model_executor/models/qwen3_5.cpp` and ZERO lines of `qwen3_moe.cpp`, `qwen3.cpp`, `deepseek_v2.cpp`, `voxtral.cpp` and `qwen3_dflash.cpp`, and the four drivers without it replay against HOST vectors — which is why `qwen3.cpp:1106` DECLINES its decode graph outright while the asynchronous device-token mirror is live (`depth-1 graph ON PASS 78/78`, `depth-2 graph OFF PASS 82/82`, `depth-2 graph ON FAIL, slots 1-3 degenerate`; [#323](https://github.com/mudler/vllm.cpp/issues/323), [#1179](https://github.com/mudler/vllm.cpp/issues/1179)) and why [#1305](https://github.com/mudler/vllm.cpp/issues/1305)'s three registrations admit an asynchronous step with no decline at all. Lands `vt::PersistentStepInput`: one capture-stable per-step device input, bound to a destination the driver owns, refreshed IN PLACE from either a pinned host staging block or a DEVICE source, with the address-stability rule as a refusal rather than a comment and the refreshing ARM as an observable rather than an inference. Also migrates `Qwen3_5DecodeGraph` and `Qwen3_5DenseDecodeGraph` onto `vt::GraphCaptureScope` + `vt::BreakableGraph`, which retires the last two batched-driver `VLLM_CPP_CUDAGRAPH` reads | feature |

## Resolution

-
