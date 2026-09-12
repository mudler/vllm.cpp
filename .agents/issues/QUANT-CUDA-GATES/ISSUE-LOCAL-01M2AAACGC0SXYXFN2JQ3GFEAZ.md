ID: ISSUE-LOCAL-01M2AAACGC0SXYXFN2JQ3GFEAZ
Title: The GGUF k-quant MoE arm has no device-resident path: MoeBlock's reference arm round-trips through host memory per layer, so no GGUF MoE model can be CUDA-graph captured
Row: QUANT-CUDA-GATES
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-12
Updated: 2026-09-12
Closed: -

## Problem

`MoeBlock` (`src/vllm/model_executor/models/qwen3_5.cpp:7178`) selects among exactly three arms: NVFP4 experts go to `MoeBlockFusedCuda` / `MoeBlockFusedMarlinCuda`, bf16 experts go to `MoeBlockBf16Cuda`, and EVERYTHING ELSE takes what the code calls the reference path. A GGUF k-quant checkpoint has neither fp4 nor bf16 experts, so every GGUF MoE model this tree serves takes the third arm.

That arm is host-mediated by construction, and the tree says so at `:7191-7193`: the two fused arms do the expert compute "fully on-device (no host round-trip — capturable)", while "the bf16 / CPU / GGUF reference below keeps the per-expert token-gather path on host (not the capture target)". Per MoE layer it copies the hidden state to host and synchronizes (`:7211-7213`), downloads the router top-k weights and ids (`:7225-7226`), builds the expert lists in a host loop, and then makes three `KqGrouped` calls (`:6133-6145`) each of which uploads the activation, launches ONE grouped GEMM, and `Download`s the f32 result — a blocking drain — with the SwiGLU computed in a host loop between them (`:7256-7258`). The routed-expert output is assembled in a `std::vector<uint16_t>` on the host.

Two consequences, and the second is the one no gate can see. Per decode token on a 48-layer model that is ~4 full pipeline stalls per layer, roughly 200 device synchronizations for work whose payload at T=1 is a few hundred kilobytes. And because the path leaves and re-enters the device every layer, CUDA graph capture is structurally impossible for it — which means the absence of a decode graph on qwen4_exp is not a qwen4_exp property at all, as `.agents/specs/qwen4-exp-flash-next.md` §"Known bottlenecks" currently states it, but a property of the whole GGUF MoE lane.

Nothing here is a missing kernel. `vt::MatmulBTQuantGrouped` already runs the three grouped GEMMs on device for the encodings the shipped artifacts use (`IsCuda32BlockKeepQuantSupported` covers IQ4_NL / Q5_0 / Q4_0; `IsCudaKeepQuantSupported` covers the Q8_K family), and `vt::OpId::kMoeSiluMul` is a registered device op. What is missing is an arm that keeps the intermediates on device between them, in the shape `MoeBlockBf16Cuda` already has.

NOT MEASURED YET, and this issue claims no number. The sync count and the per-step launch count are owed from an nsys trace on `dgx:gpu0`; this issue records the structure, which is readable from the source, and not a cost.

## Resolution

-

### MEASURED 2026-09-12: this is a CAPTURABILITY defect, not a speed one

The same `nsys` window that profiled the decode step counts 367
`cudaStreamSynchronize` calls per step costing **1 ms in total** -- 0.0% of a
3.95 s step. The host round-trip this issue describes is real and is exactly
where the structure says it is, but it is not what makes the model slow today:
97% of the step is `cudaMalloc`/`cudaFree`/memcpy from the per-step MoE adapter
rebuild (`.agents/issues/MODEL-MM-QWEN4-EXP/ISSUE-LOCAL-01M2AA9C31GCSDV8NRW26GMEVS.md`).

That reorders the work but does not close this. Two things keep it open. Graph
capture is impossible while the path leaves the device, and the tree says so at
`qwen3_5.cpp:7191-7193`. And the trace counts 2,534 `cudaLaunchKernel` per step:
20 ms, which is 0.5% of a 3.95 s step and would be ~20% of a 101 ms one. Both
arguments get STRONGER once the allocator defect is fixed, so this should be
scheduled after it and re-measured against the new step, not against this one.
