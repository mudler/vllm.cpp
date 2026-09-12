ID: ISSUE-LOCAL-01M2AA9C31GCSDV8NRW26GMEVS
Title: The qwen4_exp layer loop rebuilds the MoE adapter per step, and BorrowWholeOwnedTensor drops d_dev, so every decode step re-derives device residency for all 48 layers of expert towers
Row: MODEL-MM-QWEN4-EXP
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-12
Updated: 2026-09-12
Closed: -

## Problem

`ForwardQwen4Exp` composes `Qwen4ExpMoeBlockWeights` INSIDE the per-layer loop (`src/vllm/model_executor/models/qwen4_exp_forward.cpp:673`), so the adapter is rebuilt once per layer per decode step. `Qwen4ExpMoeBlockWeights` passes the three expert towers through `BorrowWholeOwnedTensor` (`src/vllm/model_executor/models/qwen3_5_weights.cpp:340-368`), which copies dtype, rank, shape, nk and every layout marker and does NOT copy `d_dev`. `d_dev` is the device-residency memo: `ResidentWeight` (`qwen3_5.cpp:1180-1305`) serves a cached device pointer when it is set and otherwise re-decides residency from scratch — on GB10 an alias attempt via `MakeHostBytesDeviceAliasable`, and on a declined borrow a full `Alloc` + `Copy` of the tower. The borrowed view starts with `d_dev == nullptr` on EVERY step, so the memo can never be hit for the expert towers: 3 towers x 48 layers x every decode step, over the ~68 GB the UD-IQ1_S artifact stores its experts in.

The code already names this. `qwen4_exp_forward.cpp:655-660` calls it "a SPEED ceiling this wave inherits rather than a wrong answer" and cites [#2336](https://github.com/mudler/vllm.cpp/issues/2336) §3. That citation does not hold: #2336 is the PLE block, its gate op and the layer loop, and its one `ResidentWeight::d_dev` remark is about the GDN adapter, not the MoE one. So NO issue owns the MoE rebuild, and this is that issue.

The cost is NOT yet measured, and this issue does not claim a number. Which arm the rebuild lands in depends on a 256-byte alignment test (`qwen3_5_weights.cpp:252`) that source inspection cannot decide, so the two arms differ by orders of magnitude: a cheap re-test of an aligned pointer, or ~68 GB of re-upload per token. The measurement owed is the per-step delta of `load_stats` `device_upload_bytes` and of `HostAliasStats`, which is blocked on a second defect: the GGUF load path never registers the reporter (separate issue).

The fix, if the measurement ranks it, is a hoist: build each layer's `MoeBlockWeights` once at load and keep it on the model. The comment at `qwen4_exp_forward.cpp:661-668` states the constraint any hoist must honour — the adapter takes a non-const reference and mutates it through `OwnedBytes::KeepAlive()`, so it is not a pure function of the layer and must not be re-evaluated per placement arm — and warns that the neighbouring GDN adapter spelled the same pass-through as assignment, which deep-copies, and there a per-step rebuild was a freed operand underneath a queued GEMM (#2476).

## Resolution

-
