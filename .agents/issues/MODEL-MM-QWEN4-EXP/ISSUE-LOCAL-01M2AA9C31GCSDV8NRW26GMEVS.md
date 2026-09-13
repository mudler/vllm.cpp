ID: ISSUE-LOCAL-01M2AA9C31GCSDV8NRW26GMEVS
Title: The qwen4_exp layer loop rebuilds the MoE adapter per step, and BorrowWholeOwnedTensor drops d_dev, so every decode step re-derives device residency for all 48 layers of expert towers
Row: MODEL-MM-QWEN4-EXP
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-12
Updated: 2026-09-13
Closed: 2026-09-13

## Problem

`ForwardQwen4Exp` composes `Qwen4ExpMoeBlockWeights` INSIDE the per-layer loop (`src/vllm/model_executor/models/qwen4_exp_forward.cpp:673`), so the adapter is rebuilt once per layer per decode step. `Qwen4ExpMoeBlockWeights` passes the three expert towers through `BorrowWholeOwnedTensor` (`src/vllm/model_executor/models/qwen3_5_weights.cpp:340-368`), which copies dtype, rank, shape, nk and every layout marker and does NOT copy `d_dev`. `d_dev` is the device-residency memo: `ResidentWeight` (`qwen3_5.cpp:1180-1305`) serves a cached device pointer when it is set and otherwise re-decides residency from scratch — on GB10 an alias attempt via `MakeHostBytesDeviceAliasable`, and on a declined borrow a full `Alloc` + `Copy` of the tower. The borrowed view starts with `d_dev == nullptr` on EVERY step, so the memo can never be hit for the expert towers: 3 towers x 48 layers x every decode step, over the ~68 GB the UD-IQ1_S artifact stores its experts in.

The code already names this. `qwen4_exp_forward.cpp:655-660` calls it "a SPEED ceiling this wave inherits rather than a wrong answer" and cites [#2336](https://github.com/mudler/vllm.cpp/issues/2336) §3. That citation does not hold: #2336 is the PLE block, its gate op and the layer loop, and its one `ResidentWeight::d_dev` remark is about the GDN adapter, not the MoE one. So NO issue owns the MoE rebuild, and this is that issue.

The cost is NOT yet measured, and this issue does not claim a number. Which arm the rebuild lands in depends on a 256-byte alignment test (`qwen3_5_weights.cpp:252`) that source inspection cannot decide, so the two arms differ by orders of magnitude: a cheap re-test of an aligned pointer, or ~68 GB of re-upload per token. The measurement owed is the per-step delta of `load_stats` `device_upload_bytes` and of `HostAliasStats`, which is blocked on a second defect: the GGUF load path never registers the reporter (separate issue).

The fix, if the measurement ranks it, is a hoist: build each layer's `MoeBlockWeights` once at load and keep it on the model. The comment at `qwen4_exp_forward.cpp:661-668` states the constraint any hoist must honour — the adapter takes a non-const reference and mutates it through `OwnedBytes::KeepAlive()`, so it is not a pure function of the layer and must not be re-evaluated per placement arm — and warns that the neighbouring GDN adapter spelled the same pass-through as assignment, which deep-copies, and there a per-step rebuild was a freed operand underneath a queued GEMM (#2476).

## Resolution

FIXED 2026-09-13 by `72498897144afb4d7037e3c3f48a8c0f8b4f3223`, which hoists the MoE adapter onto `Qwen4ExpLayerWeights::moe_block` and emplaces it on first use, so the `d_dev` residency memo `ResidentWeight` writes survives the decode step instead of dying with the per-layer temporary.

MEASURED: interleaved same-tree A/B, BASE `3cafbcaf718816f8e60bde525d3c30bac1016a20` vs FIX `72498897144afb4d7037e3c3f48a8c0f8b4f3223`, two rounds alternating BASE/FIX, one boot per arm, released `unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ1_S staged to local disk, `--max-num-seqs 1 --device cuda`, 16-token decode, median inter-token delta. On `dgx:gpu0` (GB10, sm_121a): round 1 BASE 0.2570 tok/s (3.88307 s/token) -> FIX 8.5655 tok/s (0.11689 s/token); round 2 BASE 0.2570 -> FIX 8.5816 tok/s (0.11685 s/token). 33.4x, reproduced in both rounds. TTFT 9.0 s -> 1.23 s. FIX per-token spread across all 14 deltas is 0.1154-0.1180 s. `thor:gpu0` (sm_110) confirms the direction on the same harness: BASE 0.2645 / 0.3004 tok/s -> FIX 4.9278 tok/s.

The measured step lands on the floor this issue's own nsys profile predicted: 0.101 s of GPU kernel time per step against 3.69 s of allocator and copy time, so removing the allocator had to leave roughly 0.10 s, and it left 0.1169 s. That agreement is what makes this a confirmed diagnosis rather than a lucky fix.

MEMORY: the capacity risk the spec raised -- roughly 68 GiB of host mapping plus roughly 68 GiB of device allocations held at once -- did NOT materialise on this artifact. `dgx:gpu0` `VmHWM` 77,397,988 kB BASE against 77,402,288 kB and 77,398,924 kB FIX, identical within 0.006%; `thor:gpu0` 77,304,396 kB BASE against 77,305,524 kB FIX. Steady-state `VmRSS` is LOWER on FIX, 43,977,844 kB and 43,603,596 kB against 77,397,988 kB on BASE. No arm's server log contains `out of memory`, `bad_alloc` or `cudaErrorMemoryAllocation`. This is one artifact (~68 GiB) on two unified-memory boxes of 122-128 GB and does not generalise to a larger checkpoint or a discrete device memory pool.

### The diagnosis this closes on: MEASURED 2026-09-12 on `dgx:gpu0`, 97% of the decode step

Retained because it is the evidence the fix was chosen from, and the denominator
the 2026-09-13 A/B above is read against.

`nsys`, 60 s window over ~15.2 steady-state decode steps, `51c248190`, GB10,
released UD-IQ1_S staged locally. Per step: `cudaMalloc` 378 calls / 2.25 s
(60.5%), `cudaMemcpyAsync` 1,308 calls / 0.82 s (22.2%), `cudaFree` 378 calls /
0.62 s (16.7%), all GPU kernels 0.101 s (2.6%). Host-to-device traffic is 38 GiB
per step over 846 copies, largest 471.859 MB -- one `[512, 640, 2560]` IQ4_NL
expert tower.

The allocation count identifies the caller exactly: 48 layers x (3 expert towers
+ 3 shared-expert weights + router + shared gate) = 384, against 378 measured.
Nothing else in the forward allocates per step. `Backend::Alloc` on CUDA is a
plain `cudaMalloc`, and a ~470 MB one costs 5.96 ms average / 30 ms worst here;
the pooled `cudaMallocAsync` lane is separate at 1,424 calls.

So the cost is the ALLOCATOR, not the copy bandwidth, and the earlier hedge
about the 256-byte alignment test is resolved: the weights decline the alias
(`declined_borrow=96.958 GiB` vs `aliased_in_place=0.014 GiB`) because a GGUF
mmap borrow owns no anonymous pages, so every one of them takes the staging arm.

Decode is 0.25 tok/s and kernel time is 101 ms per token. The fix is the hoist,
and its precedent is twelve lines away in the same function: GDN carried this
exact defect and #2476 repaired it by building `lw.gdn_block` once and holding
it on the model.

This section closed with "DO NOT record an expected speedup here", because
removing the allocator can expose host-side cost it hides. The re-measurement
has now been taken and is recorded above: the exposed host-side cost is
0.016 s per step, against the 0.101 s kernel floor this profile measured.
