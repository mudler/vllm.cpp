ID: ISSUE-LOCAL-01M2TYBE9TWX09XG62QVA5TXHZ
Title: DeepSeek-V4-Flash-Vision refuses at the first forward on --device cpu: the aarch64 i8mm repack auto-enables at load and the stacked-expert row-slice GEMM rejects repacked blocks (deepseek_v4.cpp:631)
Row: MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-18
Updated: 2026-09-18
Closed: -

## Problem

MEASURED 2026-09-18, rc job `1ec05714-8cbd-41d6-8c45-9396f4d92223`, `thor:gpu0` (NVIDIA Thor, compute_cap 11.0, driver 595.78, aarch64, 14 cpu, 122 GiB host RAM), 17:42:56Z-18:57Z. Head `7722c8c716f91be2188408d688e0cc5f3268c6f1`, re-verified ON THE WORKER by re-hashing the extracted tree with `git write-tree` -> `852f827f5bab1ac85b96a436f20e4e39c9ddf464`, not by trusting a label. Build `Release` with NDEBUG defined (compiled probe rc=7), `-DVLLM_CPP_CUDA=ON -DVLLM_CPP_CUDA_ARCHITECTURES=110`, nvcc 13.0, 41 `.cu.o`, `fa2: DISABLED` (sm_110 is outside the FA-2 arch set), `BUILD_RC=0`, cli md5 `964ef06d43231e8125673dae0fdfbc3c`. Artifact byte-verified on the worker: 5,305,248 + 49,991,832,128 + 32,441,484,736 = 82,438,622,112 B plus mmproj 934,462,656 B, from `unsloth/DeepSeek-V4-Flash-Vision-Exp-GGUF` @ `b977d3c0ea2da58dbc12ddae8fb8951a7b3854d0`.

THE FAILURE. On `--device cpu` the SAME binary and the SAME artifact now get PAST engine construction and die at the FIRST FORWARD, verbatim:

  engine-fatal: EngineCore busy loop threw: vt: deepseek-v4 keep-quant expert/group slice requires non-repacked blocks (disable VT_CPU_QUANT_REPACK for the stacked-expert weights) at src/vllm/model_executor/models/deepseek_v4.cpp:631

This is a NEW wall. It is not the block-size refusal (ISSUE-LOCAL-01M2EMPC6T63TVDPQ90GVPRC5F, closed), and it is not the W7-device refusal at `deepseek_v4.cpp:4658`, which did not fire on this build because `V4DeviceKernelsAvailable()` was TRUE.

ANCHOR, verified in the tree at this head rather than quoted from the message. `src/vllm/model_executor/models/deepseek_v4.cpp:631` is the `VT_CHECK(!w.repacked, ...)` inside `GemmRowSlice` (declared at `:627`), the keep-quant GEMM against a ROW-SLICE `[row_off, row_off+N)` of a stacked block weight `[E*out, K]` -- the per-expert (`moe_*_exps`) and per-group (`wo_a`) slice. The message text spans `:632`-`:633`. Three sibling guards carry the same requirement and are NOT what fired here: `:711` (grouped expert GEMM), `:2566` (`resident GemmRowSliceInto`), `:2663` and `:2686` (the resident grouped / fused MoE gate+up paths).

MECHANISM. `VT_CPU_QUANT_REPACK` defaults to `auto` (`src/vt/cpu/cpu_quant_repack_arm.cpp:174`, read once into `QuantRepackActive()`), and `auto` RESOLVES TO ON when the host probe finds i8mm -- which thor's aarch64 CPUs do. The loader's `p.quant_repack` rides `keep_quant AND QuantRepackActive() AND the resolved device being kCPU` (`src/vllm/model_executor/model_loader/gguf_keep_quant.cpp:458`-`:470`, contract at `include/vllm/model_executor/model_loader/gguf_keep_quant.h:344`-`:358`). So on `--device cpu` on an i8mm aarch64 host the stacked-expert q8_0 blocks are rewritten at load into the `block_q8_0x4` i8mm interleave, and the row-slice GEMM then refuses them, because a slice is computed as a byte offset over whole `RowSizeBytes` rows and the interleave breaks that row-major block layout.

The two conditions are therefore BOTH defaults: nobody sets `VT_CPU_QUANT_REPACK`, and `--device cpu` is a documented production flag. A user on an aarch64 host reaches this with no environment at all.

SCOPE, stated narrowly. This is the `--device cpu` arm only. The CUDA arm of the same run LOADED AND GENERATED through `vllm-cli` (` Paris` as the first token; that liveness result is recorded on the row spec and in `docs/USAGE.md`), and on CUDA `QuantRepackForDevice` returns false for the device term, so the repack never happens and `:631` is never reached.

TWO CANDIDATE FIXES, NAMED AND NOT CHOSEN. This issue records the gap; selecting between them is a production change that needs its own spec and a fresh review.
(a) Suppress repack at load for the `kStackedExpertWeight` role, so the stacked expert / group towers keep row-major blocks while every other q8_0 weight still gets the i8mm interleave. This keeps the prefill lever for the tensors that have a reader and removes it only from the tensors that do not. It needs a per-role term in the loader beside the existing device term, and it must not silently change the residency choice for unrelated tensors.
(b) Teach `GemmRowSlice` (and its four siblings) to read repacked rows, by making the slice offset interleave-aware instead of a flat `RowSizeBytes` byte offset. This keeps the lever on every tensor but puts the layout knowledge in five call sites.

WHAT IS NOT MEASURED HERE. Whether `VT_CPU_QUANT_REPACK=0` actually carries the `--device cpu` load to a token was NOT run on this job, so the workaround the message itself suggests is UNVERIFIED. `vllm-server` is unmeasured on this run in either direction: its readiness poll was 600 s against a load that took ~1390 s, so the server was still loading when it was curled and killed, and `srv-completion.json` is 0 bytes.

## Resolution

-
