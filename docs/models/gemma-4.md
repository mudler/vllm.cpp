# Gemma 4

Use this page for Gemma 4 checkpoints, commands, supported arms, and current limitations.

## Gemma4 FP8 on ROCm (RDNA4)

Dual-GPU resident FP8 MoE and SharedK-WMMA prefill are controlled via
ENVIRONMENT.md (`VT_GEMMA4_RESIDENT_*`, `VT_ATTN_*`). Defaults stay safe off RDNA4.
GetBlas keeps two per-thread hipBLAS handles (`tls_slots[2]`, device 1 → slot 1)
so a 0→1 hop does not destroy GPU0's handle. `ProductGetBlasHandle` is the
test accessor for that file-local `GetBlas`. HIP live probe is a separate CTest
target (exit 77 if `HIP_VISIBLE_DEVICES` empty); it enters capture so production `StreamIsCapturing` is load-bearing. No new env.
Prefill peer (#839) unpins dequant cache only after observed retirement; a failed fill/ready lease is retired with RetireFillLocked after the producer stream sync (never under cache.mu); restore-fail after publish retires before rethrow; failed retire quarantines the pin.
This path does **not** restructure the Gemma-4 layer loop or enable decode hipGraph
(those stay lab-only until a CUDA token-exact gate can land them).

Contributor KEEP recipe (2x R9700 gfx1201, ROCm 7.2.4, `PREFIX_CACHE=0`, unique
pads, 2026-08-13): SharedK-WMMA on, FLASH/FMHA off, `VT_GEMMA4_PREFILL_GEMM_M=2048`
(the default), `VT_GEMMA4_PREFILL_PEER_ACT=1` (the default), batch MoE `T>=64`.
Fair median prefill **2014 t/s @~11k** and **1099 t/s @~42k**; stream decode
**55 t/s** temp=0. Paris / arith `63` / `gemma4` tool_calls held. Speculative,
ngram, FMHA, and layer-split are **out of this recipe**. Details:
[spec](../../.agents/specs/gemma4-rocm-fp8-moe.md).

These are contributor-lab numbers against **no denominator**: no pinned vLLM-ROCm
run on the same box, same model, same quantization and same request shape exists
for them, so `docs/BENCHMARKS.md` still records this backend as `PENDING: no
binding throughput number` and this recipe does not change that. The decode
figure is also not reproducible from the knobs above: the as-run recipe set four
further decode splits that no product code in this tree reads
([#845](https://github.com/mudler/vllm.cpp/issues/845)), and they are recorded in
the spec rather than here, because a recipe on this page has to be one a reader
can follow.
