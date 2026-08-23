# Gemma 4

Gemma 4 runs through the shared paths, so [the quickstart](../QUICKSTART.md) and
[the usage guide](../USAGE.md) cover starting a server and sending a request.

This page carries one thing that is specific to Gemma 4: an FP8 mixture-of-experts
recipe for a dual-GPU AMD RDNA4 host. Every knob it names is off by default, so a
default build on any other hardware is unaffected.

## FP8 on ROCm, RDNA4

Two features carry this path, and both are controlled by environment variables
documented in [`docs/ENVIRONMENT.md`](../ENVIRONMENT.md): dual-GPU resident FP8
MoE (`VT_GEMMA4_RESIDENT_*`) and SharedK-WMMA prefill (`VT_ATTN_*`). **The
defaults stay off outside RDNA4.**

The path does **not** restructure the Gemma 4 layer loop and does not enable
decode hipGraph. Both stay lab-only until a CUDA token-exact gate can land them.

### The contributor recipe

One recipe has been run, by a contributor, on 13 August 2026:

| Field | Value |
|---|---|
| Hardware | 2x R9700, `gfx1201` |
| ROCm | 7.2.4 |
| Prefix cache | `PREFIX_CACHE=0`, unique pads |
| Attention | SharedK-WMMA on, FLASH and FMHA off |
| `VT_GEMMA4_PREFILL_GEMM_M` | `2048`, which is the default |
| `VT_GEMMA4_PREFILL_PEER_ACT` | `1`, which is the default |
| Batch MoE | `T >= 64` |

| Axis | Result |
|---|---|
| Fair median prefill at about 11k | 2014 t/s |
| Fair median prefill at about 42k | 1099 t/s |
| Stream decode, `temperature` 0 | 55 t/s |

The Paris, arithmetic `63`, and `gemma4` tool-call checks all held. Speculative
decoding, ngram, FMHA, and layer-split are **out of this recipe**.

### What these numbers are not

**They have no denominator.** No pinned vLLM-ROCm run exists on the same box,
the same model, the same quantization, and the same request shape, so
[`docs/BENCHMARKS.md`](../BENCHMARKS.md) still records this backend as
`PENDING: no binding throughput number`. This recipe does not change that.

**The decode figure is not reproducible from the knobs above.** The recipe as run
set four further decode splits that no product code in this tree reads
([#845](https://github.com/mudler/vllm.cpp/issues/845)). They are recorded in the
spec rather than here, because a recipe on this page has to be one you can
follow.

## Implementation notes

These are kept here because they name seams a reader of the ROCm path will meet.
[The spec](../../.agents/specs/gemma4-rocm-fp8-moe.md) owns the design and the
evidence.

- `GetBlas` keeps two per-thread hipBLAS handles, `tls_slots[2]`, with device 1
  mapped to slot 1, so a hop from device 0 to device 1 does not destroy GPU 0's
  handle. `ProductGetBlasHandle` is the test accessor for that file-local
  `GetBlas`.
- The HIP live probe is a separate CTest target. It exits 77 when
  `HIP_VISIBLE_DEVICES` is empty. It enters capture, so the production
  `StreamIsCapturing` is load-bearing. It adds no environment variable.
- The prefill peer path ([#839](https://github.com/mudler/vllm.cpp/issues/839))
  unpins the dequant cache only after an observed retirement. A failed fill or
  ready lease is retired with `RetireFillLocked` after the producer stream sync,
  never under `cache.mu`. A restore failure after publish retires before it
  rethrows, and a failed retire quarantines the pin.
