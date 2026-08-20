# vllm.cpp status

This page records the current lifecycle state of the public product surfaces.
Use [Features](FEATURES.md) for supported capabilities and
[Benchmarks](BENCHMARKS.md) for measured performance. Model-specific commands,
checkpoints, and limitations are in the [model guides](models/README.md).

The internal matrices remain the detailed records for contributors:

- [Engine and serving matrix](../.agents/engine-matrix.md)
- [Model matrix](../.agents/model-matrix.md)
- [Backend matrix](../.agents/backend-matrix.md)
- [Kernel matrix](../.agents/kernel-matrix.md)
- [Feature matrix](../.agents/feature-matrix.md)

Git history, row specs, the [parity ledger](../.agents/parity-ledger.md), and
[completed state events](../.agents/completed/state-events/) retain the dated
implementation and verification record.

## Parity pin

The primary reference is vLLM 0.26.0.dev0 at `555967922`, with transformers
5.14.1. The [upstream sync record](../.agents/upstream-sync.md) owns the exact
pin, comparison scope, and changes since the previous reference.

Correctness claims use the pinned oracle and the gate named by the owning row.
Measurements against an older pin remain attributed in
[Benchmarks](BENCHMARKS.md); they do not become current measurements after a
pin advance.

## Capability status

The project uses these lifecycle terms:

| State | Meaning |
|---|---|
| Correctness-complete | The declared correctness gate passes against the pinned oracle |
| Speed-pending | Correctness passes, but one or more performance axes remain open |
| Partial | A usable path exists with named missing behavior or evidence |
| Build-only | The target compiles, but this project has no runtime proof on that hardware |
| Hardware-blocked | The required hardware is not currently available for the gate |
| Inventoried | The gap has a stable record but no accepted implementation |

### User-facing surfaces

| Surface | Current state | Open gate or limitation |
|---|---|---|
| Text generation | Correctness-complete on the gated model paths | Performance remains checkpoint-, model-, backend-, and concurrency-dependent |
| OpenAI server | Subset; v0.0.2 publishes eight server bundles; Windows v0.0.3-pre.1 pending | Some vLLM endpoints, pooling paths, and multimodal server paths remain incomplete |
| C ABI and C++ library | Available | The C ABI is the stable public embedding surface; internal C++ headers are not ABI-stable |
| Continuous batching, chunked prefill, prefix caching, and recompute | Available on the documented engine paths | Some hybrid-cache modes and scheduling policies still need broader gates |
| Quantized inference | Available for the formats and backends listed in Features | Block-wise FP8 matches the CPU reference on seven GB10 shapes, but has no model token gate or speed result; unsupported format/backend pairs refuse or fall back as documented (#1437) |
| Speculative decoding | Partial | DFlash2 safetensors drafts run the grouped convolution and the candidate selector, then refuse at the unimplemented path walk; GGUF DFlash2 drafts still refuse at startup (#1314) |
| Image, video, audio, speech, music, and diffusion models | Partial by model | LTX-2.5 validates declared checkpoint class; its video VAE convolution routes through `vt::Conv3d`, but the CUDA arm has not run on a GPU and other decode stages remain on the host (#1007, #1451, #1452). The DFR pipeline and its temporal rounds are gated on reduced-dimension fixtures only, because no `keyframe_slot_sft` base is published, and the unclamped tiling arm is ungated (#986, #1137, #1493) |
| Distributed execution | Inventoried or partial by lane | Tensor, pipeline, data, and expert parallel coverage is not a general shipped promise |
| LoRA and adapters | Partial | The standalone implementation is not a general server-integrated capability |

The [Features table](FEATURES.md) is the current keyed capability projection.
It lists each supported architecture, quantization format, backend, and serving
feature once. The [Usage index](USAGE.md) links each runnable workflow.

### Backends and releases

| Surface | Current state | Open gate or limitation |
|---|---|---|
| CPU | Correctness and continuous-integration reference | Performance depends on architecture and quantization; current comparisons are in Benchmarks |
| CUDA GB10 (`sm_121a`) | Runtime-gated | Other CUDA architectures can be build-supported without a runtime gate |
| CUDA Thor (`sm_110`) | Runtime proof exists for the documented portable path | Fast fp8, fp4, CUTLASS, Marlin, and FlashAttention paths are not implied |
| Metal | Partial runtime support on Apple Silicon | Only the operations and models listed in Features are covered |
| Vulkan | Partial runtime support | The documented OPT and Qwen gates define the proved scope |
| ROCm | Build and focused community test evidence | Model and oracle runtime gates remain open |
| Tenstorrent Blackhole | Active, partial runtime support | The full rerun and performance path remain open |
| Intel XPU | Hardware-blocked | No accepted runtime gate |

The eight v0.0.2 server bundles are published; Windows v0.0.3-pre.1 remains
pending. The
[release reference](RELEASES.md) owns artifact names, checksums, provenance,
and verification instructions. Windows ZIP downloads do not exist while the
hosted runtime, merged-SHA dry run, prerelease publication, and 32-asset audit
remain pending.
<!-- ENG-RELEASE-WINDOWS: state=ACTIVE publication=pending artifact=unpublished -->

## Not supported yet

The following areas do not have a general supported path:

- automatic memory sizing and a complete preflight memory bound;
- external LMCache-compatible KV transport as a fully gated deployment path;
- complete tensor, pipeline, expert, and data parallel execution;
- every vLLM model, endpoint, plugin group, and structured-output mode;
- every quantization format on every backend;
- runtime proof for every architecture that the build accepts;
- complete end-to-end multimodal serving for every registered multimodal model.

An absent item is not an implied refusal or promise. Check the
[Feature matrix](../.agents/feature-matrix.md) and
[Engine matrix](../.agents/engine-matrix.md) for inventoried work and its owner.
